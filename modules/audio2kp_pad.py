from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d
from modules.util import Hourglass3D

from modules.util import gaussian2kp
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


class AudioModel3d_pad(nn.Module):
    def __init__(self, seq_len, block_expansion, num_blocks, max_features,
                 num_kp, estimate_jacobian=True):
        super(AudioModel3d_pad, self).__init__()
        # self.opt = opt
        self.seq_len = seq_len
        self.pad = 0
        self.num_kp = num_kp
        self.down_id = AntiAliasInterpolation2d(3, 0.25)
        self.down_pose = AntiAliasInterpolation2d(seq_len, 0.25)
        # self.embedding = nn.Sequential(nn.ConvTranspose2d(1, 8, (29, 14), stride=(1, 1), padding=(0, 11)),
        #                                BatchNorm2d(8),
        #                                nn.ReLU(inplace=True),
        #                                nn.Conv2d(8, 2, (13, 13), stride=(1, 1), padding=(6, 6)))

        num_channels = 5
        self.predictor = Hourglass3D(block_expansion, in_features=num_channels,
                                     max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=self.num_kp, kernel_size=(7, 7, 7),
                            padding=(3, 0, 0))
        if estimate_jacobian:
            self.num_jacobian_maps = self.num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=(0, 0))
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None
        self.temperature = 0.1

    def forward(self, x):
        bs, _, time, dim = x["audio"].shape
        #输入的[2,86,2048]特征，先permute变为[2,2048,86]，对86通过线性层变为【2，2048，256】维度，然后还原回【2，256，2048】
        #todo第一步的【2，86，2048】插值到【2，64，2048】,a1 = torch.nn.functional.interpolate(a.permute(0,2,1), (64)),reshape(2)
        """
        a- =【2，86，2048】
        a1 = torch.nn.functional.interpolate(a.permute(0,2,1), (64))
        a1.shape
        Out[88]: torch.Size([2, 2048, 64])
        a2 = a1.reshape(2,32,64,64)最后的64代表帧数
        a3 = a2.permute(0,3,1,2)
        a3.shape
        Out[91]: torch.Size([2, 64, 32, 64])然后对32进行插值，接下来反转在cat
        a4 = torch.nn.functional.interpolate(a3,(64,64))
        a4.shape
        """
        x["audio"] = F.interpolate(x["audio"].squeeze(1).permute(0, 2, 1), 64).reshape(bs, 32, 64, 64).permute(0,3,1,2)
        audio_embedding_zheng = F.interpolate(x["audio"], (64, 64)).unsqueeze(1)
        audio_embedding_fan = torch.flip(audio_embedding_zheng, [2, 3])#对后两个维度取反
        audio_embedding = torch.cat((audio_embedding_zheng, audio_embedding_fan), dim=1)
        id_feature = self.down_id(x["id_img"])  # [1,3,64,64]
        pose_feature = self.down_pose(x["pose"])
        embeddings = torch.cat(
            [audio_embedding, id_feature.unsqueeze(2).repeat(1, 1, self.seq_len, 1, 1), pose_feature.unsqueeze(1)],
            dim=1)  # [1,6,64,64,64]
        feature_map = self.predictor(embeddings)
        feature_shape = feature_map.shape  # [1,38,64,64,64]
        prediction = self.kp(feature_map).permute(0, 2, 1, 3, 4)  # [1,64,10,58,58]
        prediction = prediction.reshape(-1, prediction.shape[2], prediction.shape[3],
                                        prediction.shape[4])  # [64,10,58,58]
        final_shape = prediction.shape  # [64,10,58,58]
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)  # [64,10,3364]
        heatmap = F.softmax(heatmap / self.temperature, dim=2)  # [64,10,3364]
        heatmap = heatmap.view(*final_shape)  # #[64,10,58,58]

        out = gaussian2kp(heatmap)  # 得到value，[64,10,2]
        out["value"] = out["value"].reshape(-1, self.seq_len, self.num_kp, 2)  # [1,64,10,2]
        if self.jacobian is not None:
            jacobian_map = self.jacobian(
                feature_map.permute(0, 2, 1, 3, 4).reshape(-1, feature_shape[1], feature_shape[3], feature_shape[4]))

            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            out["jacobian_map"] = jacobian_map
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            out['jacobian'] = jacobian.reshape(-1, self.seq_len, self.num_kp, 2, 2)

        out["pred_fature"] = prediction
        return out




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default=64)
    parser.add_argument("--lr", default=1.0e-2)
    parser.add_argument("--batch_size", default=6)
    parser.add_argument("--model_path", default=r"./checkpoint/audio2head.pth.tar", help="pretrained model path")
    parser.add_argument("--train_datapath", default=r"./data/audio_data_girl/audio_train")
    parser.add_argument("--test_datapath", default=r"./data/audio_data_girl/audio_test")
    parser.add_argument("--epochs", default=200)
    parser.add_argument("--config", default="./config/parameters.yaml")
    parser.add_argument("--seq_len", default=64)
    parser.add_argument("--num_kp", default=10)
    parser.add_argument("--AudioModel_num_blocks", default=5, help="AudioModel3D model num_blocks")
    parser.add_argument("--AudioModel_max_features", default=512, help="AudioModel3D model max_features")
    parser.add_argument("--estimate_jacobian", default=True)
    parser.add_argument("--AudioModel_block_expansion", default=32, help="AudioModel3D model block_expansion")
    parser.add_argument("--kp_dete_block_expansion", default=32)
    parser.add_argument("--num_channels", default=3)
    parser.add_argument("--kp_dete_max_features", default=1024)
    parser.add_argument("--kp_dete_scale_factor", default=0.25)
    parser.add_argument("--kp_dete_num_blocks", default=5)
    parser.add_argument("--kp_dete_temperature", default=0.1)
    args = parser.parse_args()
    x = {}
    x["audio"] = torch.rand([2,1,86,2048]).cuda()#(bs,frame*4*3)
    x["id_img"] = torch.rand([2,3,256,256]).cuda()
    x["pose"] = torch.rand([2,64,256,256]).cuda()
    model = AudioModel3d_pad(seq_len=args.seq_len, block_expansion=args.AudioModel_block_expansion, num_blocks=args.AudioModel_num_blocks, max_features=args.AudioModel_max_features, num_kp=args.num_kp).cuda()
    check = torch.load("/home/ssd2/suimang/project/checkpoint/audio_check/2e-6_19_0.32327.pth")
    mode_dict = model.state_dict()
    pretraind_dic = {k: v for k, v in check.items() if k in mode_dict and mode_dict[k].shape == v.shape}
    mode_dict.update(pretraind_dic)
    model.load_state_dict(mode_dict)
    a = model(x)