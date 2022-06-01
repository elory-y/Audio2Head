from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d
from modules.util import Hourglass3D

from modules.util import gaussian2kp
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


class AudioModel3D(nn.Module):
    def __init__(self,seq_len, block_expansion,num_blocks, max_features,
                            num_kp, estimate_jacobian=True, estimate_kpvalue=True):
        super(AudioModel3D,self).__init__()
        # self.opt = opt
        self.seq_len = seq_len
        self.pad = 0
        self.num_kp = num_kp
        self.down_id = AntiAliasInterpolation2d(3,0.25)
        self.down_pose = AntiAliasInterpolation2d(seq_len,0.25)

        self.embedding = nn.Sequential(nn.ConvTranspose2d(1, 8, (29, 14), stride=(1, 1), padding=(0, 11)),
                                       BatchNorm2d(8),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(8, 2, (13, 13), stride=(1, 1), padding=(6, 6)))

        num_channels = 6
        self.predictor = Hourglass3D(block_expansion, in_features=num_channels,
                                       max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=self.num_kp, kernel_size=(7, 7, 7),
                            padding=(3,0,0))
        if estimate_jacobian:
            self.num_jacobian_maps = self.num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=(0,0))
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = 0.1
        self.estimate_kpvalue = estimate_kpvalue

    def forward(self, x):
        bs,_,_,c_dim = x["audio"].shape #[1,64,4,41]

        audio_embedding = self.embedding(x["audio"].reshape(-1,1,4,c_dim)) #[64,2,32,32]
        audio_embedding = F.interpolate(audio_embedding,scale_factor=2).reshape(bs,self.seq_len,2,64,64).permute(0,2,1,3,4) #[1,2,64,64,64]

        id_feature = self.down_id(x["id_img"]) #[1,3,64,64]
        pose_feature = self.down_pose(x["pose"]) #[1,64,64,64]

        embeddings = torch.cat([audio_embedding,id_feature.unsqueeze(2).repeat(1,1,self.seq_len,1,1),pose_feature.unsqueeze(1)],dim=1) #[1,6,64,64,64]

        feature_map = self.predictor(embeddings)
        feature_shape = feature_map.shape # [1,38,64,64,64]
        prediction = self.kp(feature_map).permute(0,2,1,3,4) #[1,64,10,58,58]
        prediction = prediction.reshape(-1,prediction.shape[2],prediction.shape[3],prediction.shape[4]) #[64,10,58,58]
        final_shape = prediction.shape  #[64,10,58,58]
        heatmap = prediction.view(final_shape[0], final_shape[1], -1) #[64,10,3364]
        heatmap = F.softmax(heatmap / self.temperature, dim=2) #[64,10,3364]
        heatmap = heatmap.view(*final_shape)# #[64,10,58,58]
        if self.estimate_kpvalue:
            out = gaussian2kp(heatmap)#得到value，[64,10,2]
            out["value"] = out["value"].reshape(-1,self.seq_len,self.num_kp,2) #[1,64,10,2]
        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map.permute(0,2,1,3,4).reshape(-1, feature_shape[1],feature_shape[3],feature_shape[4]))

            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            out["jacobian_map"] = jacobian_map
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            out['jacobian'] = jacobian.reshape(-1,self.seq_len,self.num_kp,2,2)

        out["pred_fature"] = prediction
        return out

