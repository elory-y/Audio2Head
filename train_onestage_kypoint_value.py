from datasets.online_KPdataset import KeyPoint_Data, KeyPoint_PaddleAudioData
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from modules.audio2kp import AudioModel3D
from modules.audio2kp_pad import AudioModel3d_pad
import yaml
import argparse
import cv2
import numpy as np
import random
from skimage import io, img_as_float32
from modules.keypoint_detector import KPDetector
import os
import wandb


wandb.init(entity="suimang", project="kypoint_value", name="v1_2e-4")

def preprocess(mp4_paths, star_frame, kp_detector, pad, frames=64, device='cuda'):
    imgs = []
    kpvalues = []
    kpjacobians = []
    kpjacobian_maps = []
    paddings = []
    # print(mp4_paths, star_frame)
    for number in range(0, len(mp4_paths)):
        num_frame = 0
        kpvalue = []
        kpjacobian = []
        kpjacobian_map = []
        # 读取视频第1帧，预测所有kp、jacobian和特征图
        cap = cv2.VideoCapture(mp4_paths[number])
        while cap.isOpened():
            success, get_img = cap.read()
            if success:
                if num_frame in range(star_frame[number].item(), star_frame[number].item() + frames):
                    get_img = cv2.resize(get_img, (256, 256))
                    get_img = np.array(img_as_float32(get_img))
                    get_img = get_img.transpose((2, 0, 1))
                    get_img = torch.from_numpy(get_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        kp = kp_detector(get_img)
                        kpvalue.append(kp["value"])
                        kpjacobian.append(kp["jacobian"])
                        kpjacobian_map.append(kp["jacobian_map"])
                    if num_frame == star_frame[number].item():
                        imgs.append(get_img)
                if num_frame > star_frame[number].item() + frames:
                    break
            else:
                break
            num_frame += 1

        # 对所有视频pad，如果pad=0就默认补长度0，不需要额外进行判断
        kpvalue_pad = torch.zeros(pad[number], 10, 2).to(device)
        # kpjacobian_pad = torch.zeros([pad[number], 10, 2, 2]).to(device)
        # kpjacobian_map_map = torch.zeros([pad[number], 10, 4, 58, 58]).to(device)
        kpvalue = torch.cat((torch.stack(kpvalue, dim=1).squeeze(0), kpvalue_pad))
        # kpjacobian = torch.cat((torch.stack(kpjacobian, dim=1).squeeze(0), kpjacobian_pad))
        # kpjacobian_map = torch.cat((torch.stack(kpjacobian_map, dim=1).squeeze(0), kpjacobian_map_map))
        cap.release()
        # padding index，用于在算损失时仅计算非pad区域
        pad_index = torch.cat((torch.ones(frames-pad[number]), torch.zeros(pad[number]))).to(device)

        kpvalues.append(kpvalue)
        # kpjacobians.append(kpjacobian)
        # kpjacobian_maps.append(kpjacobian_map)
        paddings.append(pad_index)

    kpvalues = torch.stack(kpvalues, dim=0)
    # kpjacobians = torch.stack(kpjacobians, dim=0)
    # kpjacobian_maps = torch.stack(kpjacobian_maps, dim=0)
    # lab_kpjacobian_map = kpjacobian_maps.view(-1, 10, 4, 58, 58)
    paddings = torch.stack(paddings, dim=0)
    imgs = torch.cat(imgs)
    # return kpvalues, kpjacobians, kpjacobian_maps, lab_kpjacobian_map, imgs, paddings
    return kpvalues, imgs, paddings
def calculate_loss(kpvalues, gen_kp, paddings, loss_function, interation, istrain):
    kp_loss = loss_function(kpvalues, gen_kp["value"])
    # jacobian_loss = loss_function(kpjacobians, gen_kp["jacobian"])
    # jacobian_map_loss = loss_function(lab_kpjacobian_map, gen_kp["jacobian_map"])

    # TODO: 我不确认paddings这个对不对，如果报错的话，把上面三个和paddings的特征维度告诉我下
    total_frames = paddings.sum()
    kp_loss = (kp_loss.flatten(2).mean(-1) * paddings).sum() / total_frames
    # jacobian_loss = (jacobian_loss.flatten(2).mean(-1) * paddings).sum() / total_frames
    # jacobian_map_loss = (jacobian_map_loss.flatten(1).mean(-1) * paddings.view(-1)).sum() / total_frames
    # loss = 10 * kp_loss + 10 * jacobian_loss + jacobian_map_loss
    loss = 100 * kp_loss
    if istrain == True:
        wandb.log({"train_kp_loss": 100*kp_loss.item()}, step=interation)
        # wandb.log({"train_jacobian_loss": 10 * jacobian_loss.item()}, step=interation)
        # wandb.log({"train_jacobian_map_loss": jacobian_map_loss.item()}, step=interation)
        # wandb.log({"train_total_loss": loss.item()}, step=interation)
        # with open("train_1oss.txt", "a+") as f:
        #     f.write("train_kp_loss %s train_jacobian_loss %s " % (10 * kp_loss.item(), 10 * jacobian_loss.item()) + "\n")
        # print("train_kp_loss %s train_jacobian_loss %s  totalloss %s" % (10 * kp_loss.item(), 10 * jacobian_loss.item(), loss.item()))
    else:
        wandb.log({"test_kp_loss": 100 * kp_loss.item()}, step=interation)
        # wandb.log({"test_jacobian_loss": 10 * jacobian_loss.item()}, step=interation)
        # wandb.log({"test_jacobian_map_loss": jacobian_loss.item()}, step=interation)
        # wandb.log({"test_total_loss": loss.item()}, step=interation)
    #
    #     # with open("test_loss.txt", "a+") as f:
    #     #     f.write("test_kp_loss %s test_jacobian_loss %s " % (10 * kp_loss.item(), 10 * jacobian_loss.item()) + "\n")
    #     # print("test_kp_loss %s test_jacobian_loss %s jacobian_map_loss %s totalloss %s" % (10 * kp_loss.item(), 10 * jacobian_loss.item(), jacobian_map_loss.item(), loss.item()))
    return loss

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_file = r"./config/vox-256.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)
    # kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
    #                          **config['model_params']['common_params'])
    # kp_detector = KPDetector(block_expansion=32, num_kp=10, num_channels=3, max_features=1024, num_blocks=5, temperature=0.1,estimate_jacobian=True, scale_factor=0.25)
    kp_detector = KPDetector(block_expansion=args.kp_dete_block_expansion, num_kp=args.num_kp, num_channels=args.num_channels,
                             max_features=args.kp_dete_max_features, num_blocks=args.kp_dete_num_blocks, temperature=args.kp_dete_temperature,
                             estimate_jacobian=args.estimate_jacobian, scale_factor=args.kp_dete_scale_factor)
    kp_detector = kp_detector.to(device)
    # model_path = "./checkpoints/audio2head.pth.tar"
    # foom = torch.load("/home/user/Database/fomm/log1/00000012-checkpoint.pth.tar")
    checkpoint = torch.load(args.model_path)
    kp_detector.load_state_dict(checkpoint["kp_detector"])
    kp_detector.eval()
    if args.paddle_audio:
        train_dataset = KeyPoint_PaddleAudioData(root_dir=args.train_datapath, frames=64, model_path=args.model_path, pad_feature_root=args.pad_feature_root)
        test_dataset = KeyPoint_PaddleAudioData(root_dir=args.test_datapath, frames=64, model_path=args.model_path, pad_feature_root=args.pad_feature_root)
        train_data = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=0)
        test_data = DataLoader(test_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=0)
        audio2kp = AudioModel3d_pad(seq_len=args.seq_len, block_expansion=args.AudioModel_block_expansion,
                                num_blocks=args.AudioModel_num_blocks, max_features=args.AudioModel_max_features,
                                num_kp=args.num_kp).to(device)
        # train_check = torch.load("/home/ssd2/suimang/project/checkpoint/audio_check/2e-7—s_10_0.31557.pth")
        # audio2kp.load_state_dict(train_check)
        model_dict = audio2kp.state_dict()
        pretraind_dic = {k: v for k, v in checkpoint["audio2kp"].items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretraind_dic)
        audio2kp.load_state_dict(model_dict)
    else:
        train_dataset = KeyPoint_Data(root_dir=args.train_datapath, frames=64, model_path=args.model_path)
        test_dataset = KeyPoint_Data(root_dir=args.test_datapath, frames=64, model_path=args.model_path)
        train_data = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=0)
        test_data = DataLoader(test_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=0)
        audio2kp = AudioModel3D(seq_len=args.seq_len, block_expansion=args.AudioModel_block_expansion, num_blocks=args.AudioModel_num_blocks, max_features=args.AudioModel_max_features, num_kp=args.num_kp, estimate_jacobian=False).to(device)
        train_check = torch.load("/home/user/Database/audio_data_girl/use_checkpoint/girl_onestage.pth")
        # audio2kp.load_state_dict(checkpoint["audio2kp"])
        model_dict = audio2kp.state_dict()
        pretraind_dic = {k: v for k, v in train_check.items() if
                         k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretraind_dic)
        audio2kp.load_state_dict(model_dict)

    loss_function = nn.L1Loss(reduction='none')
    optimizer = torch.optim.Adam(audio2kp.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dataset))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    # log_writer = SummaryWriter()
    train_interation = 0
    test_interation = 0
    for epoch in range(args.epochs):
        audio2kp.train()
        for i, dic in enumerate(train_data):
            mp4_paths, audio_feature, poses, pad, star_frame = dic["mp4_path"], dic["audio_features"], dic["poses"], dic["pad"], dic["star_frame"]
            # kpvalues, kpjacobians, kpjacobian_maps, lab_kpjacobian_map, imgs, paddings = preprocess(mp4_paths=mp4_paths, star_frame=star_frame, kp_detector=kp_detector, pad=pad)
            kpvalues, imgs, paddings = preprocess(mp4_paths=mp4_paths, star_frame=star_frame, kp_detector=kp_detector, pad=pad)
            t = {}
            t["audio"] = audio_feature.type(torch.FloatTensor).to(device)
            t["pose"] = poses.type(torch.FloatTensor).to(device)
            t["id_img"] = imgs.to(device)
            gen_kp = audio2kp(t)
            train_interation += 1
            loss = calculate_loss(kpvalues, gen_kp, paddings, loss_function, train_interation, istrain=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item(), train_interation)


        audio2kp.eval()
        test_loss = 0
        num = 0
        with torch.no_grad():
            for i, dic in enumerate(test_data):
                mp4_paths, audio_feature, poses, pad, star_frame = dic["mp4_path"], dic["audio_features"], dic["poses"], \
                                                                   dic["pad"], dic["star_frame"]
                kpvalues,imgs, paddings = preprocess(mp4_paths=mp4_paths, star_frame=star_frame, kp_detector=kp_detector, pad=pad)
                t = {}
                test_interation += 1
                t["audio"] = audio_feature.type(torch.FloatTensor).to(device)
                t["pose"] = poses.type(torch.FloatTensor).to(device)
                t["id_img"] = imgs.to(device)
                gen_kp = audio2kp(t)
                loss = calculate_loss(kpvalues, gen_kp, paddings, loss_function, test_interation, istrain=False)
                test_loss += loss.item()
                num += 1
                print("test", loss.item())
        torch.save(audio2kp.state_dict(), os.path.join("/home/user/Database/audio_data_girl/girl_kpvalue", '2e-4_kypoint_%s_%.5f.pth' % (epoch, test_loss/num)))
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default=64)
    parser.add_argument("--paddle_audio", default=False)
    parser.add_argument("--lr", default=2.0e-4)
    parser.add_argument("--batch_size", default=6)
    parser.add_argument("--model_path", default=r"./checkpoint/audio2head.pth.tar", help="pretrained model path")
    parser.add_argument("--train_datapath", default=r"/home/user/Database/audio_data_girl/audio_train")
    parser.add_argument("--test_datapath", default=r"/home/user/Database/audio_data_girl/audio_test")
    parser.add_argument("--pad_feature_root", default=r"/home/user/Database/audio_data_girl/audio_train")
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
    main(args)