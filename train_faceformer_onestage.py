from datasets.online_KPdataset import KeyPoint_Data, KeyPoint_PaddleAudioData
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from datasets.online_KPdataset import FaceformerData
from modules.audio2ky_faceformer import Faceformer
import yaml
import argparse
import cv2
import numpy as np
import random
from skimage import io, img_as_float32
from modules.keypoint_detector import KPDetector
import os
import wandb
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
wandb.init(entity="priv", project="faceformer", name="paddle_faceformer_2e-4_batch256_feature_dim64_layer1")

def preprocess(mp4_paths, star_frame, kp_detector, pad, frames=96, device='cuda'):
    kpvalues = []
    kpjacobians = []
    paddings = []
    # print(mp4_paths, star_frame)
    for number in range(0, len(mp4_paths)):
        num_frame = 0
        kpvalue = []
        kpjacobian = []
        kppred_fature = []
        # 读取视频第1帧，预测所有kp、jacobian和特征图
        cap = cv2.VideoCapture(mp4_paths[number])
        while cap.isOpened():
            success, get_img = cap.read()
            if success:
                if num_frame in range(star_frame[number].item(), star_frame[number].item() + frames):
                    get_img = get_img[..., ::-1]
                    get_img = cv2.resize(get_img, (256, 256))
                    get_img = np.array(img_as_float32(get_img))
                    get_img = get_img.transpose((2, 0, 1))
                    get_img = torch.from_numpy(get_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        kp = kp_detector(get_img)
                        kpvalue.append(kp["value"])
                        kpjacobian.append(kp["jacobian"])
                        kppred_fature.append(kp["pred_fature"])

                if num_frame > star_frame[number].item() + frames:
                    break
            else:
                break
            num_frame += 1

        # 对所有视频pad，如果pad=0就默认补长度0，不需要额外进行判断
        kpvalue_pad = torch.zeros(pad[number], 10, 2).to(device)
        kpjacobian_pad = torch.zeros([pad[number], 10, 2, 2]).to(device)
        kpvalue = torch.cat((torch.stack(kpvalue, dim=1).squeeze(0), kpvalue_pad))
        kpjacobian = torch.cat((torch.stack(kpjacobian, dim=1).squeeze(0), kpjacobian_pad))
        cap.release()
        # padding index，用于在算损失时仅计算非pad区域
        pad_index = torch.cat((torch.ones(frames-pad[number]), torch.zeros(pad[number]))).to(device)
        kpvalues.append(kpvalue)
        kpjacobians.append(kpjacobian)
        paddings.append(pad_index)

    kpvalues = torch.stack(kpvalues, dim=0)
    kpjacobians = torch.stack(kpjacobians, dim=0)
    paddings = torch.stack(paddings, dim=0)
    return kpvalues, kpjacobians, paddings

def calculate_loss(true_kpvalues, true_kpjacobians, infer_kpvalues, infer_kpjacobians, paddings, loss_function):
    kp_loss = loss_function(true_kpvalues, infer_kpvalues)
    jacobian_loss = loss_function(true_kpjacobians, infer_kpjacobians)

    # TODO: 我不确认paddings这个对不对，如果报错的话，把上面三个和paddings的特征维度告诉我下
    total_frames = paddings.sum()
    kp_loss = (kp_loss.flatten(2).mean(-1) * paddings).sum() / total_frames
    jacobian_loss = (jacobian_loss.flatten(2).mean(-1) * paddings).sum() / total_frames
    loss = 10 * kp_loss + 10 * jacobian_loss
    return kp_loss, jacobian_loss, loss

def wand_curve(kp_loss, jacobian_loss, loss, iteration, istrain):
    phase = 'train' if istrain else 'test'
    log_dict = {
        f'{phase}_kp_loss': kp_loss,
        f'{phase}jacobian_loss': jacobian_loss,
        f'{phase}_loss': loss
    }
    wandb.log(log_dict, step=iteration)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kp_detector = KPDetector(block_expansion=args.kp_dete_block_expansion, num_kp=args.num_kp, num_channels=args.num_channels,
                             max_features=args.kp_dete_max_features, num_blocks=args.kp_dete_num_blocks, temperature=args.kp_dete_temperature,
                             estimate_jacobian=args.estimate_jacobian, scale_factor=args.kp_dete_scale_factor)
    kp_detector = kp_detector.to(device)
    # model_path = "./checkpoints/audio2head.pth.tar"
    # foom = torch.load("/home/user/Database/fomm/log1/00000012-checkpoint.pth.tar")
    checkpoint = torch.load(args.model_path)
    kp_detector.load_state_dict(checkpoint["kp_detector"])
    kp_detector.eval()
    train_dataset = FaceformerData(root_dir=args.train_datapath, frames=args.frames, model_path=args.model_path, pad_feature_root=os.path.join(args.pad_feature_root, "audio_train_wav16_feature"), istrain=True)
    test_dataset = FaceformerData(root_dir=args.test_datapath, frames=args.frames, model_path=args.model_path, pad_feature_root=os.path.join(args.pad_feature_root, "audio_test_wav16_feature"), istrain=False)
    train_data = DataLoader(train_dataset, batch_size=args.train_batch_size,
                            shuffle=True, num_workers=0)
    test_data = DataLoader(test_dataset, batch_size=args.test_batch_size,
                           shuffle=True, num_workers=0)
    audio2kp = Faceformer(args, device=device).to(device)
    faceformer_check = torch.load("/home/ssd2/suimang/project/checkpoint/faceformer_audio_64_1/2e-4_1_20.18463.pth")
    model_dict = audio2kp.state_dict()
    pretraind_dic = {k: v for k, v in faceformer_check.items() if
                     k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretraind_dic)
    audio2kp.load_state_dict(model_dict)
    loss_function = nn.L1Loss(reduction='none')
    optimizer = torch.optim.Adam([{"params": audio2kp.parameters(), "initial_lr": 2e-4}], lr=args.lr)
    optimizer = torch.optim.Adam(audio2kp.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dataset), last_epoch=12)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    train_iteration = 12
    for epoch in range(args.epochs):
        audio2kp.train()
        for i, dic in enumerate(train_data):
            mp4_paths, audio_feature, poses, pad, star_frame = dic["mp4_path"], dic["audio_features"], dic["poses"], dic["pad"], dic["star_frame"]
            kpvalues, kpjacobians, paddings = preprocess(mp4_paths=mp4_paths, star_frame=star_frame, kp_detector=kp_detector, pad=pad)
            infer_kpvalue, infer_jacobian = audio2kp(audio_tensor=audio_feature.squeeze(1).to(device), kp=kpvalues, jac=kpjacobians, teacher_forcing=True)
            train_iteration += 1
            kp_loss, jacobian_loss, loss = calculate_loss(kpvalues, kpjacobians, infer_kpvalue, infer_jacobian, paddings, loss_function)
            optimizer.zero_grad()
            loss.backward()
            wand_curve(kp_loss.item(), jacobian_loss.item(), loss.item(),
                       train_iteration, istrain=True)
            optimizer.step()
            print(loss.item(), train_iteration)
        audio2kp.eval()
        test_loss = 0
        test_kp_loss = 0
        test_jacobian_loss = 0
        num = 0
        with torch.no_grad():
            for i, dic in enumerate(test_data):
                mp4_paths, audio_feature, poses, pad, star_frame = dic["mp4_path"], dic["audio_features"], dic["poses"], \
                                                                   dic["pad"], dic["star_frame"]
                kpvalues, kpjacobians, paddings = preprocess(mp4_paths=mp4_paths, star_frame=star_frame,
                                                             kp_detector=kp_detector, pad=pad)

                infer_kpvalue, infer_jacobian = audio2kp(audio_tensor=audio_feature.squeeze(1).to(device), kp=kpvalues, jac=kpjacobians, teacher_forcing=False)
                kp_loss, jacobian_loss, loss = calculate_loss(kpvalues, kpjacobians, infer_kpvalue, infer_jacobian, paddings, loss_function)
                test_kp_loss += kp_loss.item()
                test_jacobian_loss += jacobian_loss.item()
                test_loss += loss.item()
                num += 1
        print(test_kp_loss/num, test_jacobian_loss/num, test_loss/num, test_loss/num)
        wand_curve(test_kp_loss/num, test_jacobian_loss/num, test_loss/num, train_iteration, istrain=False)
        torch.save(audio2kp.state_dict(), os.path.join("/home/ssd2/suimang/project/checkpoint/faceformer_audio_64_1", '2e-4_%s_%.5f.pth' % (epoch, test_loss/num)))
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default=96)
    parser.add_argument("--paddle_audio", default=True)
    parser.add_argument("--lr", default=2.0e-4)
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--test_batch_size", default=64, type=int)
    parser.add_argument("--model_path", default=r"/home/ssd1/Database/audio2head/audio2head.pth.tar", help="pretrained model path")
    parser.add_argument("--train_datapath", default=r"/home/ssd2/suimang/Database/girl_data/onestage_data/audio_data_girl/audio_train")
    parser.add_argument("--test_datapath", default=r"/home/ssd2/suimang/Database/girl_data/onestage_data/audio_data_girl/audio_test")
    parser.add_argument("--pad_feature_root", default=r"/home/ssd2/suimang/Database/girl_data/onestage_data/audio_data_girl")
    parser.add_argument("--epochs", default=200)
    parser.add_argument("--config", default="./config/parameters.yaml")
    parser.add_argument("--seq_len", default=64)
    parser.add_argument("--num_kp", default=10)
    parser.add_argument("--vertice_dim", type=int, default=60)
    parser.add_argument("--period", type=int, default=30)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--estimate_jacobian", default=True)
    parser.add_argument("--kp_dete_block_expansion", default=32)
    parser.add_argument("--num_channels", default=3)
    parser.add_argument("--kp_dete_max_features", default=1024)
    parser.add_argument("--kp_dete_scale_factor", default=0.25)
    parser.add_argument("--kp_dete_num_blocks", default=5)
    parser.add_argument("--kp_dete_temperature", default=0.1)
    args = parser.parse_args()
    main(args)