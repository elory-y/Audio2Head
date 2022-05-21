import sys
sys.path.append('..')
sys.path.append('../modules')
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from modules.audio2pose import get_pose_from_audio, get_audio_feature_from_audio
import os
import glob
from skimage import io, img_as_float32
import cv2
import numpy as np
import torch
from scipy.io import wavfile
import yaml
from modules.keypoint_detector import KPDetector


config_file = r"./config/vox-256.yaml"
with open(config_file) as f:
    config = yaml.safe_load(f)
kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                         **config['model_params']['common_params'])
kp_detector = kp_detector.cuda()
model_path = "./checkpoints/audio2head.pth.tar"
checkpoint  = torch.load(model_path)
kp_detector.load_state_dict(checkpoint["kp_detector"])
kp_detector.eval()
def save_npy(mp4_path,save_path):
    model_path = "./checkpoints/audio2head.pth.tar"
    audio_path = mp4_path.split(".")[0] + ".wav"
    name = mp4_path.split("/")[-1].split(".")[0]
    #存音频特征
    audio_features = get_audio_feature_from_audio(audio_path)
    np.save(os.path.join(save_path,"audio_features"+"_"+name+".npy"),audio_features)
    cap = cv2.VideoCapture(mp4_path)
    keypoint_value = []
    keypoint_jacobian = []
    keypoint_jacobian_map = []
    keypoint_pred_fature = []
    num_frame = 0
    while cap.isOpened():
        success, frames = cap.read()
        if success:
            frames = cv2.resize(frames, (256, 256))
            frames = np.array(img_as_float32(frames))
            frames = frames.transpose((2, 0, 1))
            frames = torch.from_numpy(frames).unsqueeze(0).cuda()
            #存头部姿态，用自带的模型,
            if num_frame == 0:
                re, tra = get_pose_from_audio(frames, audio_features, model_path)
                poses = np.concatenate((re, tra), axis=1)
                np.save(os.path.join(save_path,"poses"+"_"+name+'.npy'),poses)
            #存每帧图片存关键点
            kp = kp_detector(frames)
            keypoint_value.append(kp["value"].cpu().detach().numpy())
            keypoint_jacobian.append(kp["jacobian"].cpu().detach().numpy())
            keypoint_jacobian_map.append(kp["jacobian_map"].cpu().detach().numpy())
            keypoint_pred_fature.append(kp["pred_fature"].cpu().detach().numpy())
            num_frame += 1
        else:
            break
    # keypoint = np.array(keypoint)
    np.save(os.path.join(save_path,"kpvalue"+"_"+name+".npy"), keypoint_value)
    np.save(os.path.join(save_path, "kpjacobian" + "_" + name + ".npy"), keypoint_value)
    np.save(os.path.join(save_path, "jacobian_map" + "_" + name + ".npy"), keypoint_value)
    np.save(os.path.join(save_path, "keypoint_pred_fature" + "_" + name + ".npy"), keypoint_value)
if __name__ == "__main__":
    mp4_lis = glob.glob("/home/caopu/workspace/Audio2Head/test_data/*.mp4")
    save_path = "/home/caopu/workspace/Audio2Head/test_data"
    # save_npy("/home/caopu/workspace/Audio2Head/test_data/5.mp4")
    for mp4 in mp4_lis:
        save_npy(mp4, save_path)