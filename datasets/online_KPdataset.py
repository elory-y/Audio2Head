import sys
sys.path.append('..')
sys.path.append('../modules')
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from modules.audio2pose import get_pose_from_audio
import os
import glob
from skimage import io, img_as_float32
import cv2
import numpy as np
import torch
from scipy.io import wavfile
import yaml
from modules.keypoint_detector import KPDetector
import random
from moviepy.editor import *




def draw_annotation_box(image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
    """Draw a 3D box as annotation of pose"""

    camera_matrix = np.array(
        [[233.333, 0, 128],
         [0, 233.333, 128],
         [0, 0, 1]], dtype="double")

    dist_coeefs = np.zeros((4, 1))

    point_3d = []
    rear_size = 75
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeefs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

class KeyPoint_Data(Dataset):
    def __init__(self, root_dir, frames, model_path,):
        self.root_dir = root_dir
        self.mp4_lis = glob.glob(os.path.join(self.root_dir, "*.mp4"))
        self.frames = frames
        self.model_path = model_path


    def __getitem__(self, item):
        mp4_path = self.mp4_lis[item]
        name = os.path.basename(mp4_path).split(".")[0]
        audiofeatures_path = os.path.join(self.root_dir, "audio_features_" + name+".npy")
        pose_path = os.path.join(self.root_dir, "pose_" + name +".npy")
        # 存音频特征
        audio_feature = np.load(audiofeatures_path)
        poses = np.load(pose_path)
        re = poses[:, :3]
        tra = poses[:, 3:]
        dic={}
        dic["mp4_path"] = mp4_path
        audio_frames = len(audio_feature) // 4
        pose_frames = poses.shape[0]
        frames = min(audio_frames, pose_frames)
        if frames > self.frames + 1:
            star_frame = random.randint(0, max(0, frames - self.frames-2))
            end_frame = star_frame + self.frames
            audio_feature = audio_feature[star_frame * 4: end_frame * 4, :]
            assert audio_feature.shape == (256, 41)
            re = re[star_frame : end_frame , :]
            tra = tra[star_frame: end_frame, :]
            total_poses = []
            audio_features = []
            for i in range(64):
                audio_features.append(audio_feature[(i) * 4:(i) * 4 + 4, :])
                trans = tra[i, :]
                rot = re[i, :]
                pose = np.zeros([256, 256])
                draw_annotation_box(pose, np.array(rot), np.array(trans))
                total_poses.append(pose)
            dic["star_frame"] = star_frame
            dic["pad"] = 0
        else:
            star_frame = 0
            end_frame = frames
            pad_frame = self.frames - end_frame
            pad_audio = torch.zeros((pad_frame * 4, 41))
            audio_feature = np.concatenate((audio_feature[star_frame * 4:end_frame * 4,:],pad_audio))
            total_poses = []
            audio_features = []
            pad_pose = torch.zeros((pad_frame, 3))
            re = np.concatenate((re[star_frame:end_frame, :], pad_pose))
            tra = np.concatenate((tra[star_frame:end_frame, :], pad_pose))
            for i in range(re.shape[0]):
                audio_features.append(audio_feature[(i) * 4:(i) * 4 + 4, :])
                trans = tra[i, :]
                rot = re[i, :]
                pose = np.zeros([256, 256])
                draw_annotation_box(pose, np.array(rot), np.array(trans))
                total_poses.append(pose)
            padd_vec = pad_frame
            dic["pad"] = padd_vec
            dic["star_frame"] = star_frame
        dic["audio_features"] = torch.from_numpy(np.array(audio_features))
        dic["poses"] = torch.from_numpy(np.array(total_poses))

        return dic

    def __len__(self):
        return len(os.listdir(self.root_dir)) // 4

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = KeyPoint_Data(root_dir="/home/caopu/workspace/Audio2Head/data/test", frames=64, model_path="./checkpoints/audio2head.pth.tar")
    data = DataLoader(dataset, batch_size=2,
                      shuffle=True, num_workers=0)

    for number, dic in enumerate(data):
        for i in dic:
            print(i)
            print(len(dic))
            mp4_path, audio_feature, poses, pad, = dic["mp4_path"], dic["audio_features"], dic["poses"], dic["pad"]
            print(mp4_path)