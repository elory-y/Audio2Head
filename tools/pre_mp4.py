#!/user/bin/env python
# coding=utf-8
import sys
sys.path.append('..')
sys.path.append('../modules')
import glob
import cv2
import os
import csv
from tqdm import tqdm
import numpy as np
# from get_pose_from_audio import get_audio_feature_from_audio
# from moviepy.editor import *


# def mp4_wav(mp4):
#     change_name =mp4.replace("mp4", "wav")
#     video = VideoFileClip(mp4)
#     audio = video.audio
#     audio.write_audiofile(change_name)

path = "/media/user/93651a1e-7b15-4a2a-8126-53546427a/youtube_data/split_data/*.mp4"
img_lis = glob.glob(path)
root_dir = "/media/user/93651a1e-7b15-4a2a-8126-53546427a/youtube_data/split_npy"
for mp4 in tqdm(img_lis):
    cmd = ' "/media/user/93651a1e-7b15-4a2a-8126-53546427a/zqy_data/OpenFace/build/bin/FeatureExtraction" verbose -f "%s" -cam_width 256 -cam_height 256 -out_dir "%s" -pose'%(mp4, root_dir)
    os.system(cmd)
    name = os.path.basename(mp4).split(".")[0]
    csv_path = os.path.join(root_dir, name+".csv")
    poses = []
    with open(csv_path) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            pose = []
            # [res, tra]
            pose = [float(row[8]), float(row[9]), float(row[10]), float(row[5]), float(row[6]), float(row[7])]
            poses.append(pose)
    np.save(os.path.join(root_dir, "pose_"+name + ".npy"), np.array(poses))
    os.remove(csv_path)
    os.remove(os.path.join(root_dir,name+"_of_details.txt"))
    #存wav音频
    # mp4_wav(mp4)
    # audio_path = os.path.join(root_dir, name+".wav")
    # audio_features = get_audio_feature_from_audio(audio_path)
    # np.save(os.path.join(root_dir, "audio_features" + "_" + name + ".npy"), audio_features)