from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tools.insightface_head_pose import HeadPose
from modules.audio2pose import get_pose_from_audio, get_audio_feature_from_audio
import os
import glob
from skimage import io, img_as_float32
import cv2
import numpy as np
import torch
from scipy.io import wavfile


class MyData(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.img_lis = glob.glob(os.path.join(self.root_dir, "*.jpg"))
        self.headpose = HeadPose()
    def __getitem__(self, item):
        dic = {}
        rots = []
        tras = []
        img_path = self.img_lis[item]
        mp4_path = img_path.split('.')[0] + ".mp4"
        audio_path = img_path.split('.')[0] + '.wav'
        img = io.imread(img_path)[:, :, :3]
        img = cv2.resize(img, (256, 256))
        img = np.array(img_as_float32(img))
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
        dic["img"] = img
        audio_feature = get_audio_feature_from_audio(audio_path)
        dic["audio_feature"] = audio_feature
        capture = cv2.VideoCapture(mp4_path)
        if capture.isOpened():
            while True:
                success, frame = capture.read()
                if success:
                    rot, tra = self.headpose.caculate_pose_vector(frame)
                    rots.append(rot)
                    tras.append(tra)
                else:
                    break
        dic["rots"] = rots
        dic["tras"] = tras
        return dic

    def __len__(self):
        return len(os.listdir(self.root_dir))


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = MyData(root_dir="/home/ssd/suimang/project/Audio2Head/data")
    data = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=4)
    dataiter = iter(data)
    dic = dataiter.next()
    print(dic["audio_feature"].size())
    print(len(dic["rots"]))
    print(len(dic["tras"]))
    img=dic["img"]
