import glob
import os

import cv2
import numpy as np
from skimage import io, img_as_float32
from torch.utils.data import Dataset

from modules.audio2pose import get_audio_feature_from_audio
from tools.insightface_head_pose import HeadPose


class MyData(Dataset):
    def __init__(self, root_dir):
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
        dic["img"] = img
        audio_feature = get_audio_feature_from_audio(audio_path)
        dic["audio_feature"] = audio_feature
        print(audio_feature.shape, audio_path)
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
    from modules.audio2pose import get_pose_from_audio

    dataset = MyData(root_dir="/home/ssd/suimang/project/Audio2Head/data")
    data = DataLoader(dataset, batch_size=1,
                      shuffle=False, num_workers=4)
    for i, dic in enumerate(data):
        img, audio_feature, rots, tras = dic["img"], dic["audio_feature"], dic["rots"], dic["tras"]
        rot, trans = get_pose_from_audio(img, audio_feature, model_path="./checkpoints/audio2head.pth.tar")
