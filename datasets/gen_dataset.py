from torch.utils.data import Dataset, DataLoader
import glob
import cv2
import numpy as np
import torch
import random
import os
from skimage import img_as_float32


class Gen_Data(Dataset):
    def __init__(self, root_dir, frames, star_frame):
        self.root_dir = root_dir
        self.mp4_lis = glob.glob(os.path.join(self.root_dir, "*.mp4"))
        self.frame = frames
        self.star_frame = star_frame
    def __getitem__(self, item):
        dic = {}
        values = []
        jacobian = []
        mp4_path = self.mp4_lis[item]
        cap = cv2.VideoCapture(mp4_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.star_frame-1)
        success, get_img = cap.read()
        get_img = cv2.resize(get_img, (256, 256))
        get_img = np.array(img_as_float32(get_img))
        get_img = get_img.transpose((2, 0, 1))
        get_img = torch.from_numpy(get_img).unsqueeze(0)
        name = os.path.basename(mp4_path).split(".")[0]
        kp_path = os.path.join(self.root_dir, "kp"+name+".npy")
        kp_features = np.load(kp_path, allow_pickle=True)
        kp_features= kp_features[self.star_frame:self.star_frame+self.frame]
        for kp_feature in kp_features:
            values.append(kp_feature["value"])
            jacobian.append(kp_feature["jacobian"])
        values = torch.stack(values, dim=1).squeeze(0)
        jacobian = torch.stack(jacobian, dim=1).squeeze(0)
        dic["img"] = get_img
        dic["value"] = values
        dic["jacobian"] = jacobian
        return dic
    def __len__(self):
        return len(os.listdir(self.root_dir)) // 5


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = Gen_Data(root_dir="/home/caopu/workspace/Audio2Head/data", frames=8, star_frame=3)
    data = DataLoader(dataset, batch_size=2,
                      shuffle=True, num_workers=0)
    for number, dic in enumerate(data):
        print(number)