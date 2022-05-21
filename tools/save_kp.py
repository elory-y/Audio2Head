import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from modules.keypoint_detector import KPDetector
import os
import numpy as np
import glob
import torch
import cv2
from skimage import img_as_float32
from tqdm import tqdm


save_dir = "/home/caopu/workspace/Audio2Head/data"
check_path = "./checkpoints/audio2head.pth.tar"
kp_detector = KPDetector(block_expansion=32, num_kp=10, num_channels=3, max_features=1024, num_blocks=5, temperature=0.1,estimate_jacobian=True, scale_factor=0.25)
checkpoint = torch.load(check_path)
kp_detector = kp_detector.cuda()
kp_detector.load_state_dict(checkpoint["kp_detector"])
kp_detector.eval()
mp4_path = "/home/caopu/workspace/Audio2Head/data/*.mp4"
mp4_lis = glob.glob(mp4_path)
for mp4 in tqdm(mp4_lis):
    cap = cv2.VideoCapture(mp4)
    lis = []
    name = os.path.basename(mp4).split(".")[0]
    while cap.isOpened():
        success, get_img = cap.read()
        if success:
            get_img = cv2.resize(get_img, (256, 256))
            get_img = np.array(img_as_float32(get_img))
            get_img = get_img.transpose((2, 0, 1))
            get_img = torch.from_numpy(get_img).unsqueeze(0).cuda()
            with torch.no_grad():
                kp = kp_detector(get_img)
                del kp["jacobian_map"]
                del kp["pred_fature"]
                lis.append(kp)
        else:
            break
    np.save(os.path.join(save_dir, "kp"+name+".npy"), lis)
    cap.release()
