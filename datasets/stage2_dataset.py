from torch.utils.data import Dataset, DataLoader
import glob
from skimage import io, img_as_float32
import cv2
from augmentation import AllAugmentationTransform
import numpy as np
import os
import torch
import glob
import random



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

class Stage2_Dataset(Dataset):
    def __init__(self, frames, source_root, driving_root, pre_images, istrain):
        self.frames = frames
        self.source_root = source_root
        self.driving_root = driving_root
        self.pre_images = pre_images
        self.source_lis = glob.glob(os.path.join(self.source_root, "*"))
        self.istrain = istrain
    def __len__(self):
        return len(self.source_lis)

    def __getitem__(self, item):
        # mp4_path = self.mp4_lis[item]
        source_dir = self.source_lis[item]
        img_id = os.path.basename(source_dir)
        source_imgs_lis = np.sort(glob.glob(os.path.join(source_dir, "*")))
        audio_feature_path = os.path.join(self.driving_root, "train", 'audio_features_' + img_id + ".npy")
        if not os.path.exists(audio_feature_path):
            audio_feature_path = os.path.join(self.driving_root, "test", 'audio_features_' + img_id + ".npy")
            pose_feature_path = os.path.join(self.driving_root, "test", "pose_" + img_id + ".npy")
        else:
            pose_feature_path = os.path.join(self.driving_root, "train", "pose_" + img_id + ".npy")
        audio_feature = np.load(audio_feature_path)
        poses = np.load(pose_feature_path)
        re = poses[:, :3]
        tra = poses[:, 3:]
        dic={}
        audio_frames = len(audio_feature) // 4
        pose_frames = poses.shape[0]
        frames = min(audio_frames, pose_frames)
        if self.istrain:
            star_frame = random.randint(0, max(0, frames - self.frames-2))
        else:
            star_frame = 0
        end_frame = star_frame + self.frames
        audio_feature = audio_feature[star_frame * 4: end_frame * 4, :]
        assert audio_feature.shape == (256, 41)
        re = re[star_frame : end_frame , :]
        tra = tra[star_frame: end_frame, :]
        total_poses = []
        audio_features = []
        for i in range(self.frames):
            audio_features.append(audio_feature[i * 4:i * 4 + 4, :])
            trans = tra[i, :]
            rot = re[i, :]
            pose = np.zeros([256, 256])
            draw_annotation_box(pose, np.array(rot), np.array(trans))
            total_poses.append(pose)
        source_img = []
        source_img_lis = source_imgs_lis[star_frame:star_frame + self.pre_images]
        star_img = cv2.imread(source_imgs_lis[star_frame])
        star_img = np.array(img_as_float32(star_img))
        star_img = star_img.transpose((2, 0, 1))
        for img_path in source_img_lis:
            img = cv2.imread(img_path)
            img = np.array(img_as_float32(img))
            img = img.transpose((2, 0, 1))
            source_img.append(img)
        dic["star_frame"] = star_frame
        dic["audio_features"] = torch.from_numpy(np.array(audio_features))
        dic["poses"] = torch.from_numpy(np.array(total_poses))
        dic["source_img"] = torch.from_numpy(np.array(source_img))
        dic["star_img"] = star_img
        return dic



class Stage2_PaddleAudioData(Dataset):
    def __init__(self, frames, source_root, driving_root, pre_images, pad_feature_root, istrain):
        self.frames = frames
        self.source_root = source_root
        self.driving_root = driving_root
        self.pre_images = pre_images
        self.source_lis = glob.glob(os.path.join(self.source_root, "*"))
        self.pad_feature_root = pad_feature_root
        self.istrain = istrain
    def __len__(self):
        return len(self.source_lis)

    def __getitem__(self, item):
        # mp4_path = self.mp4_lis[item]
        source_dir = self.source_lis[item]
        img_id = os.path.basename(source_dir)
        source_imgs_lis = np.sort(glob.glob(os.path.join(source_dir, "*")))
        audio_feature_path = os.path.join(self.pad_feature_root,  'audio_feature_pad_' + img_id + "_new.npy")
        if not os.path.exists(audio_feature_path):
            pose_feature_path = os.path.join(self.driving_root, "test", "pose_" + img_id + ".npy")
        else:
            pose_feature_path = os.path.join(self.driving_root, "train", "pose_" + img_id + ".npy")
        audio_feature = np.load(audio_feature_path)
        poses = np.load(pose_feature_path)
        re = poses[:, :3]
        tra = poses[:, 3:]
        dic = {}
        pose_frames = poses.shape[0]
        frames = pose_frames
        if self.istrain:
            star_frame = random.randint(0, max(0, frames - self.frames-2))
        else:
            star_frame = 0
        end_frame = star_frame + self.frames
        audio_feature = audio_feature[:, int(star_frame * 4 // 3): int(star_frame * 4 // 3) + 86, :]
        re = re[star_frame: end_frame, :]
        tra = tra[star_frame: end_frame, :]
        total_poses = []
        for i in range(self.frames):
            trans = tra[i, :]
            rot = re[i, :]
            pose = np.zeros([256, 256])
            draw_annotation_box(pose, np.array(rot), np.array(trans))
            total_poses.append(pose)
        source_img = []
        source_img_lis = source_imgs_lis[star_frame:star_frame + self.pre_images]
        star_img = cv2.imread(source_imgs_lis[star_frame])
        star_img = np.array(img_as_float32(star_img))
        star_img = star_img.transpose((2, 0, 1))
        for img_path in source_img_lis:
            img = cv2.imread(img_path)
            img = np.array(img_as_float32(img))
            img = img.transpose((2, 0, 1))
            source_img.append(img)
        dic["star_frame"] = star_frame
        dic["audio_features"] = torch.from_numpy(np.array(audio_feature))
        dic["poses"] = torch.from_numpy(np.array(total_poses))
        dic["source_img"] = torch.from_numpy(np.array(source_img))
        dic["star_img"] = star_img
        return dic

if __name__ == "__main__":
    from modules.generator import OcclusionAwareGenerator
    from modules.keypoint_detector import KPDetector
    from modules.audio2kp import AudioModel3D
    import yaml
    import glob
    import random

    # model_path = r"./checkpoints/audio2head.pth.tar"
    # config_file = r"./config/vox-256.yaml"
    # checkpoint = torch.load(model_path)
    # with open(config_file) as f:
    #     config = yaml.safe_load(f)
    # kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
    #                          **config['model_params']['common_params'])
    # generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
    #                                     **config['model_params']['common_params'])
    # kp_detector = kp_detector.cuda()
    # generator = generator.cuda()
    # audio2kp = AudioModel3D(seq_len=64, block_expansion=32, num_blocks=5, max_features=512, num_kp=10)
    # audio2kp = audio2kp.cuda()
    # kp_detector.load_state_dict(checkpoint["kp_detector"])
    # generator.load_state_dict(checkpoint["generator"])
    # audio2kp.load_state_dict(checkpoint["audio2kp"])
    sou_dir = "/home/user/Database/fomm/train"
    dri = "/home/caopu/workspace/Audio2Head/data"
    dataset = Stage2_Dataset(source_root=sou_dir, driving_root=dri, frames=64,pre_images=3)
    data = DataLoader(dataset, batch_size=2,
                      shuffle=True, num_workers=0)
    for number, dic in enumerate(data):
        imgs = dic["source_img"]
        # audio_feature_paths = dic["audio_feature_path"]
        # pose_feature_paths = dic["pose_feature_path"]
        # predictions_gens = info_img(imgs=imgs,audio_feature_paths=audio_feature_paths,pose_feature_paths=pose_feature_paths,kp_detector=kp_detector,audio2kp=audio2kp,generator=generator)
        print(len(imgs))