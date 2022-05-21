import os
import numpy as np
import cv2
import torch
import random
from skimage import io, img_as_float32
from torch.utils.data import Dataset, DataLoader
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
def info_img(imgs, audio_features, poses, kp_detector, audio2kp, generator,pre_images, estimate_jacobian=True):
    predictions_gen = []
    key_points = []
    for i in range(len(imgs)):
        img = imgs[i].unsqueeze(0).cuda()
        # img = torch.from_numpy(img).unsqueeze(0).cuda()
        audio_feature = audio_features[i].unsqueeze(0).type(torch.FloatTensor).cuda()
        pose = poses[i].unsqueeze(0).type(torch.FloatTensor).cuda()
        t = {}
        t["audio"] = audio_feature
        t["pose"] = pose
        t["id_img"] = img
        kp_gen_source = kp_detector(img)
        gen_kp = audio2kp(t)
        key_points.append(gen_kp["value"][:, :pre_images].squeeze(0))
        for bs_idx in range(pre_images):
            tt = {}
            tt["value"] = gen_kp["value"][:, bs_idx]
            tt["jacobian"] = gen_kp["jacobian"][:, bs_idx]
            out_gen = generator(img, kp_source=kp_gen_source, kp_driving=tt)
            predictions_gen.append(out_gen['prediction'][0])
    predictions_gen = torch.stack(predictions_gen, dim=0)
    key_points = torch.cat(key_points, dim=0)
    return predictions_gen, key_points


if __name__== "__main__":
    sou_dir = "/home/user/Database/fomm/train"
    dri = "/home/user/Database/audio2head"
    import yaml
    from datasets.stage2_dataset import Stage2_Dataset
    from modules.generator import OcclusionAwareGenerator
    from modules.keypoint_detector import KPDetector
    from modules.audio2kp import AudioModel3D
    dataset = Stage2_Dataset(source_root=sou_dir, driving_root=dri, frames=64)
    data = DataLoader(dataset, batch_size=2,
                      shuffle=True, num_workers=0)
    model_path = r"./checkpoints/audio2head.pth.tar"
    config_file = r"./config/vox-256.yaml"
    checkpoint = torch.load(model_path)
    with open(config_file) as f:
        config = yaml.safe_load(f)
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = kp_detector.cuda()
    generator = generator.cuda()
    audio2kp = AudioModel3D(seq_len=64, block_expansion=32, num_blocks=5, max_features=512, num_kp=10)
    audio2kp = audio2kp.cuda()
    kp_detector.load_state_dict(checkpoint["kp_detector"])
    generator.load_state_dict(checkpoint["generator"])
    audio2kp.load_state_dict(checkpoint["audio2kp"])
    for number, dic in enumerate(data):
        imgs = dic["source_img"]
        audio_features = dic["audio_features"]
        poses = dic["poses"]
        source_img = dic["source_img"]
        star_img = dic["star_img"]
        a, b = info_img(star_img, audio_features, poses,kp_detector, audio2kp, generator)
        print(a.shape)