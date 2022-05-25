import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules.audio2kp import AudioModel3D
from modules.generator import OcclusionAwareGenerator
import yaml
import argparse
import cv2
import numpy as np
import random
from skimage import io, img_as_float32
from modules.keypoint_detector import KPDetector
from tools.stage2_loss import GeneratorFullModel
from modules.discriminator import MultiScaleDiscriminator
import glob
from tools.info_image import info_img
from datasets.stage2_dataset import Stage2_Dataset
from torch.nn.parallel import DistributedDataParallel
import wandb
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

wandb.init(entity="suimang", project="ky_predictor_fomm_2stage", name="newgen_newaudio_6")


def preapare_kypoint(imgs, KPDetector):
    # 计算原图的关键点
    source_kypoint = []
    for img in imgs:
        img = img.unsqueeze(0)
        kp_gen_source = KPDetector(img)
        source_kypoint.append(kp_gen_source["value"])
    source_kypoint = torch.stack(source_kypoint, dim=0)
    return source_kypoint
def calculate_loss(losses_generator, driving_kypoint, source_kypoint, iteration, istrain=True):
    loss_function = nn.L1Loss(reduction='none')
    perceptual_loss = losses_generator["perceptual"]
    equivariance_value = losses_generator['equivariance_value']
    equivariance_jacobian_loss = losses_generator['equivariance_jacobian']
    ky_loss = loss_function(source_kypoint, driving_kypoint)
    number = ky_loss.shape[0] * ky_loss.shape[1]
    ky_loss = ky_loss.flatten(2).mean(-1).sum() / number
    loss = 100 * ky_loss +  perceptual_loss + 10 * equivariance_value + 10*equivariance_jacobian_loss
    phase = 'train' if istrain else 'test'
    log_dict = {
        f'{phase}_kp_loss': 100 * ky_loss.item(),
        f'{phase}_perceptual_loss':  perceptual_loss.item(),
        f'{phase}_equivariance_value': 10 * equivariance_value.item(),
        f'{phase}_equivariance_jacobian_loss': 10 * equivariance_jacobian_loss.item(),
        f'{phase}_loss': loss.item()
    }
    wandb.log(log_dict, step=iteration)
    return loss

def process_data(source_img, kp_detector):
    driving_kp = kp_detector(source_img)
    # for img in source_img:
    #     img = img.unsqueeze(0)
    #     ky_point = kp_detector(img)
    #     driving_kp.append(ky_point["value"])
    # driving_kp = torch.stack(driving_kp, dim=1).squeeze(0)
    return driving_kp

def run(args, generator, kp_detector, audio2kp, device):
    optimizer = torch.optim.Adam(audio2kp.parameters(), lr=args.lr)
    generator_full = GeneratorFullModel(kp_detector,generator,
                                        transform_params=args.generator_transform_params,
                                        loss_weights=args.generator_loss_weights)

    train_dataset = Stage2_Dataset(source_root=args.train_source, driving_root=args.train_driving, frames=64, pre_images=args.pre_images)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dataset))
    train_data = DataLoader(train_dataset, batch_size=args.batch_size,
                       shuffle=True, num_workers=0)
    test_dataset = Stage2_Dataset(source_root=args.test_source, driving_root=args.test_driving, frames=64, pre_images=args.pre_images)
    test_data = DataLoader(test_dataset, batch_size=args.batch_size,
                       shuffle=True, num_workers=0)

    train_iteration = 0
    test_iteration = 0

    for epoch in range(args.epochs):
        audio2kp.train()
        for iteration, dic in enumerate(train_data):
            audio_features = dic["audio_features"]
            poses = dic["poses"]
            source_img = dic["source_img"].view(-1,3,256,256)
            star_img = dic["star_img"]
            train_iteration += 1
            predictions_gens, driving_key_points = info_img(star_img, audio_features, poses,kp_detector, audio2kp, generator, pre_images=args.pre_images)
            traget_imgs = {}
            traget_imgs["driving"] = predictions_gens
            traget_imgs["source"] = source_img.to(device)
            source_key_points = process_data(traget_imgs["source"], kp_detector)
            losses_generator = generator_full(traget_imgs, source_key_points)
            loss = calculate_loss(losses_generator, driving_kypoint=driving_key_points, source_kypoint=source_key_points["value"], iteration=train_iteration, istrain=True)
            print(loss.item(),"_______",train_iteration)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        audio2kp.eval()
        test_loss = 0
        num = 0
        with torch.no_grad():
            for iteration, dic in enumerate(test_data):
                audio_features = dic["audio_features"]
                poses = dic["poses"]
                source_img = dic["source_img"].view(-1, 3, 256, 256)
                star_img = dic["star_img"]
                train_iteration += 1
                predictions_gens, driving_key_points = info_img(star_img, audio_features, poses, kp_detector, audio2kp,
                                                                generator, pre_images=args.pre_images)
                traget_imgs = {}
                traget_imgs["driving"] = predictions_gens
                traget_imgs["source"] = source_img.to(device)
                source_key_points = process_data(traget_imgs["source"], kp_detector)
                losses_generator = generator_full(traget_imgs, source_key_points, is_train=False)
                loss = calculate_loss(losses_generator, driving_kypoint=driving_key_points,
                                      source_kypoint=source_key_points["value"], iteration=train_iteration,
                                      istrain=True)
                test_iteration += 1
                num += 1
                test_loss += loss.item()
                print("test_loss", loss.item())
        torch.save(audio2kp.state_dict(), os.path.join("/home/user/Database/audio2head/fomm_checkpoint3",
                                                               '3_%s_%.5f.pth' % (epoch, test_loss / num)))



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kp_detector = KPDetector(block_expansion=args.kp_dete_block_expansion, num_kp=args.num_kp,
                             num_channels=args.num_channels,
                             max_features=args.kp_dete_max_features, num_blocks=args.kp_dete_num_blocks,
                             temperature=args.kp_dete_temperature,
                             estimate_jacobian=args.estimate_jacobian, scale_factor=args.kp_dete_scale_factor)
    kp_detector = kp_detector.to(device)
    checkpoint = torch.load(args.model_path)
    kp_detector.load_state_dict(checkpoint["kp_detector"])
    kp_detector.eval()
    audio2kp = AudioModel3D(seq_len=args.seq_len, block_expansion=args.AudioModel_block_expansion,
                            num_blocks=args.AudioModel_num_blocks, max_features=args.AudioModel_max_features,
                            num_kp=args.num_kp).to(device)
    check_path = "/home/user/Database/audio2head/fomm_checkpoint3/1_5_526.53601.pth"
    audio2kp.load_state_dict(torch.load(check_path))
    generator = OcclusionAwareGenerator(num_channels=args.num_channels, num_kp=args.num_kp,
                                        block_expansion=args.generator_block_expansion,
                                        max_features=args.generator_max_features,
                                        num_down_blocks=args.generator_num_down_blocks,
                                        num_bottleneck_blocks=args.generator_num_bottleneck_blocks,
                                        estimate_occlusion_map=args.generator_estimate_occlusion_map,
                                        dense_motion_params=args.generator_dense_motion_params).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    # discriminator.load_state_dict(checkpoint['discriminator'])
    # discriminator.eval()
    # if torch.cuda.is_available():
    #     discriminator.to(device)
    run(args=args, kp_detector=kp_detector, audio2kp=audio2kp, generator=generator, device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default=64)
    parser.add_argument("--lr", default=2.0e-4)
    parser.add_argument("--batch_size", default=2)
    parser.add_argument("--model_path", default=r"./checkpoint/audio2head.pth.tar",
                        help="pretrained model path")
    parser.add_argument("--train_source", default=r"/home/user/Database/fomm/train")
    parser.add_argument("--test_source", default=r"/home/user/Database/fomm/test")
    parser.add_argument("--train_driving", default=r"/home/caopu/workspace/Audio2Head/data")
    parser.add_argument("--test_driving", default=r"/home/caopu/workspace/Audio2Head/data")
    parser.add_argument("--epochs", default=2000)
    parser.add_argument("--config", default="./config/parameters.yaml")
    parser.add_argument("--seq_len", default=64)
    parser.add_argument("--num_kp", default=10)
    parser.add_argument("--pre_images", default=5)
    parser.add_argument("--AudioModel_num_blocks", default=5, help="AudioModel3D model num_blocks")
    parser.add_argument("--AudioModel_max_features", default=512, help="AudioModel3D model max_features")
    parser.add_argument("--estimate_jacobian", default=True)
    parser.add_argument("--AudioModel_block_expansion", default=32, help="AudioModel3D model block_expansion")
    parser.add_argument("--kp_dete_block_expansion", default=32)
    parser.add_argument("--num_channels", default=3)
    parser.add_argument("--kp_dete_max_features", default=1024)
    parser.add_argument("--kp_dete_scale_factor", default=0.25)
    parser.add_argument("--kp_dete_num_blocks", default=5)
    parser.add_argument("--kp_dete_temperature", default=0.1)
    parser.add_argument("--generator_block_expansion", default=64)
    parser.add_argument("--generator_max_features", default=512)
    parser.add_argument("--generator_num_down_blocks", default=2)
    parser.add_argument("--generator_num_bottleneck_blocks", default=6)
    parser.add_argument("--generator_estimate_occlusion_map", default=True)
    parser.add_argument("--generator_dense_motion_params",
                        default={'block_expansion': 64, 'max_features': 1024, 'num_blocks': 5, 'scale_factor': 0.25})
    parser.add_argument("--generator_transform_params",
                        default={"sigma_affine": 0.05, "sigma_tps": 0.005, "points_tps": 5})
    parser.add_argument("--generator_loss_weights",
                        default={"generator_gan": 0, "discriminator_gan": 1, "feature_matching": [10, 10, 10, 10],
                                 "perceptual": [10, 10, 10, 10, 10], "equivariance_value": 10,
                                 "equivariance_jacobian": 10})
    args = parser.parse_args()
    main(args)