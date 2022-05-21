from dataset.dataset import MyData
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from modules.util import MyResNet34
from modules.audio2pose import audio2poseLSTM
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch_num = 200
lr = 0.1
data_path = "/home/ssd/suimang/project/Audio2Head/data"
batch_size = 2
loss_function = nn.MSELoss()
generator = audio2poseLSTM().to(device)

optimizer = torch.optim.Adam(generator.parameters(), lr= lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.9)

dataset = MyData(data_path)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)
for epoch in range(epoch_num):
    generator.train()
    for i, dic in enumerate(dataloader):
        img, audio_feature, lab_rots, lab_tras = dic["img"], dic["audio_feature"], dic["rots"], dic["tras"]
        num_frame = audio_feature.shape[1] // 4
        minv = np.array([-0.639, -0.501, -0.47, -102.6, -32.5, 184.6], dtype=np.float32)
        maxv = np.array([0.411, 0.547, 0.433, 159.1, 116.5, 376.5], dtype=np.float32)
        audio_seqs = []
        a = audio_feature.numpy()
        for j in range(batch_size):
            audio_seq = []
            for i in range(num_frame):
                audio_seq.append(a[j][i * 4:i * 4 + 4])
            audio_seqs.append(audio_seq)
        # for j in range(batch_size):
        #     audio_seq = []
        #     for i in range(num_frame):
        #         audio_seq.append(audio_feature[j][i * 4:i * 4 + 4])
        #         if i * 4 + 4 > 840:
        #             print(i)
        #     audio_seqs.append(audio_seq)
        # audio_seqs = np.array(audio_seqs[0],dtype=np.float32)
        # audio = torch.from_numpy(audio_seqs).cuda()
        audio = torch.from_numpy(np.array(audio_seqs, dtype=np.float32)).cuda()
        x = {}
        img = img.cuda()
        x["img"] = img
        x["audio"] = audio
        poses = generator(x)
        poses = poses.cpu().data.numpy()[0]
        poses = (poses + 1) / 2 * (maxv - minv) + minv
        pre_rot, pre_tras = poses[:, :3].copy(), poses[:, 3:].copy()
        lab = torch.cat((lab_rots[0],lab_tras[0]), 0)
        pre = torch.cat((pre_rot, pre_tras), 0)
        loss = loss_function(pre, lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("loss", loss.data)
