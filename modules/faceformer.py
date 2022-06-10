import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
# from wav2vec import Wav2Vec2Model
import argparse


# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Faceformer(nn.Module):
    def __init__(self, args):
        super(Faceformer, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.dataset = args.dataset
        # self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # wav2vec 2.0 weights initialization
        # self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(768, args.feature_dim)
        # motion encoder
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        # periodic positional encoding
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 600, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        # motion decoder
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        # style embedding
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)
        self.device = args.device
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, audio_tensor, template, vertice, one_hot,teacher_forcing=True):
        #输入音频特征，检测关键的，每个图的关键点
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        template = template.unsqueeze(1) # (1,1, V*3)，目标点
        obj_embedding = self.obj_vector(one_hot)#(1, feature_dim),不同目标做编码
        frame_num = vertice.shape[1]
        hidden_states = audio_tensor

        if teacher_forcing:
            vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
            style_emb = vertice_emb #对每个不同id做风格画，onehot代表不同人说话风格
            vertice_input = torch.cat((template,vertice[:,:-1]), 1) # shift one position
            vertice_input = vertice_input - template
            vertice_input = self.vertice_map(vertice_input)#[1,2,64] 修改vertice_dim维度改变点个数，10个点设置20
            # vertice_input = vertice_input + style_emb
            vertice_input = self.PPE(vertice_input)#[1,2,64]
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device) #[4,2,2]
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])#[2,345]
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask) #[1,2,64]
            vertice_out = self.vertice_map_r(vertice_out) #[1,2,15096]
        else:
            for i in range(frame_num):
                if i==0:
                    vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                    style_emb = vertice_emb
                    vertice_input = self.PPE(style_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
                memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                vertice_out = self.vertice_map_r(vertice_out)
                new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + template

        return vertice_out
    #原版的forward
    # def forward(self, audio, template, vertice, one_hot, criterion, teacher_forcing=True):
    #     #audio音频特征，template是3d人脸mesh的特征，vertice一段视频每帧的人脸mesh特征，one——hot是代表每个不同说话的人，cirterion的lossfunction
    #     # tgt_mask: :math:`(T, T)`.
    #     # memory_mask: :math:`(T, S)`.
    #     template = template.unsqueeze(1)  # (1,1, V*3)
    #     obj_embedding = self.obj_vector(one_hot)  # (1, feature_dim)
    #     frame_num = vertice.shape[1]
    #     hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state #[1, frame_num, 64]
    #     if self.dataset == "BIWI":
    #         if hidden_states.shape[1] < frame_num * 2:
    #             vertice = vertice[:, :hidden_states.shape[1] // 2]
    #             frame_num = hidden_states.shape[1] // 2
    #     hidden_states = self.audio_feature_map(hidden_states) #[1, frame_num, 64]
    #
    #     if teacher_forcing:
    #         vertice_emb = obj_embedding.unsqueeze(1)  # (1,1,feature_dim)
    #         style_emb = vertice_emb
    #         vertice_input = torch.cat((template, vertice[:, :-1]), 1)  # shift one position [1,frame_num, 15096]
    #         vertice_input = vertice_input - template
    #         vertice_input = self.vertice_map(vertice_input) #[1,frame_num,64]
    #         vertice_input = vertice_input + style_emb
    #         vertice_input = self.PPE(vertice_input) #[1,frame_num,64]
    #         tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(
    #             device=self.device) #[4,frame_num,frame_num]
    #         memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])#[frame_num,frame_num]
    #         vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask,
    #                                                memory_mask=memory_mask) #[1,frame_num,64]
    #         vertice_out = self.vertice_map_r(vertice_out) #[1,frame_num, 15096]
    #     else:
    #         for i in range(frame_num):
    #             if i == 0:
    #                 vertice_emb = obj_embedding.unsqueeze(1)  # (1,1,feature_dim)
    #                 style_emb = vertice_emb
    #                 vertice_input = self.PPE(style_emb)
    #             else:
    #                 vertice_input = self.PPE(vertice_emb)
    #             tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(
    #                 device=self.device)
    #             memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
    #             vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask,
    #                                                    memory_mask=memory_mask)
    #             vertice_out = self.vertice_map_r(vertice_out)
    #             new_output = self.vertice_map(vertice_out[:, -1, :]).unsqueeze(1)
    #             new_output = new_output + style_emb
    #             vertice_emb = torch.cat((vertice_emb, new_output), 1)
    #
    #     vertice_out = vertice_out + template
    #     loss = criterion(vertice_out, vertice)  # (batch, seq_len, V*3)
    #     loss = torch.mean(loss)
    #     return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3,
                        help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default="wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
                                                              " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
                                                              " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
                                                              " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
                                                            " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
                                                             " FaceTalk_170731_00024_TA")
    args = parser.parse_args()

    # build model
    model = Faceformer(args)
    model.train()
    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    # audio_feature = torch.rand(1,184274).cuda()
    one_hot = torch.rand(1,8).cuda()
    audio_feature = torch.rand(1, 345, 64).cuda()#中间345对应视频帧数,2048维度通过self.audio_feature_map修改
    template = torch.rand(1, 10).cuda()
    vertice = torch.rand(1,3,10).cuda()
    model(audio_feature,template,vertice,one_hot,teacher_forcing=True)
    '''
    原版随机的tensor
    audio_feature = torch.rand(1,184274).cuda()
    template = torch.rand(1, 15069).cuda()
    vertice = torch.rand(1,3,15069).cuda()
    model(audio_feature,template,vertice)
    '''