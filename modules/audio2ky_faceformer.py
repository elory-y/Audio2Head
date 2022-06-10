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

# Alignment Bias todo
def enc_dec_mask(device, T, S):
    mask = torch.ones(T, S)
    for i in range(T):
        j = max(i * 4 // 3 - 2, 0)
        mask[i, j:j+4] = 0
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
class EncoderMap(nn.Module):
    def __init__(self, args):
        super(EncoderMap, self).__init__()
        self.up_kp = nn.Linear(20, 64)
        self.up_jac = nn.Linear(40, 64)
        self.down = nn.Linear(128, 64)
        self.feature_dim = args.feature_dim
    def forward(self, key_point, jacobian):
        bs, frame, _, _ =key_point.shape
        key_point = key_point.reshape(bs, frame, -1)
        jacobian = jacobian.reshape(bs, frame, -1)
        up_key_point = F.gelu(self.up_kp(key_point))
        up_jacobian = F.gelu(self.up_jac(jacobian))
        cat_fea = torch.cat((up_key_point, up_jacobian), dim=-1)
        if self.feature_dim == 128:
            return cat_fea
        else:
            return self.down(cat_fea)
class DecoderMap(nn.Module):
    def __init__(self, args):
        super(DecoderMap, self).__init__()
        self.dw = nn.Linear(args.feature_dim, 60)

    def forward(self, x):
        x = self.dw(x)
        bs, frame, _ = x.shape
        key_point = x[:, :, :20].reshape(bs, frame, 10, 2)
        jacobian = x[:, :, 20:60].reshape(bs, frame, 10, 2, 2)
        return key_point, jacobian
class Faceformer(nn.Module):
    def __init__(self, args, device):
        super(Faceformer, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        # self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # wav2vec 2.0 weights initialization
        # self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(2048, args.feature_dim)
        # motion encoder
        # self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        # periodic positional encoding
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period=args.period)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head=4, max_seq_len=600, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.num_layers)
        # motion decoder
        # self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        self.device = device
        self.encodermap = EncoderMap(args)
        self.decodermap = DecoderMap(args)
        # nn.init.constant_(self.vertice_map_r.weight, 0)
        # nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, audio_tensor,  kp, jac, teacher_forcing=True):
        hidden_states = self.audio_feature_map(audio_tensor)
        # hidden_states = F.interpolate(hidden_states.permute(0,2,1),128).permute(0,2,1)
        frame =kp.shape[1]
        batch_size = kp.shape[0]
        if teacher_forcing:
            driving_input = self.encodermap(kp, jac)
            driving_input = self.PPE(driving_input)
            tgt_mask = self.biased_mask[:, :driving_input.shape[1], :driving_input.shape[1]].clone().detach().to(device=self.device)
            tgt_mask = tgt_mask.repeat(batch_size, 1, 1)
            memory_mask = enc_dec_mask(self.device, driving_input.shape[1], hidden_states.shape[1])
            driving_output = self.transformer_decoder(driving_input, hidden_states, tgt_mask=tgt_mask,
                                                   memory_mask=memory_mask)
            key_point, jacobian = self.decodermap(driving_output)

        else:
            kp, jac = kp[:, :1], jac[:, :1]
            for i in range(frame):
                if i == 0:
                    driving_emd = self.encodermap(kp, jac)
                    driving_input = self.PPE(driving_emd)
                else:
                    driving_input = self.PPE(driving_emd)

                tgt_mask = self.biased_mask[:, :driving_input.shape[1],
                           :driving_input.shape[1]].clone().detach().to(device=self.device)
                tgt_mask = tgt_mask.repeat(batch_size, 1, 1)
                memory_mask = enc_dec_mask(self.device, driving_input.shape[1],
                                           hidden_states.shape[1])
                driving_output = self.transformer_decoder(driving_input, hidden_states, tgt_mask=tgt_mask,
                                                          memory_mask=memory_mask)
                #todo 解码出每个位置的关键点和加科比，然后和之前的的拼一起然后在棉麻
                new_kp, new_jac = self.decodermap(driving_output)
                new_emb = self.encodermap(new_kp[:, -1].unsqueeze(1), new_jac[:, -1].unsqueeze(1))
                driving_emd = torch.cat((driving_emd, new_emb), 1)
            key_point, jacobian = new_kp, new_jac
        return key_point, jacobian








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=60,
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
    model = model.to(torch.device("cuda"))
    model.train()
    template_kp = torch.rand(4,1,10,2).cuda()

    template_jiakebi = torch.rand(4,1,10,2,2).cuda()


    vertice_kp = torch.rand(4,64,10,2).cuda()
    vertice_jakebi = torch.rand(4,64,10,2,2).cuda()

    audio_feature = torch.rand(4,86,2048).cuda()
    one_hot = torch.tensor([[0., 0., 0., 0., 0., 1., 0., 0.]]).cuda()
    # model(audio_feature, source, driving, one_hot, teacher_forcing=False)
    model(audio_feature, template_kp, template_jiakebi, vertice_kp, vertice_jakebi, one_hot, teacher_forcing=False)



