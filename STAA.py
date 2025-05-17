import sys
sys.path.append('/home/gc/projects/weixing')
import copy

import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
import math
from model.transformer.tau import TAUSubBlock


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class layerNormFeedForward(nn.Module):
    def __init__(self, input_channel,d_model):
        super().__init__()
        self.ff1 = TAUSubBlock(dim=d_model)


    def forward(self, x):
        b, c , t , h ,w = x.shape

        x = x + self.ff1(x.reshape(b*c,t, h, w)).view(b, c, t, h, w)

        return x


def attention_s(q, k, v):
    # q,k,v : [b*heads, s, c, h, w]
    # [b*heads,s,c*h*w] * [b*heads,c*h*w,s] => [b*heads,s,s]
    scores_s = torch.matmul(q.view(q.size(0), q.size(1), -1), k.view(k.size(0), k.size(1), -1).permute(0, 2, 1)) \
             / math.sqrt(q.size(2) * q.size(3) * q.size(4))

    # [b*heads,s,s] 按照最后一维 经过softmax 得到相应的权重
    scores_s = F.softmax(scores_s, dim=-1)

    # [b*heads,s,s] * [b*heads,s,c*h*w] =>  [b*heads,s,c*h*w]
    v_s = torch.matmul(scores_s, v.reshape(v.size(0), v.size(1), -1))

    # [b*heads,s,c*h*w] => [b*heads, s, c, h, w]
    output = v_s.reshape(q.size())
    return output


class MultiHeadAttention_S(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        self.d_model = d_model
        self.h = heads

        self.q_Conv = nn.Sequential(nn.Conv2d(self.d_model, self.d_model, 1, 1, 0, bias=False),
                                    nn.GroupNorm(1, self.d_model),
                                    )
        self.v_Conv = nn.Sequential(nn.Conv2d(self.d_model, self.d_model, 1, 1, 0, bias=False),
                                    nn.GroupNorm(1, self.d_model),
                                    )
        self.k_Conv = nn.Sequential(nn.Conv2d(self.d_model, self.d_model, 1, 1, 0, bias=False),
                                    nn.GroupNorm(1, self.d_model),
                                    )
        self.v_post_f = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, 1, 0, bias=False),
            nn.GroupNorm(1, d_model),
            nn.SiLU(inplace=False),
        )

    def forward(self, q, k, v):
        b_q, s_q, c_q, h_q, w_q = q.size()
        b_k, s_k, c_k, h_k, w_k = k.size()
        b_v, s_v, c_v, h_v, w_v = v.size()
        # wq * x；输入[b*s,c,h,w] 输出 q : [b*s,heads,c//heads,h,w]；heads 为多头注意力的头的数量
        q = self.q_Conv(q.reshape(q.size(0) * q.size(1), *q.shape[2:])).reshape(q.size(0)*q.size(1), self.h,
                                                                                self.d_model // self.h, h_q, w_q)

        # 将q的形状改为 [b,heads,s,c,h,w]
        q = q.reshape(b_q, s_q, self.h, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        # [b*heads,s,c,h,w]
        q = q.reshape(q.size(0)*q.size(1), *q.shape[2:])

        # wk * x 得到 k
        k = self.k_Conv(k.reshape(k.size(0) * k.size(1), *k.shape[2:])).reshape(k.size(0) * k.size(1), self.h,
                                                                                self.d_model // self.h, h_q, w_q)
        k = k.reshape(b_k, s_k, self.h, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        # [b*heads,s,c,h,w]
        k = k.reshape(k.size(0) * k.size(1), *k.shape[2:])

        # wv * x 得到 v
        v = self.v_Conv(v.reshape(v.size(0) * v.size(1), *v.shape[2:])).reshape(v.size(0) * v.size(1), self.h,
                                                                                self.d_model // self.h, h_q, w_q)
        v = v.reshape(b_v, s_v, self.h, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        # [b*heads,s,c,h,w]
        v = v.reshape(v.size(0) * v.size(1), *v.shape[2:])
        # [b, s, heads, c, h, w]
        output = attention_s(q, k, v).reshape(b_q, self.h, s_q, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        # [b, s, c, h, w]
        output = self.v_post_f(output.reshape(b_q*s_q, self.h, self.d_model // self.h,
                                              h_q, w_q).reshape(b_q*s_q, self.d_model, h_q, w_q)).view(b_q, s_q, c_q, h_q, w_q)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, input_channel,d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.norm_1 = nn.GroupNorm(1, d_model)
        self.norm_2 = nn.GroupNorm(1, d_model)
        self.attn_1 = MultiHeadAttention_S(heads, d_model) # 多头注意力机制
        self.ff = layerNormFeedForward(input_channel,d_model) # 前馈

    def forward(self, x):
        b, s, c, h, w = x.size()

        x = x + self.attn_1(x, x, x) # q k v
        x = self.norm_1(x.view(-1, c, h, w)).view(b, s, c, h, w) # 单独的对每个通道进行独立的归一化

        x = x + self.ff(x)
        x = self.norm_2(x.view(-1, c, h, w)).view(b, s, c, h, w) # 单独的对每个通道进行独立的归一化
        return x

# sin cos位置编码
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=20):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        pos_ec = self.pe[:10,:]
        pos_ec = torch.squeeze(pos_ec).reshape(-1,101,201)
        pos_ec = pos_ec.unsqueeze(0).unsqueeze(0)
        pos_ec = pos_ec.expand(x.size(0),x.size(1),10,101,201)
        return x + pos_ec


class Encoder(nn.Module):
    def __init__(self, input_channel,d_model, N, heads,is_pos=False):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.is_pos = is_pos
        self.input_channel = input_channel
        self.layers = get_clones(EncoderLayer(self.input_channel,d_model, heads), N)
        self.conv = nn.Sequential(
            # TAUSubBlock(dim=12),
            nn.Conv2d(12, d_model, 3, 2, 1, bias=False), # 使图像高宽减半
            nn.GroupNorm(d_model, d_model), # 对每个通道进行独立的归一化
            nn.SiLU(inplace=False), # 激活函数 
            # TAUSubBlock(dim=d_model),
        )



    def forward(self, x):
        

        b, t, c, h, w = x.shape
        # 交换 t，c
        x = x.permute(0, 2, 1, 3, 4) 



        x = self.conv(x.reshape(b*c, t, h, w)).view(b, c, self.d_model, 51, 101)

        for i in range(self.N):
            x = self.layers[i](x)
        return x


class Transformer(nn.Module):
    def __init__(self, input_channel, d_model, output_channel, n_layers, heads,is_pos=False):
        super().__init__()
        self.encoder = Encoder(input_channel,d_model, n_layers, heads,is_pos=is_pos)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model, 3, 2, 1, output_padding=1, bias=False),
            nn.GroupNorm(1, d_model),
            nn.SiLU(inplace=False),

            nn.Conv2d(d_model, 12, 1, 1, 0, bias=False),# 不会改变图片的大小
        )
        self.pre = nn.Sequential(
            nn.Conv2d(input_channel, d_model, kernel_size=1, stride=1),
            nn.GroupNorm(1, d_model),
            nn.SiLU(inplace=False),

            nn.Conv2d(d_model, 1, kernel_size=1, stride=1),
        )
        
    
        self.d_model = d_model

    def forward(self, src):
        e_outputs = self.encoder(src)
        b , c, t, h ,w = e_outputs.shape

        y = self.conv(e_outputs.view(b*c, t, h, w))
        y = y[:,:,1:,1:]
        y = y.view(b, c, 12, 101 ,201)
        y = y.permute(0, 2, 1, 3, 4)  
        y = self.pre(y.reshape(b*12, c, 101 ,201)).view(b, 12, 1, 101 ,201)
        return y


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.randn((3, 12, 11, 101, 201)).to(device) # b t c h w
    model = Transformer(input_channel=11, d_model=128, output_channel=1, n_layers=6, heads=8).to(device)
    summary(model, (data.shape), device=device)
    # y = model(data,[])
    # print(y.shape)
    # # 计算模型参数总数
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")
    