import torch
import torch.nn as nn
from ..blocks.pconvblocks import PCBActiv, ResnetBlock, UnetSkipConnectionDBlock


class Encoder(nn.Module):  # 典型的卷积编码器
    def __init__(self, input_nc, ngf=64, res_num=4, norm_layer="instance"):
        # input_nc：输入图像的通道数（例如 RGB 图像为 3）。
        # ngf：初始通道数（默认值为 64），随着网络的深度，通道数逐层加倍。
        # res_num：中间残差块（ResnetBlock）的数量，通常用于瓶颈部分特征提炼。
        # norm_layer：指定归一化类型（例如 instance 或 batch），用于控制每一层的归一化操作。
        super(Encoder, self).__init__()

        # construct unet structure  Encoder_1 是最浅层（初始层），Encoder_6 是最深层（瓶颈层） 每次将特征图的分辨率减半，通道数加倍
        Encoder_1 = PCBActiv(
            input_nc, ngf, norm_layer=None, activ=None, outer=True
        )  # 128
        Encoder_2 = PCBActiv(ngf, ngf * 2, norm_layer=norm_layer)  # 64
        Encoder_3 = PCBActiv(ngf * 2, ngf * 4, norm_layer=norm_layer)  # 32
        Encoder_4 = PCBActiv(ngf * 4, ngf * 8, norm_layer=norm_layer)  # 16
        Encoder_5 = PCBActiv(ngf * 8, ngf * 8, norm_layer=norm_layer)  # 8  特征图的分辨率减半,通道数不变
        Encoder_6 = PCBActiv(ngf * 8, ngf * 8, norm_layer=None, inner=True)  # 4  特征图的分辨率减半 ， 通道数不变

        blocks = []
        for _ in range(res_num):
            block = ResnetBlock(ngf * 8)  # 残差块  在最深层（瓶颈层）进一步提炼特征
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.Encoder_1 = Encoder_1
        self.Encoder_2 = Encoder_2
        self.Encoder_3 = Encoder_3
        self.Encoder_4 = Encoder_4
        self.Encoder_5 = Encoder_5
        self.Encoder_6 = Encoder_6

    def forward(self, x):
        out_1 = self.Encoder_1(x)
        out_2 = self.Encoder_2(out_1)
        out_3 = self.Encoder_3(out_2)
        out_4 = self.Encoder_4(out_3)
        out_5 = self.Encoder_5(out_4)
        out_6 = self.Encoder_6(out_5)
        out_7 = self.middle(out_6)
        return out_7, out_5, out_4, out_3, out_2, out_1  # out_7：最底层经过残差块处理后的深层语义特征。  out_5, out_4, ..., out_1：逐层编码的特征图（用于跳跃连接）



class Decoder(nn.Module):  # U-Net 解码器，作用是将编码器提取到的多尺度特征图逐层上采样并恢复到原始分辨率，同时结合跳跃连接（skip connection）保留浅层细节特征。
    def __init__(self, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(Decoder, self).__init__()

        # construct unet structure
        Decoder_1 = UnetSkipConnectionDBlock(
            ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True
        )  # 上采样 8
        Decoder_2 = UnetSkipConnectionDBlock(ngf * 16, ngf * 8, norm_layer=norm_layer)  # 输入通道数为上一层解码器输出的特征图通道数与对应编码器的特征图通道数之和
        Decoder_3 = UnetSkipConnectionDBlock(ngf * 16, ngf * 4, norm_layer=norm_layer)  # 32
        Decoder_4 = UnetSkipConnectionDBlock(ngf * 8, ngf * 2, norm_layer=norm_layer)  # 64
        Decoder_5 = UnetSkipConnectionDBlock(ngf * 4, ngf, norm_layer=norm_layer)  # 128
        Decoder_6 = UnetSkipConnectionDBlock(
            ngf * 2, output_nc, norm_layer=norm_layer, outermost=True
        )  # 256

        self.Decoder_1 = Decoder_1
        self.Decoder_2 = Decoder_2
        self.Decoder_3 = Decoder_3
        self.Decoder_4 = Decoder_4
        self.Decoder_5 = Decoder_5
        self.Decoder_6 = Decoder_6

    def forward(self, input_1, input_2, input_3, input_4, input_5, input_6):
        y_1 = self.Decoder_1(input_6[0])
        y_2 = self.Decoder_2(torch.cat([y_1, input_5[0]], 1))
        y_3 = self.Decoder_3(torch.cat([y_2, input_4[0]], 1))
        y_4 = self.Decoder_4(torch.cat([y_3, input_3[0]], 1))
        y_5 = self.Decoder_5(torch.cat([y_4, input_2[0]], 1))
        y_6 = self.Decoder_6(torch.cat([y_5, input_1[0]], 1))
        out = y_6  # （b, 3, 256, 256）
        return out


# class PCConv(nn.Module):
#     def __init__(self, input_nc,  output_nc, ngf, norm_layer):
#         super().__init__()
#         self.encoder = Encoder(input_nc, ngf, norm_layer=norm_layer)
#         self.decoder = Decoder(output_nc, ngf, norm_layer=norm_layer)
#
#     def forward(self, x):
#         out_6, out_5, out_4, out_3, out_2, out_1 = self.encoder(x)
#         out = self.decoder(out_1, out_2, out_3, out_4, out_5, out_6)
#         return out
