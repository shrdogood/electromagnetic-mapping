import torch.nn as nn
import torch


class PartialConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()

        self.input_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.mask_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)  # 权重被初始化为常数 1.0
        self.mask_conv.no_init = True  # 添加一个标志变量

        # mask is not updated
        for param in self.mask_conv.parameters():  # 这些权重不参与梯度更新（requires_grad=False），即它是一个固定的卷积操作
            param.requires_grad = False

    def forward(self, x):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        input = x[0]  # 输入数据
        mask = x[1].float().cuda()   # 掩码 （值为1的区域是有效区域）

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes.bool(), 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        # 将归一化后的结果中无效区域填充为 0，并生成新的遮罩
        output = output_pre.masked_fill_(no_update_holes.bool(), 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes.bool(), 0.0)
        out = [output, new_mask]  # 返回新的卷积结果和更新的遮罩
        return out


class PCBActiv(nn.Module):  # 结合了部分卷积，归一化层，激活函数
    def __init__(
        self,
        in_ch,
        out_ch,
        norm_layer="instance",
        sample="down-4",  # 通过参数 sample 决定使用何种类型的卷积
        activ="leaky",
        conv_bias=False,
        inner=False,
        outer=False,  # inner 和 outer：这些参数控制激活函数和归一化的应用顺序。
    ):
        super().__init__()
        if sample == "same-5":
            self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == "same-7":
            self.conv = PartialConv(in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == "down-4":  # 特征图尺寸变为原来的一半
            self.conv = PartialConv(in_ch, out_ch, 4, 2, 1, bias=conv_bias)
        else:  # 默认：卷积核大小为 3，步长为 1
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if norm_layer == "instance":
            self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        elif norm_layer == "batch":
            self.norm = nn.BatchNorm2d(out_ch, affine=True)
        else:
            pass

        if activ == "relu":
            self.activation = nn.ReLU()
        elif activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            pass
        self.inner = inner
        self.outer = outer

    def forward(self, input):  # 根据 inner 和 outer 的值，调整处理顺序
        out = input
        if self.inner:  # 先应用激活函数，然后卷积
            out[0] = self.activation(out[0])
            out = self.conv(out)  # 部分卷积
        elif self.outer:  # 只应用卷积，不做其他处理。
            out = self.conv(out)
        else:  # 先应用激活函数，卷积后再归一化。
            out[0] = self.activation(out[0])
            out = self.conv(out)
            out[0] = self.norm(out[0])
        return out


# Define the resnet block
class ResnetBlock(nn.Module):  # 基于 部分卷积 的 ResNet 残差块
    def __init__(self, dim, norm="instance"):  # 输入dim和输出dim相同，通过每个残差块提取更加细化的特征，网络逐层增强输入的信息表达能力
        super(ResnetBlock, self).__init__()
        self.conv_1 = PartialConv(dim, dim, 3, 1, 1, 1)  # 保持输出尺寸与输入一致
        if norm == "instance":
            self.norm_1 = nn.InstanceNorm2d(dim, track_running_stats=False)
            self.norm_2 = nn.InstanceNorm2d(dim, track_running_stats=False)
        elif norm == "batch":
            self.norm_1 = nn.BatchNorm2d(dim, track_running_stats=False)
            self.norm_2 = nn.BatchNorm2d(dim, track_running_stats=False)
        self.active = nn.ReLU(True)  # inplace=True 表示激活函数会在输入张量上直接进行操作，节省内存。
        self.conv_2 = PartialConv(dim, dim, 3, 1, 1, 1)

    def forward(self, x):
        out = self.conv_1(x)
        out[0] = self.norm_1(out[0])
        out[0] = self.active(out[0])
        out = self.conv_2(out)
        out[0] = self.norm_2(out[0])
        out[0] = x[0] + out[0]  # 残差连接（skip connection）
        return out  # [output_image, updated_mask]


class UnetSkipConnectionDBlock(nn.Module):  # 用于 U-Net 解码器部分的跳跃连接块
    def __init__(
        self,
        inner_nc,  # 输入特征图的通道数（较大）
        outer_nc,  # 输出特征图的通道数
        outermost=False,  # 布尔值，标记是否是 U-Net 的最外层（即最后一层输出层）
        innermost=False,  # 布尔值，标记是否是解码器的最内层（即最底部的特征恢复层）
        norm_layer="instance",
    ):
        super(UnetSkipConnectionDBlock, self).__init__()
        uprelu = nn.ReLU()
        upconv = nn.ConvTranspose2d(
            inner_nc, outer_nc, kernel_size=4, stride=2, padding=1
        )  # 通过反卷积操作对特征图进行上采样（分辨率变为原来的 2 倍）
        if norm_layer == "instance":
            upnorm = nn.InstanceNorm2d(outer_nc, affine=True)
        elif norm_layer == "batch":
            upnorm = nn.BatchNorm2d(outer_nc, affine=True)
        else:
            pass

        if outermost:
            up = [uprelu, upconv, nn.Tanh()]  # 输出层将特征图转换为最终的图像，通常通过 Tanh 激活函数限制值域
            model = up
        elif innermost:
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            up = [uprelu, upconv, upnorm]
            model = up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)  # 输出的特征图形状为 [B, outer_nc, H*2, W*2]，空间分辨率扩大 2 倍，通道数由 inner_nc 转为 outer_nc
