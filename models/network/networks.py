# Define networks, init networks
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .pconv import Encoder, Decoder
from models.network.Discriminator import MultiscaleDiscriminator
from .pdgan import SPDNormGenerator


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, cfg):
    if cfg.lr_policy == "lambda":
        def lambda_rule(epoch):  # 学习率在 cfg.niter 轮次之后开始线性衰减，衰减持续 cfg.niter_decay 个轮次
            lr_l = 1.0 - max(0, epoch - cfg.niter) / float(cfg.niter_decay)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif cfg.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_iters, gamma=0.1)  # 每隔固定的轮次（cfg.lr_decay_iters）就将学习率乘以一个衰减因子（gamma=0.1）
    elif cfg.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)  # 当模型的性能（比如验证集的损失）在若干轮次内没有改善时，降低学习率。
    elif cfg.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.niter, eta_min=0
        )  # 使用余弦退火方法调整学习率，学习率呈现余弦曲线下降。
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", cfg.lr_policy
        )
    return scheduler


def init_weights(net, init_type="normal", gain=0.02):  # 初始化神经网络权重的函数
    def init_func(m):  # 这是一个递归函数，它会被应用到 net 的每一层，检查其类型并根据 init_type 对权重进行初始化。
        classname = m.__class__.__name__  # 获取层的类名  如 Conv2d、Linear
        if hasattr(m, "no_init") and m.no_init:
            return

        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):  # 如果当前层有 weight 属性，并且类名包含 Conv 或 Linear，则可以进行权重初始化
            if init_type == "normal":  # 正态分布初始化，均值为 0.0，标准差为 gain
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == "xavier":  # Xavier 初始化，适用于激活函数为 Sigmoid 或 Tanh 的网络
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == "kaiming":  # Kaiming 初始化，适用于 ReLU 激活函数的网络
                init.kaiming_normal(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":  # orthogonal：正交初始化，保持权重矩阵的正交性。
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:  # 如果层有 bias（偏置）属性且不为 None，则将偏置初始化为常数 0.0。
                init.constant(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:  # 如果是 BatchNorm2d 层：
            # 初始化 weight（权重）为正态分布，均值为 1.0，标准差为 gain。
            # 初始化 bias 为常数 0.0。
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

        # 针对 InstanceNorm2d 层的初始化
        elif classname.find("InstanceNorm2d") != -1:
            init.normal_(m.weight.data, mean=1.0, std=gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # net.apply(init_func) 会遍历 net 中的每一个子模块（包括嵌套的模块），并将 init_func 应用到每一个子模块上（都会将这个模块作为参数传递给 init_func）


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])  # 将模型移动到指定的 GPU
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(
    cfg, model_name, norm="batch", init_type="normal", gpu_ids=[], init_gain=0.02  # 比例因子
):
    if model_name is "PDGAN":
        PConvEncoder = Encoder(cfg.input_nc, cfg.ngf, norm_layer=norm)
        PConvDecoder = Decoder(cfg.output_nc, cfg.ngf, norm_layer=norm)
        PDGANNet = SPDNormGenerator(cfg)
        return (
            init_net(PDGANNet, init_type, init_gain, gpu_ids),
            init_net(PConvEncoder, init_type, init_gain, gpu_ids),
            init_net(PConvDecoder, init_type, init_gain, gpu_ids),
        )
    elif model_name is "PConv":
        PConvEncoder = Encoder(cfg.input_nc, cfg.ngf, norm_layer=norm)
        PConvDecoder = Decoder(cfg.output_nc, cfg.ngf, norm_layer=norm)
        return (
            init_net(PConvEncoder, init_type, init_gain, gpu_ids),
            init_net(PConvDecoder, init_type, init_gain, gpu_ids)
        )
    else:
        raise ValueError("select wrong model name:{}".format(model_name))


def define_D(cfg, init_type="normal", gpu_ids=[], init_gain=0.02):
    netD = MultiscaleDiscriminator(cfg)
    return init_net(netD, init_type, init_gain, gpu_ids)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: %d" % num_params)
