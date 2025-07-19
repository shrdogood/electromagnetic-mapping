import torch
from collections import OrderedDict
from .base_model import BaseModel
from .network import networks
from models.blocks.loss import PerceptualLoss, StyleLoss, TVloss
from loguru import logger


class PCConv(BaseModel):
    def __init__(self, cfg):
        super(PCConv, self).__init__(cfg)
        self.isTrain = cfg.isTrain
        self.cfg = cfg
        # define network
        self.netEN, self.netDE = networks.define_G(
            cfg, self.select_model, cfg.norm, cfg.init_type, self.gpu_ids, cfg.init_gain
        )
        self.model_names = ["EN", "DE"]
        logger.info("network {} has been defined".format(self.select_model))
        if self.isTrain:  # 如果是训练模式（isTrain=True），则会初始化优化器、损失函数和学习率调度器。
            self.old_lr = cfg.lr
            # define loss functions
            self.PerceptualLoss = PerceptualLoss()  # 感知损失
            self.StyleLoss = StyleLoss()  # 风格损失
            self.criterionL1 = torch.nn.L1Loss()  # L1 损失函数，计算预测值和目标值之间的绝对误差
            self.criterionL2 = torch.nn.MSELoss()  # 均方误差损失函数 (L2)，计算平方误差

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_EN = torch.optim.Adam(
                self.netEN.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999)  # 优化器只有一个参数组，这个参数组里面是：self.netEN模型的所有参数
            )
            self.optimizer_DE = torch.optim.Adam(
                self.netDE.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999)  # betas=(cfg.beta1, 0.999) 是 Adam 优化器的 beta 参数，影响一阶和二阶矩的估计平滑
            )
            self.optimizers.append(self.optimizer_EN)
            self.optimizers.append(self.optimizer_DE)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, cfg))  # 创建调度器

        logger.info("---------- Networks initialized -------------")
        networks.print_network(self.netEN)
        networks.print_network(self.netDE)
        logger.info("---------- Networks initialized Done-------------")
        if not self.isTrain or cfg.continue_train:
            logger.info(
                "Loading pre-trained network {}! You choose the results of {} epoch".format(
                    self.select_model, cfg.which_epoch
                )
            )
            self.load_networks(cfg.which_epoch)  # 加载模型

    def name(self):
        return self.select_model  # 'PConv' or 'PDGAN'

    def set_input(self, mask, gt):  # 将输入的掩码 (mask) 和目标图像 (gt) 转换成适合训练的形式
        """
        Args:
            mask: input mask, the pixel value of masked regions is 1
            gt: ground truth image

        """
        self.gt = gt.to(self.device)
        self.mask = mask.to(self.device)
        self.input_img = self.gt.clone()  # 将 gt 克隆一份作为输入图像 input_img，这样可以避免直接修改原始数据。
        self.mask = self.mask.repeat(1, 3, 1, 1)  # 将单通道的 mask 转换为多通道（3 通道，R、G、B），与图像维度一致。(b, 3, 256, 256)
        #  unpositve with original mask
        self.inv_mask = 1 - self.mask
        # Do not set the mask regions as 0, this process can stable training
        # 对遮挡区域的像素值进行填充，避免将遮挡区域直接设置为零值（设置为零可能导致训练不稳定）
        self.input_img.narrow(1, 0, 1).masked_fill_(
            self.mask.narrow(1, 0, 1).bool(), 2 * 123.0 / 255.0 - 1.0  # 平均值需要替换！！！！   2 * 123.0 / 255.0 - 1.0是把平均像素归一化到（-1，1），因为预处理过的图像的像素值也是在（-1，1）
        )  # 从 self.input_img 中选取红色通道（通道 0）
        # masked_fill_(mask, value) 中：
        # mask 是布尔张量（True 表示需要填充的区域）。
        # value 是要填充的具体值。
        self.input_img.narrow(1, 1, 1).masked_fill_(
            self.mask.narrow(1, 0, 1).bool(), 2 * 104.0 / 255.0 - 1.0
        )
        self.input_img.narrow(1, 2, 1).masked_fill_(
            self.mask.narrow(1, 0, 1).bool(), 2 * 117.0 / 255.0 - 1.0
        )

    def forward(self):
        fake_p_6, fake_p_5, fake_p_4, fake_p_3, fake_p_2, fake_p_1 = self.netEN(
            [self.input_img, self.inv_mask]
        )
        self.G_out = self.netDE(
            fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6
        )
        self.Completion = self.G_out * self.mask + self.inv_mask * self.gt  # 将生成图像部分（掩码为1）和目标图像部分（掩码为0）拼接在一起

    def backward_G(self):
        self.hole_loss = self.criterionL1(self.G_out * self.mask, self.gt * self.mask)  # 孔洞区域损失
        self.valid_loss = self.criterionL1(self.G_out * self.inv_mask, self.gt * self.inv_mask)  # 有效区域损失
        self.Perceptual_loss = self.PerceptualLoss(self.G_out, self.gt) + self.PerceptualLoss(self.Completion, self.gt)  # 感知损失
        self.Style_Loss = self.StyleLoss(self.G_out, self.gt)
        self.TV_Loss = TVloss(self.Completion, self.inv_mask, "mean")  # 全变分损失  鼓励生成图像的局部平滑性，减少不必要的噪声或伪影
        # The weights of losses are same as
        # https://openaccess.thecvf.com/content_ECCV_2018/papers/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.pdf
        self.loss_G = (
            self.hole_loss * 6
            + self.Perceptual_loss * 0.05
            + self.Style_Loss * 120
            + self.TV_Loss * 0.1
            + self.valid_loss
        )
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # 前向传播
        self.optimizer_EN.zero_grad()  # 清空梯度
        self.optimizer_DE.zero_grad()
        self.backward_G()  # 计算梯度
        self.optimizer_EN.step()  # 更新参数
        self.optimizer_DE.step()

    def get_current_errors(self):
        # show the current loss
        return OrderedDict(  # 有序字典
            [
                ("L1_valid", self.valid_loss),
                ("L1_hole", self.hole_loss),
                ("Style", self.Style_Loss),
                ("Perceptual", self.Perceptual_loss),
                ("TV_Loss", self.TV_Loss),
            ]
        )

    # You can also see the Tensorborad
    def get_current_visuals(self):  # 返回一个字典
        return {
            "input_image": self.input_img,
            "output": self.Completion,
            "mask": self.mask,
            "ground_truth": self.gt,
        }
