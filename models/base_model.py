import os
import torch


class BaseModel:
    def __init__(self, cfg):
        self.model_names = None  # ["EN", "DE"] 或 ["PDGAN"]
        self.cfg = cfg
        self.gpu_ids = cfg.gpu_ids
        self.isTrain = cfg.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.device = (
            torch.device("cuda:{}".format(self.gpu_ids[0]))
            if self.gpu_ids
            else torch.device("cpu")
        )
        self.save_dir = os.path.join(cfg.basic_dir, cfg.checkpoints_dir, cfg.name)
        self.select_model = cfg.model  # 'PConv' or 'PDGAN'
        self.input_img = None
        self.gt = None
        self.mask = None
        self.inv_mask = None

    def name(self):
        return "BaseModel"

    def set_input(self, **kwargs):
        pass

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    # 用于保存神经网络模型和优化器的状态字典（state_dict）
    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):  # 确保列表中的元素是字符串类型
                save_filename = "%s_net_%s.pth" % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename).replace(
                    "\\", "/"  # 将路径中的反斜杠 \ 替换为正斜杠 /，确保路径格式在所有系统上都能被正确解析
                )
                net = getattr(self, "net" + name)  # 返回当前类对象中名为 netG 的属性值（通常是一个模型对象）
                optimize = getattr(self, "optimizer_" + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():  # 如果有可用GPU
                    torch.save(
                        {
                            "net": net.cpu().state_dict(),  # 当模型使用 torch.nn.DataParallel 进行多 GPU 训练时，模型会被封装在一个 DataParallel 实例中
                            # 直接访问net，会获取 DataParallel 封装器。  net.module才是访问内部的实际模型
                            "optimize": optimize.state_dict(),  # 优化器状态的设备信息会被自动保留（如 cuda:0），不需要显式移动到 CPU，加载优化器状态到的时候，状态会被加载到保存时的设备（如 cuda:0）
                        },
                        save_path,
                    )
                    net.cuda(self.gpu_ids[0])  # 将模型重新移回 GPU 上
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = "%s_net_%s.pth" % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)

                net = getattr(self, "net" + name)  # 返回当前对象的"net" + name属性（模型对象）
                optimize = getattr(self, "optimizer_" + name)  # 返回当前对象的"optimizer_" + name属性（优化器对象）
                if isinstance(net, torch.nn.DataParallel):  # 如果模型使用了 torch.nn.DataParallel 进行多 GPU 训练，net 是一个 DataParallel 封装器
                    net = net.module  # 访问实际的网络模型，避免加载参数时产生不匹配的问题
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(
                    load_path.replace("\\", "/"), map_location=str(self.device)
                )  # 将加载的参数映射到指定设备（如 cuda:0 或 cpu）  模型和优化器的参数都必须在同一个设备上
                optimize.load_state_dict(state_dict["optimize"])  # 加载保存的优化器参数（如学习率、动量等）
                net.load_state_dict(state_dict["net"])  # 加载保存的网络参数（如权重和偏置）。

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):  # 如果传入的 nets 不是列表（即只有一个网络），将其转换为包含单个网络的列表 [nets]。
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self, epoch):
        if "PDGAN" in self.model_names:
            # We use TTUR  Two Time-scale Update Rule 时间尺度更新规则
            if epoch > self.cfg.niter:
                lrd = self.cfg.lr / self.cfg.niter_decay  # 学习率衰减速率，等于初始学习率 (self.cfg.lr) 除以总衰减阶段的轮次数 (self.cfg.niter_decay)
                new_lr = self.old_lr - lrd
            else:
                new_lr = self.old_lr

            if new_lr != self.old_lr:

                new_lr_G = new_lr / 2  # 生成器的学习率是基础学习率的一半（更低的学习率）
                new_lr_D = new_lr * 2  # 判别器的学习率是基础学习率的两倍（更高的学习率）。

                for param_group in self.optimizer_D.param_groups:  # 遍历优化器的参数组（param_groups），将对应的学习率设置为计算得到的新值。
                    param_group["lr"] = new_lr_D
                for param_group in self.optimizer_PDGAN.param_groups:
                    param_group["lr"] = new_lr_G
                print("update learning rate: %f -> %f" % (self.old_lr, new_lr))
                self.old_lr = new_lr
        elif "EN" in self.model_names or "DE" in self.model_names:
            for scheduler in self.schedulers:  # 调用学习率调度器（scheduler）来更新学习率，分别更新编码器和译码器的学习率
                scheduler.step()
            lr = self.optimizers[0].param_groups[0]["lr"]  # 获取并打印第一个优化器中第一个参数组的学习率
            print("learning rate = %.7f" % lr)
        else:
            raise ValueError(f"wrong model name, please select one of (PDGAN|PConv)")
