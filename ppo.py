import fire
from pathlib import Path
from shutil import rmtree
from collections import deque, namedtuple
from random import randrange
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, tensor, cat, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical

from einops import reduce, repeat, einsum, rearrange
from ema_pytorch import EMA
from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2
from hl_gauss_pytorch import HLGaussLoss
from hyper_connections import HyperConnections
import gymnasium as gym


"""
设置计算设备。
如果CUDA（GPU）可用，则使用第一个CUDA设备；否则，使用CPU。
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 内存结构定义（Memory）
Memory = namedtuple('Memory', [
    'state',        # 状态，表示环境的状态
    'action',       # 动作，表示智能体在状态下的动作
    'action_log_prob',  # 动作的对数概率，用于计算策略梯度
    'reward',       # 奖励，表示在执行动作后环境给予的奖励
    'done',         # 结束标志，表示当前状态是否为终止状态
    'value'         # 价值，表示状态的价值估计
])


# 经验数据集类（ExperienceDataset）
class ExperienceDataset(Dataset):
    """
    经验数据集类，用于将智能体的经验数据转换为可迭代的数据集。

    该类继承自 `torch.utils.data.Dataset`，并重写了 `__len__` 和 `__getitem__` 方法，
    以便与 PyTorch 的数据加载器（DataLoader）兼容。

    Args:
        data (tuple of torch.Tensor): 包含多个张量的元组，每个张量对应体验数据的一个字段。
    """
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        """
        返回数据集的大小，即体验数据的数量。

        Returns:
            int: 体验数据的数量。
        """
        return len(self.data[0])

    def __getitem__(self, ind):
        """
        根据索引获取单个体验数据。

        Args:
            ind (int): 索引。

        Returns:
            tuple: 对应索引的体验数据，包含状态、动作、动作对数概率、奖励、结束标志和价值。
        """
        return tuple(map(lambda t: t[ind], self.data))
    

# 创建打乱的数据加载器函数（create_shuffled_dataloader）
def create_shuffled_dataloader(data, batch_size):
    """
    创建一个打乱的数据加载器。

    该函数将体验数据转换为 `ExperienceDataset`，然后使用 `DataLoader` 创建数据加载器，
    并设置 `shuffle=True` 以打乱数据。

    Args:
        data (tuple of torch.Tensor): 包含多个张量的元组，每个张量对应体验数据的一个字段。
        batch_size (int): 小批量大小。

    Returns:
        DataLoader: 打乱后的数据加载器。
    """
    ds = ExperienceDataset(data)
    return DataLoader(ds, batch_size = batch_size, shuffle = True)


def exists(val):
    """
    检查一个值是否存在（即不为None）。

    Args:
        val: 任意类型的值。

    Returns:
        bool: 如果值不为None，则返回True；否则返回False。
    """
    return val is not None


def default(v, d):
    """
    如果值存在（即不为None），则返回该值；否则返回默认值。

    Args:
        v: 任意类型的值。
        d: 默认值。

    Returns:
        任意类型: 如果v不为None，则返回v；否则返回d。
    """
    return v if exists(v) else d


def divisible_by(num, den):
    """
    检查一个数是否可以被另一个数整除。

    Args:
        num (int): 被除数。
        den (int): 除数。

    Returns:
        bool: 如果 `num` 可以被 `den` 整除，则返回True；否则返回False。
    """
    return (num % den) == 0


def normalize(t, eps = 1e-5):
    """
    对输入张量进行归一化处理。

    该函数对输入张量 `t` 进行标准化，使其均值为0，标准差为1。

    Args:
        t (torch.Tensor): 输入张量。
        eps (float, optional): 用于防止除以零的小常数，默认为1e-5。

    Returns:
        torch.Tensor: 归一化后的张量。
    """
    return (t - t.mean()) / (t.std() + eps)


def update_network_(loss, optimizer):
    """
    更新神经网络参数。

    该函数计算损失函数对网络参数的梯度，并使用优化器更新参数。

    Args:
        loss (torch.Tensor): 损失张量。
        optimizer (torch.optim.Optimizer): 优化器。
    """
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()


class RSMNorm(Module):
    """
    RSMNorm（Running Statistics Mean Normalization）类。

    该类实现了基于运行统计量的均值归一化。与传统的批归一化（BatchNorm）不同，
    RSMNorm 使用全局的运行均值和方差进行归一化，而不是基于当前批次的统计量。
    这使得它在处理小批量数据或在线学习时更加稳定。

    Args:
        dim (int): 输入特征的维度大小。
        eps (float, optional): 用于数值稳定性的一个小常数，默认为1e-5。
    """
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        super().__init__()
        # 输入特征的维度大小
        self.dim = dim
        # 用于数值稳定性的常数
        self.eps = 1e-5
        
        # 注册为缓冲区的张量，这些张量在模型参数更新时不参与梯度计算
        self.register_buffer('step', tensor(1)) # 记录当前步骤数，初始为1
        self.register_buffer('running_mean', torch.zeros(dim)) # 初始化运行均值，形状为 (dim,)
        self.register_buffer('running_variance', torch.ones(dim)) # 初始化运行方差，形状为 (dim,)

    def forward(
        self,
        x
    ):
        """
        前向传播方法，执行归一化操作。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, ..., dim)。

        Returns:
            torch.Tensor: 归一化后的张量，形状与输入相同。
        """
        # 检查输入特征的维度是否匹配
        assert x.shape[-1] == self.dim, f'expected feature dimension of {self.dim} but received {x.shape[-1]}'

        # 获取当前步骤数
        time = self.step.item()
        # 获取当前的运行均值
        mean = self.running_mean
        # 获取当前的运行方差
        variance = self.running_variance

        # 执行归一化操作： (x - mean) / sqrt(variance + eps)
        normed = (x - mean) / variance.sqrt().clamp(min = self.eps)

        if not self.training:
            # 如果不是训练模式，直接返回归一化后的张量
            return normed

        # 如果是训练模式，则更新运行均值和方差
        with torch.no_grad():
            # 计算当前批次的均值
            # 对第0维（即批次维度）求均值
            new_obs_mean = reduce(x, '... d -> d', 'mean')
            # 计算当前批次均值与运行均值的差
            delta = new_obs_mean - mean

            # 更新运行均值： new_mean = mean + delta / time
            new_mean = mean + delta / time
            # 更新运行方差： new_variance = (time - 1) / time * (variance + (delta ** 2) / time)
            new_variance = (time - 1) / time * (variance + (delta ** 2) / time)

            # 更新缓冲区中的运行均值和方差
            self.step.add_(1) # 增加步骤数
            self.running_mean.copy_(new_mean) # 更新运行均值
            self.running_variance.copy_(new_variance) # 更新运行方差

        # 返回归一化后的张量
        return normed


class ReluSquared(Module):
    """
    ReluSquared 激活函数类。

    该激活函数定义为：f(x) = sign(x) * (ReLU(x))^2。
    它结合了ReLU和平方操作，能够在保持非线性的同时，增加对大值的敏感度。

    Args:
        None

    Returns:
        torch.Tensor: 经过 ReluSquared 激活函数处理后的张量。
    """
    def forward(self, x):
        return x.sign() * F.relu(x) ** 2


class SimBa(Module):
    """
    SimBa 架构类。

    SimBa 是一种用于深度强化学习（Deep RL）的网络架构，旨在通过注入简单性偏差来扩展参数规模。
    它由三个主要组件组成：
    1. 观测归一化层：对输入进行标准化处理。
    2. 残差前馈块：提供从输入到输出的线性路径。
    3. 层归一化：控制特征幅度。

    Args:
        dim (int): 输入特征的维度大小。
        dim_hidden (int, optional): 隐藏层的维度大小。如果为None，则默认为 `dim * expansion_factor`。
        depth (int, optional): 网络的深度，即残差块的层数，默认为3。
        dropout (float, optional): Dropout概率，默认为0。
        expansion_factor (int, optional): 扩展因子，用于扩展隐藏层的维度，默认为2。
        num_residual_streams (int, optional): 残差流的数量，默认为4。
    """

    def __init__(
        self,
        dim,
        dim_hidden = None,
        depth = 3,
        dropout = 0.,
        expansion_factor = 2,
        num_residual_streams = 4
    ):
        super().__init__()

        # 残差流的数量
        self.num_residual_streams = num_residual_streams

        # 设置隐藏层的维度，默认为 `dim * expansion_factor`
        dim_hidden = default(dim_hidden, dim * expansion_factor)

        # 初始化层列表
        layers = []

        # 定义输入线性变换层，将维度从 `dim` 扩展到 `dim_hidden`
        self.proj_in = nn.Linear(dim, dim_hidden)

        # 计算内部维度
        dim_inner = dim_hidden * expansion_factor

        # 获取初始化和扩展/减少残差流函数的函数
        init_hyper_conn, self.expand_stream, self.reduce_stream = HyperConnections.get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        for ind in range(depth):
            # 定义残差块
            layer = nn.Sequential(
                nn.RMSNorm(dim_hidden),
                nn.Linear(dim_hidden, dim_inner),
                ReluSquared(),
                nn.Linear(dim_inner, dim_hidden),
                nn.Dropout(dropout),
            )

            # 应用初始化函数对残差块进行初始化
            layer = init_hyper_conn(dim = dim_hidden, layer_index = ind, branch = layer)
            # 将残差块添加到层列表中
            layers.append(layer)

        # 将层列表转换为 ModuleList
        self.layers = ModuleList(layers)

        # 定义最后的 RMS 归一化层
        self.final_norm = nn.RMSNorm(dim_hidden)

    def forward(self, x):
        """
        前向传播方法，执行 SimBa 架构的计算。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, dim)。

        Returns:
            torch.Tensor: 经过 SimBa 架构处理后的输出张量，形状为 (batch_size, dim_hidden)。
        """
        # 检查输入是否没有批次维度
        no_batch = x.ndim == 1

        if no_batch:
            # 如果没有批次维度，则添加一个批次维度
            x = rearrange(x, '... -> 1 ...')

        # 应用输入线性变换
        x = self.proj_in(x)

        # 应用扩展残差流
        x = self.expand_stream(x)

        for layer in self.layers:
            # 应用每个残差块
            x = layer(x)

        # 应用减少残差流
        x = self.reduce_stream(x)

        # 应用最后的 RMS 归一化
        out = self.final_norm(x)

        if no_batch:
            # 如果之前添加了批次维度，则移除它
            out = rearrange(out, '1 ... -> ...')

        return out


class Actor(Module):
    """
    Actor（行动者）类。

    该类实现了行动者网络，用于在强化学习中生成动作。
    它结合了 RSMNorm、SimBa 网络、动作头和价值头。

    Args:
        state_dim (int): 状态特征的维度大小。
        hidden_dim (int): 隐藏层的维度大小。
        num_actions (int): 动作空间的维度大小。
        mlp_depth (int, optional): 多层感知机（MLP）的深度，默认为2。
        dropout (float, optional): Dropout概率，默认为0.1。
        rsmnorm_input (bool, optional): 是否使用 RSMNorm 对输入进行归一化，默认为True。
    """
    def __init__(
        self,
        state_dim,
        hidden_dim,
        num_actions,
        mlp_depth = 2,
        dropout = 0.1,
        rsmnorm_input = True  # use the RSMNorm for inputs proposed by KAIST + SonyAI
    ):
        super().__init__()
        # 如果需要，则应用 RSMNorm；否则，使用恒等映射
        self.rsmnorm = RSMNorm(state_dim) if rsmnorm_input else nn.Identity()

        self.net = SimBa(
            state_dim, # 输入特征的维度大小
            dim_hidden = hidden_dim * 2, # 隐藏层的维度大小，设置为 `hidden_dim * 2`
            depth = mlp_depth, # MLP 的深度
            dropout = dropout # Dropout 概率
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # 线性变换，将维度从 `hidden_dim * 2` 减少到 `hidden_dim`
            ReluSquared(), # 应用 ReluSquared 激活函数
            nn.Linear(hidden_dim, num_actions) # 线性变换，将维度从 `hidden_dim` 转换为 `num_actions`
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # 线性变换，将维度从 `hidden_dim * 2` 减少到 `hidden_dim`
            ReluSquared(), # 应用 ReluSquared 激活函数
            nn.Linear(hidden_dim, 1) # 线性变换，将维度从 `hidden_dim` 转换为1，用于价值估计
        )

    def forward(self, x):
        """
        前向传播方法，执行行动者网络的计算。

        Args:
            x (torch.Tensor): 输入状态张量，形状为 (batch_size, state_dim)。

        Returns:
            tuple:
                - action_probs (torch.Tensor): 动作概率，形状为 (batch_size, num_actions)。
                - values (torch.Tensor): 状态价值，形状为 (batch_size, 1)。
        """
        # 应用 RSMNorm 对输入进行归一化
        x = self.rsmnorm(x)
        # 应用 SimBa 网络
        hidden = self.net(x)

        # 应用动作头，并计算动作概率
        action_probs = self.action_head(hidden).softmax(dim = -1)
        # 应用价值头，计算状态价值
        values = self.value_head(hidden)

        # 返回动作概率和状态价值
        return action_probs, values

class Critic(Module):
    """
    Critic（评论家）类。

    该类实现了评论家网络，用于在强化学习中估计状态价值。
    它结合了 RSMNorm、SimBa 网络和价值头。

    Args:
        state_dim (int): 状态特征的维度大小。
        hidden_dim (int): 隐藏层的维度大小。
        dim_pred (int, optional): 预测维度的数量，默认为1。
        mlp_depth (int, optional): 多层感知机（MLP）的深度，默认为6。
        dropout (float, optional): Dropout概率，默认为0.1。
        rsmnorm_input (bool, optional): 是否使用 RSMNorm 对输入进行归一化，默认为True。
    """
    def __init__(
        self,
        state_dim,
        hidden_dim,
        dim_pred = 1,
        mlp_depth = 6, # recent paper has findings that show scaling critic is more important than scaling actor
        dropout = 0.1,
        rsmnorm_input = True
    ):
        super().__init__()
        # 如果需要，则应用 RSMNorm；否则，使用恒等映射
        self.rsmnorm = RSMNorm(state_dim) if rsmnorm_input else nn.Identity()

        self.net = SimBa(
            state_dim, # 输入特征的维度大小
            dim_hidden = hidden_dim, # 隐藏层的维度大小
            depth = mlp_depth, # MLP 的深度
            dropout = dropout # Dropout 概率
        )

        # 线性变换，将维度从 `hidden_dim` 转换为 `dim_pred`
        self.value_head = nn.Linear(hidden_dim, dim_pred)

    def forward(self, x):
        """
        前向传播方法，执行评论家网络的计算。

        Args:
            x (torch.Tensor): 输入状态张量，形状为 (batch_size, state_dim)。

        Returns:
            torch.Tensor: 状态价值估计，形状为 (batch_size, dim_pred)。
        """
        # 应用 RSMNorm 对输入进行归一化
        x = self.rsmnorm(x)
        # 应用 SimBa 网络
        hidden = self.net(x)
        # 应用价值头，计算状态价值
        value = self.value_head(hidden)
        # 返回状态价值估计
        return value


def log(t, eps = 1e-20):
    """
    计算输入张量的对数。

    Args:
        t (torch.Tensor): 输入张量。
        eps (float, optional): 用于数值稳定性的一个小常数，默认为1e-20。

    Returns:
        torch.Tensor: 输入张量的对数。
    """
    return t.clamp(min = eps).log()


def entropy(prob):
    """
    计算概率分布的熵。

    Args:
        prob (torch.Tensor): 输入概率分布张量。

    Returns:
        torch.Tensor: 输入概率分布的熵。
    """
    return (-prob * log(prob)).sum()


def model_spectral_entropy_loss(
    model: Module
):
    """
    计算模型的光谱熵损失。

    该函数计算模型参数的光谱熵，并将其作为损失函数的一部分。
    通过惩罚光谱熵，鼓励模型参数的低秩表示，从而实现模型压缩。

    Args:
        model (nn.Module): 需要计算损失的模型。

    Returns:
        torch.Tensor: 光谱熵损失。
    """
    # 初始化损失张量
    loss = tensor(0.).requires_grad_()

    for parameter in model.parameters():
        if parameter.ndim < 2:
            # 如果参数的维度小于2，则跳过
            continue
        
        # 获取参数的形状
        *_, row, col = parameter.shape
        # 重塑参数形状
        parameter = parameter.reshape(-1, row, col)

        # 计算奇异值
        singular_values = torch.linalg.svdvals(parameter)
        # 将奇异值转换为概率分布
        spectral_prob = singular_values.softmax(dim = -1)
        # 计算光谱熵
        spectral_entropy = entropy(spectral_prob)
        # 将光谱熵添加到损失中
        loss = loss + spectral_entropy

    return loss


def simba_orthogonal_loss(
    model: Module
):
    """
    计算SimBa网络中的正交损失。

    该函数通过计算SimBa网络各层线性变换权重之间的余弦相似度，
    并惩罚非正交的权重，从而鼓励权重矩阵的正交性。

    Args:
        model (nn.Module): 需要计算正交损失的模型。

    Returns:
        torch.Tensor: 正交损失。
    """
    # 初始化损失张量，并设置 requires_grad=True
    loss = tensor(0.).requires_grad_()

    # 遍历模型的所有模块
    for module in model.modules():
        if not isinstance(module, SimBa):
            # 如果模块不是 SimBa 类型，则跳过
            continue
        
        # 初始化权重列表
        weights = []

        # 遍历 SimBa 网络的每一层
        for layer in module.layers:
            # 获取输入和输出线性变换层
            linear_in, linear_out = layer.branch[1], layer.branch[3]

            # 添加输入线性变换的转置权重
            weights.append(linear_in.weight.t())
            # 添加输出线性变换的权重
            weights.append(linear_out.weight)

        # 遍历所有权重矩阵
        for weight in weights:
            # 对权重进行归一化，使其具有单位范数
            norm_weight = F.normalize(weight, dim = -1)
            # 计算余弦相似度矩阵
            cosine_dist = einsum(norm_weight, norm_weight, 'i d, j d -> i j')
            # 生成单位矩阵
            eye = torch.eye(cosine_dist.shape[-1], device = cosine_dist.device, dtype = torch.bool)
            # 计算非对角线元素的平均值，作为正交损失
            orthogonal_loss = cosine_dist[~eye].mean()
            # 将正交损失添加到总损失中
            loss = loss + orthogonal_loss

    return loss


class PPG:
    """
    PPG（Proximal Policy Optimization with Generalized Advantage Estimation）类。

    该类实现了基于广义优势估计（GAE）和近端策略优化（PPO）的强化学习算法。
    它结合了行动者网络、评论家网络、自适应优化器、指数移动平均（EMA）等组件。

    Args:
        state_dim (int): 状态特征的维度大小。
        num_actions (int): 动作空间的维度大小。
        actor_hidden_dim (int): 行动者网络隐藏层的维度大小。
        critic_hidden_dim (int): 评论家网络隐藏层的维度大小。
        critic_pred_num_bins (int): 评论家网络预测的离散区间数量。
        reward_range (tuple[float, float]): 奖励范围，用于高斯损失函数。
        epochs (int): 训练轮数。
        minibatch_size (int): 小批量大小。
        lr (float): 学习率。
        betas (tuple[float, float]): Adam优化器的beta参数。
        lam (float): GAE的lambda参数。
        gamma (float): 折扣因子。
        beta_s (float): 策略熵的系数。
        regen_reg_rate (float): 自适应优化器的再生率。
        spectral_entropy_reg (bool): 是否使用光谱熵正则化。
        apply_spectral_entropy_every (int): 光谱熵正则化的应用频率。
        spectral_entropy_reg_weight (float): 光谱熵正则化的权重。
        cautious_factor (float): 自适应优化器的谨慎因子。
        eps_clip (float): PPO的裁剪参数。
        value_clip (float): 价值裁剪参数。
        ema_decay (float): EMA的衰减率。
        save_path (str, optional): 模型保存路径，默认为 './ppg.pt'。
    """
    def __init__(
        self,
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        critic_pred_num_bins,
        reward_range: tuple[float, float],
        epochs,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        regen_reg_rate,
        spectral_entropy_reg,
        apply_spectral_entropy_every,
        spectral_entropy_reg_weight,
        cautious_factor,
        eps_clip,
        value_clip,
        ema_decay,
        save_path = './ppg.pt'
    ):
        # 初始化行动者网络并移动到计算设备
        self.actor = Actor(state_dim, actor_hidden_dim, num_actions).to(device)

        # 初始化评论家网络并移动到计算设备
        self.critic = Critic(state_dim, critic_hidden_dim, dim_pred = critic_pred_num_bins).to(device)

        # 初始化高斯损失函数，用于评论家网络的损失计算
        self.critic_hl_gauss_loss = HLGaussLoss(
            min_value = reward_range[0],
            max_value = reward_range[1],
            num_bins = critic_pred_num_bins,
            clamp_to_range = True
        ).to(device)

        # 初始化行动者和评论家的指数移动平均模型
        self.ema_actor = EMA(self.actor, beta = ema_decay, include_online_model = False, update_model_with_ema_every = 1000)
        self.ema_critic = EMA(self.critic, beta = ema_decay, include_online_model = False, update_model_with_ema_every = 1000)

        # 初始化自适应优化器，用于行动者和评论家网络的参数更新
        self.opt_actor = AdoptAtan2(self.actor.parameters(), lr = lr, betas = betas, regen_reg_rate = regen_reg_rate, cautious_factor = cautious_factor)
        self.opt_critic = AdoptAtan2(self.critic.parameters(), lr = lr, betas = betas, regen_reg_rate = regen_reg_rate, cautious_factor = cautious_factor)

        # 将EMA模型添加到优化器的后处理步骤中
        self.ema_actor.add_to_optimizer_post_step_hook(self.opt_actor)
        self.ema_critic.add_to_optimizer_post_step_hook(self.opt_critic)

        # 小批量大小
        self.minibatch_size = minibatch_size

        # 训练轮数
        self.epochs = epochs

        # GAE的lambda参数
        self.lam = lam
        # 折扣因子
        self.gamma = gamma
        # 策略熵的系数
        self.beta_s = beta_s

        # PPO的裁剪参数
        self.eps_clip = eps_clip
        # 价值裁剪参数
        self.value_clip = value_clip

        # 是否使用光谱熵正则化
        self.spectral_entropy_reg = spectral_entropy_reg
        # 光谱熵正则化的应用频率
        self.apply_spectral_entropy_every = apply_spectral_entropy_every
        # 光谱熵正则化的权重
        self.spectral_entropy_reg_weight = spectral_entropy_reg_weight

        # 模型保存路径
        self.save_path = Path(save_path)

    def save(self):
        """
        保存模型参数。

        将行动者和评论家网络的参数保存到指定路径。
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, str(self.save_path))

    def load(self):
        """
        加载模型参数。

        从指定路径加载行动者和评论家网络的参数。
        """
        if not self.save_path.exists():
            return

        data = torch.load(str(self.save_path))
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])

    def learn(self, memories, next_state):
        """
        执行训练过程。

        该方法使用记忆数据训练行动者和评论家网络。

        Args:
            memories (list): 包含多个记忆数据的列表，每个记忆数据是一个元组，包含状态、动作、旧的对数概率、奖励、结束标志和价值。
            next_state (numpy.ndarray): 下一个状态。
        """
        # 获取高斯损失函数
        hl_gauss = self.critic_hl_gauss_loss

        # 从记忆数据中提取并准备训练数据
        (
            states,
            actions,
            old_log_probs,
            rewards,
            dones,
            values
        ) = zip(*memories)

        # 将动作转换为张量
        actions = [tensor(action) for action in actions]
        # 生成掩码，标记非终止状态
        masks = [(1. - float(done)) for done in dones]

        # 计算广义优势估计（GAE）
        # 将下一个状态转换为张量并移动到计算设备
        next_state = torch.from_numpy(next_state).to(device)
        # 计算下一个状态的价值，并分离计算图
        next_value = self.critic(next_state).detach()

        # 将价值转换为高斯分布的标量表示
        scalar_values = hl_gauss(stack(values))
        # 计算下一个状态的价值
        scalar_next_value = hl_gauss(next_value)

        # 将下一个状态的价值添加到价值列表中
        scalar_values = list(scalar_values) + [scalar_next_value]

        # 初始化回报列表
        returns = []
        # 初始化广义优势估计
        gae = 0
        for i in reversed(range(len(rewards))):
            # 计算优势
            delta = rewards[i] + self.gamma * scalar_values[i + 1] * masks[i] - scalar_values[i]
            # 计算广义优势估计
            gae = delta + self.gamma * self.lam * masks[i] * gae
            # 计算回报并插入到回报列表的开头
            returns.insert(0, gae + scalar_values[i])

        # 将数据转换为 PyTorch 张量
        to_torch_tensor = lambda t: torch.stack(t).to(device).detach()

        # 状态
        states = to_torch_tensor(states)
        # 动作
        actions = to_torch_tensor(actions)
        # 旧的价值
        old_values = to_torch_tensor(values)
        # 旧的对数概率
        old_log_probs = to_torch_tensor(old_log_probs)

        # 奖励
        rewards = tensor(returns).float().to(device)

        # 创建数据加载器，用于策略阶段的训练
        dl = create_shuffled_dataloader([states, actions, old_log_probs, rewards, old_values], self.minibatch_size)

        # 策略阶段的训练，类似于原始的 PPO 算法
        for _ in range(self.epochs):
            for i, (states, actions, old_log_probs, rewards, old_values) in enumerate(dl):
                
                # 计算动作概率
                action_probs, _ = self.actor(states)
                # 计算价值
                values = self.critic(states)
                # 创建类别分布
                dist = Categorical(action_probs)
                # 计算动作的对数概率
                action_log_probs = dist.log_prob(actions)
                # 计算熵
                entropy = dist.entropy()

                # 将旧的价值转换为高斯分布的标量表示
                scalar_old_values = hl_gauss(old_values)

                # 计算 PPO 的裁剪代理目标
                # 计算比率
                ratios = (action_log_probs - old_log_probs).exp()
                # 计算优势
                advantages = normalize(rewards - scalar_old_values.detach())
                # 计算第一个裁剪目标
                surr1 = ratios * advantages
                # 计算第二个裁剪目标
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                # 计算策略损失
                policy_loss = - torch.min(surr1, surr2) - self.beta_s * entropy
                # 添加 SimBa 正交损失
                policy_loss = policy_loss + simba_orthogonal_loss(self.actor)

                if self.spectral_entropy_reg and divisible_by(i, self.apply_spectral_entropy_every):
                    # 添加光谱熵正则化损失
                    policy_loss = policy_loss + model_spectral_entropy_loss(self.actor) * self.spectral_entropy_reg_weight

                # 更新行动者网络的参数
                update_network_(policy_loss, self.opt_actor)

                # 计算裁剪价值损失，并更新价值网络，策略网络和价值网络分开更新
                clip = self.value_clip  # 获取价值裁剪参数

                # 将价值转换为高斯分布的标量表示
                scalar_values = hl_gauss(values)
                
                # 计算裁剪后的价值
                scalar_value_clipped = scalar_old_values + (scalar_values - scalar_old_values).clamp(-clip, clip)
                # 将裁剪后的价值转换为对数概率
                value_clipped_logits = hl_gauss.transform_to_logprobs(scalar_value_clipped)

                # 计算第一个价值损失
                value_loss_1 = hl_gauss(value_clipped_logits, rewards, reduction = 'none')
                # 计算第二个价值损失
                value_loss_2 = hl_gauss(values, rewards, reduction = 'none')

                # 计算平均价值损失
                value_loss = torch.mean(torch.max(value_loss_1, value_loss_2))

                # 添加 SimBa 正交损失
                value_loss = value_loss + simba_orthogonal_loss(self.critic)

                if self.spectral_entropy_reg and divisible_by(i, self.apply_spectral_entropy_every):
                    # 添加光谱熵正则化损失
                    value_loss = value_loss + model_spectral_entropy_loss(self.critic) * self.spectral_entropy_reg_weight

                # 更新评论家网络的参数
                update_network_(value_loss, self.opt_critic)


def main(
    env_name = 'LunarLander-v3',
    num_episodes = 50000,
    max_timesteps = 500,
    actor_hidden_dim = 64,
    critic_hidden_dim = 256,
    critic_pred_num_bins = 100,
    reward_range = (-100, 100),
    minibatch_size = 64,
    lr = 0.0008,
    betas = (0.9, 0.99),
    lam = 0.95,
    gamma = 0.99,
    eps_clip = 0.2,
    value_clip = 0.4,
    beta_s = .01,
    regen_reg_rate = 1e-4,
    spectral_entropy_reg = False,
    apply_spectral_entropy_every = 4,
    spectral_entropy_reg_weight = 0.025,
    cautious_factor = 0.1,
    ema_decay = 0.9,
    update_timesteps = 5000,
    epochs = 2,
    seed = None,
    render = True,
    render_every_eps = 250,
    save_every = 1000,
    clear_videos = True,
    video_folder = './lunar-recording',
    load = False
):
    """
    主训练函数。

    该函数设置强化学习环境，初始化PPG代理，并执行训练过程。

    Args:
        env_name (str, optional): 环境名称，默认为 'LunarLander-v3'。
        num_episodes (int, optional): 总训练轮数，默认为50000。
        max_timesteps (int, optional): 每轮的最大时间步数，默认为500。
        actor_hidden_dim (int, optional): 行动者网络的隐藏层维度，默认为64。
        critic_hidden_dim (int, optional): 评论家网络的隐藏层维度，默认为256。
        critic_pred_num_bins (int, optional): 评论家网络预测的离散区间数量，默认为100。
        reward_range (tuple, optional): 奖励范围，用于高斯损失函数，默认为 (-100, 100)。
        minibatch_size (int, optional): 小批量大小，默认为64。
        lr (float, optional): 学习率，默认为0.0008。
        betas (tuple, optional): Adam优化器的beta参数，默认为 (0.9, 0.99)。
        lam (float, optional): GAE的lambda参数，默认为0.95。
        gamma (float, optional): 折扣因子，默认为0.99。
        eps_clip (float, optional): PPO的裁剪参数，默认为0.2。
        value_clip (float, optional): 价值裁剪参数，默认为0.4。
        beta_s (float, optional): 策略熵的系数，默认为0.01。
        regen_reg_rate (float, optional): 自适应优化器的再生率，默认为1e-4。
        spectral_entropy_reg (bool, optional): 是否使用光谱熵正则化，默认为False。
        apply_spectral_entropy_every (int, optional): 光谱熵正则化的应用频率，默认为4。
        spectral_entropy_reg_weight (float, optional): 光谱熵正则化的权重，默认为0.025。
        cautious_factor (float, optional): 自适应优化器的谨慎因子，默认为0.1。
        ema_decay (float, optional): EMA的衰减率，默认为0.9。
        update_timesteps (int, optional): 每隔多少时间步更新一次策略，默认为5000。
        epochs (int, optional): 训练轮数，默认为2。
        seed (int, optional): 随机种子，如果为None，则不设置。
        render (bool, optional): 是否渲染环境，默认为True。
        render_every_eps (int, optional): 每隔多少轮渲染一次环境，默认为250。
        save_every (int, optional): 每隔多少轮保存一次模型，默认为1000。
        clear_videos (bool, optional): 是否清除之前的视频记录，默认为True。
        video_folder (str, optional): 视频保存文件夹，默认为 './lunar-recording'。
        load (bool, optional): 是否加载预训练模型，默认为False。
    """
    # 创建环境，并设置渲染模式为 'rgb_array'
    env = gym.make(env_name, render_mode = 'rgb_array')

    if render:
        if clear_videos:
            # 如果需要清除视频，则删除视频文件夹
            rmtree(video_folder, ignore_errors = True)

        # 使用 gym.wrappers.RecordVideo 包装环境，以便记录视频
        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = video_folder,
            name_prefix = 'lunar-video',
            episode_trigger = lambda eps_num: divisible_by(eps_num, render_every_eps),
            disable_logger = True
        )

    # 获取状态空间的维度
    state_dim = env.observation_space.shape[0]
    # 获取动作空间的大小
    num_actions = env.action_space.n

    # 初始化记忆数据队列
    memories = deque([])

    agent = PPG(
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        critic_pred_num_bins,
        reward_range,
        epochs,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        regen_reg_rate,
        spectral_entropy_reg,
        apply_spectral_entropy_every,
        spectral_entropy_reg_weight,
        cautious_factor,
        eps_clip,
        value_clip,
        ema_decay,
    )

    if load:
        # 如果需要加载预训练模型，则加载模型参数
        agent.load()

    if exists(seed):
        # 设置 PyTorch 的随机种子
        torch.manual_seed(seed)
        # 设置 NumPy 的随机种子
        np.random.seed(seed)
    
    # 初始化时间步计数
    time = 0
    # 初始化策略更新计数
    num_policy_updates = 0

    # 遍历每一轮训练，并显示进度条
    for eps in tqdm(range(num_episodes), desc = 'episodes'):
        
        # 重置环境，获取初始状态
        state, info = env.reset(seed = seed)

        # 增加时间步计数
        for timestep in range(max_timesteps):
            time += 1

            # 将状态转换为张量并移动到计算设备
            state = torch.from_numpy(state).to(device)
            # 获取行动者网络的动作概率
            action_probs, _ = agent.ema_actor.forward_eval(state)
            # 获取评论家网络的估计价值
            value = agent.ema_critic.forward_eval(state)

            # 创建类别分布
            dist = Categorical(action_probs)
            # 采样动作
            action = dist.sample()
            # 计算动作的对数概率
            action_log_prob = dist.log_prob(action)
            # 获取动作的数值
            action = action.item()

            # 执行动作，获取下一个状态、奖励和结束标志
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # 判断是否结束
            done = terminated or truncated

            # 创建记忆数据
            memory = Memory(state, action, action_log_prob, reward, done, value)
            # 将记忆数据添加到记忆队列中
            memories.append(memory)

            # 更新状态
            state = next_state

            if divisible_by(time, update_timesteps):
                # 如果达到更新间隔，则训练代理
                agent.learn(memories, next_state)
                # 增加策略更新计数
                num_policy_updates += 1
                # 清空记忆队列
                memories.clear()

            if done:
                # 如果结束，则跳出当前轮的训练
                break

        if divisible_by(eps, save_every):
            # 如果达到保存间隔，则保存模型参数
            agent.save()


if __name__ == '__main__':
    
    # 使用 Fire 库将 main 函数转换为命令行接口
    fire.Fire(main)
