import torch
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def SpikeShowReconv(
        img_data,
        type="gray",
        path='output_image.png'
):
    # type:默认为灰度图
    # 显示重建后的图像
    image_data = torch.sum(img_data, dim=0)
    image_data = image_data.byte().numpy()

    if type == "gray":
        image = Image.fromarray(np.squeeze(image_data))
    elif type == "rgb":
        image = Image.fromarray(np.transpose(image_data, (1, 2, 0)))

    image.save(path)


def SpikeShowScatter(
        img_data,
        num_steps,
):
    # 将张量转为二维
    img_data_2 = img_data.view(num_steps, -1)

    indices = (img_data_2 == 1).nonzero()

    # 提取索引的第一维度作为横坐标，第二维度作为纵坐标
    x_values = indices[:, 0].numpy()
    y_values = indices[:, 1].numpy()

    plt.figure(figsize=(10, 5))
    plt.scatter(x_values, y_values, color='blue', marker='o', label='Points with Value', s=5)
    x_min, x_max = -2, num_steps + 1
    plt.xlim(x_min, x_max)

    plt.title('Scatter Plot')
    plt.xlabel('Time Step')
    plt.ylabel('Neuron Index')
    plt.show()


def SpikeShowDynamic(
        img_data,
        num_steps,
        interval=100
):
    # 创建一个显示张量的动画
    fig, ax = plt.subplots()
    img = ax.imshow(img_data[0].permute(1, 2, 0), cmap='viridis', animated=True)

    text = ax.text(0.98, 0.95, '', color='white', fontsize=12, ha='right', va='top', transform=ax.transAxes,
                   bbox=dict(facecolor='black', alpha=0.7))
    # 关闭坐标轴
    ax.axis('off')

    def update(step):
        img.set_data(img_data[step].permute(1, 2, 0).numpy())
        text.set_text(f'step: {step}/{num_steps}')
        return img, text

    # 设置动画参数
    ani = FuncAnimation(fig, update, frames=num_steps, interval=interval)
    plt.show()
    return ani


def PoissonEncoding(
        data,
        num_steps,
):
    data_time = data.repeat([num_steps] + torch.ones(len(data.size()), dtype=int).tolist())
    data_spike = torch.bernoulli(data_time)

    return data_spike


def TimeToFirstSpike(
        data,
        num_steps,
        tau=5,
        threshold=0.01,
        bias=1e-7,
        normalize=False,
        linear=False
):
    # tau为时间常数，该值影响单个首脉冲时间的大小，不影响各个脉冲之间的相对时间大小
    # threshold:输入数据应大于该设定阈值，若小于该阈值，会对该输入值进行处理；阈值越大，整体首脉冲时间都会变大
    # bias:偏置大小，对输入数据小于阈值的数据进行加偏置处理，该值影响最大首脉冲时间的值，值越大，脉冲发放时间越小
    # normalize:是否对得到的所有像素点的脉冲触发时间进行归一化处理，默认值为False
    # linear:线性表示，输入数据与脉冲触发时间之间的默认关系为t=tau*ln(data/data-threshold),若linear为True，则t=tau*(1-data),
    if not linear:
        data = torch.clamp(data, threshold + bias)
        spike_times = tau * torch.log(data / (data - threshold))
    else:
        spike_times = torch.clamp_max((-tau * (data - 1)), -tau * (threshold - 1))

    if normalize:
        spike_times = (spike_times - torch.min(spike_times)) * (1 / (torch.max(spike_times) - torch.min(spike_times))) * (num_steps - 1)

    spike_data = torch.zeros((num_steps,) + spike_times.shape)
    spike_data = (spike_data.scatter(0, torch.round(torch.clamp_max(spike_times, num_steps - 1)).long().unsqueeze(0), 1))

    return spike_data


def WeightedPhase(
        data,
        num_steps,
        cycle=8
):
    # 参考论文:
    # Kim, J., Kim, H., Huh, S., Lee, J., and Choi, K. (2018c). Deep neural networks with weighted spikes. Neurocomputing 311, 373–386.
    # cycle为编码周期

    T = cycle
    # 设置最大上限  输入值的取值范围应为[0, 2^-T]
    maxweight = 1 - 2 ** -T
    image_tensor = torch.clamp(data * maxweight, 0, maxweight)

    spike_time = torch.zeros((num_steps,) + image_tensor.shape)
    w = 0.5
    # 对输入数据进行小数形式的二进制编码
    for i in range(T):
        spike_time[i] = image_tensor >= w
        image_tensor -= spike_time[i] * w
        w *= 0.5
    # 在总时间步上进行周期性赋权重
    for i in range(T, num_steps):
        spike_time[i] = spike_time[i - T]

    return spike_time


def BurstSpike(
        data,
        num_steps,
        Nmax=5,
        Tmax=10,
        Tmin=2,
):
    # 参考论文：
    # Guo W, Fouda M E, Eltawil A M, et al. Neural coding in spiking neural networks: A comparative study for robust neuromorphic systems[J]. Frontiers in Neuroscience, 2021, 15: 638474.
    # Nmax为设定的脉冲的最大数量
    # Tmax为设定的最大间隔时间
    # Tmin为设定的最小间隔时间
    #
    # 脉冲间间隔ISI的计算方法:
    # ISI = P * (Tmin - Tmax) + Tmax   Ns > 1,
    # ISI = Tmax   otherwise

    # 添加时间步长维度
    Spike_data = torch.zeros((num_steps,) + data.shape)
    Ns = torch.ceil(data * Nmax)
    Ns_over_1_idx = Ns > 1
    # 得到每个像素的脉冲间间隔
    interval_time = torch.ceil(data * (Tmin - Tmax) + Tmax) * Ns_over_1_idx + Tmax * ~Ns_over_1_idx
    # 在总时间步长上进行处理
    for t in range(num_steps):
        Spike_data[t] = (t % interval_time == 0) & (t // interval_time < Ns)

    return Spike_data
