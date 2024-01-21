from torchvision import transforms
from IPython.display import HTML
from PIL import Image
import torch
import Encoder

# 读取图片
image_path = 'mnist_test_5.png'
image = Image.open(image_path)

# 将 image 转为 PyTorch 张量
transf = transforms.ToTensor()
image_tensor = transf(image)

# 定义时间步长
num_steps = 100

# 对图片进行编码
Spike_image_data = Encoder.PoissonEncoding(image_tensor, num_steps)
#Spike_image_data = Encoder.TimeToFirstSpike(image_tensor, num_steps, tau=5, threshold=0.1, bias=1e-4, normalize=False, linear=False)
#Spike_image_data = Encoder.WeightedPhase(image_tensor, num_steps, 8)
#Spike_image_data = Encoder.BurstSpike(image_tensor, num_steps)

# 展示散点图
#Encoder.SpikeShowScatter(Spike_image_data[:, 2], num_steps)

# 展示动态图
#ani = Encoder.SpikeShowDynamic(Spike_image_data, num_steps=num_steps, interval=50)

# 所有时间步长的脉冲累加后重建得到的图像
Encoder.SpikeShowReconv(Spike_image_data)

# 在Jupyter Notebook中显示动画
#HTML(ani.to_jshtml())
