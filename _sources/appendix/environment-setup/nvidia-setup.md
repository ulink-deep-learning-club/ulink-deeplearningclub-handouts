(nvidia-setup)=
# NVIDIA驱动与CUDA：让GPU真正工作

系统装好了，SSH能连上了，然后呢？`python -c "import torch; print(torch.cuda.is_available())"` 返回 `False`——你的GPU还没"醒"过来。

让NVIDIA GPU在Linux上正常工作需要三层软件：**驱动** → **CUDA Toolkit** → **深度学习框架**。每一层都有版本兼容问题，配错了就报错。

> **文档更新**：2026年4月。NVIDIA驱动版本迭代很快，本文信息以当时最新的R595/R580分支为准。建议在[NVIDIA官方驱动下载页](https://www.nvidia.com/en-us/drivers/)确认最新版本。

(nvidia-arch)=
## 显卡架构简史

每张NVIDIA显卡都基于一个架构代号，不同架构支持的驱动版本、计算特性、深度学习能力都有差异。了解架构是选卡和排错的第一步。

```{list-table} NVIDIA GPU架构一览（2014–2026）
:header-rows: 1

* - **架构**
  - **发布年份**
  - **代表产品**
  - **Tensor Core**
  - **RT Core**
  - **驱动支持状态**
* - Maxwell
  - 2014
  - GTX 750/900系列
  - 无
  - 无
  - R580最后支持，2025终止
* - Pascal
  - 2016
  - GTX 10系列、Tesla P100
  - 无
  - 无
  - R580最后支持，2025终止
* - Volta
  - 2017
  - Tesla V100、Titan V
  - 第1代（FP16）
  - 无
  - R580最后支持，2025终止
* - Turing
  - 2018
  - RTX 20系列、GTX 16系列、T4
  - 第2代
  - 第1代
  - R590+
* - Ampere
  - 2020
  - RTX 30系列、A100、A40
  - 第3代（TF32+BFP+稀疏）
  - 第2代
  - R590+
* - Ada Lovelace
  - 2022
  - RTX 40系列、L40S
  - 第4代
  - 第3代
  - R590+
* - Hopper
  - 2022
  - H100、H200
  - 第4代（FP8+Transformer）
  - 无
  - R590+
* - Blackwell
  - 2024
  - RTX 50系列、B200
  - 第5代（FP4）
  - 第4代（神经渲染）
  - 当前
```

各代架构的关键技术跳跃：

- **Pascal → Volta**：Tensor Core诞生，深度学习训练速度提升10倍
- **Volta → Turing**：RT Core加入，Tensor Core首次进入消费级显卡
- **Turing → Ampere**：TF32精度、BF16支持、稀疏加速，算力翻倍
- **Ampere → Hopper**：Transformer Engine、FP8，专为大语言模型优化
- **Hopper → Blackwell**：FP4推理、所有核心支持FP32+INT32并发、神经渲染

## 驱动版本与架构支持

NVIDIA的驱动按Release Branch组织，每季度更新。不同分支支持的GPU架构不同：

```{list-table} 驱动分支与架构支持（2026年4月）
:header-rows: 1

* - **驱动分支**
  - **最新版本**
  - **发布日期**
  - **支持的架构**
  - **状态**
* - R595
  - 596.36
  - 2026-04-28
  - Turing ~ Blackwell
  - 当前Game Ready
* - R580
  - 582.53
  - 2026-04-28
  - Maxwell ~ Blackwell
  - 当前Enterprise LTSB
* - R570
  - 573.96
  - 2026-01
  - Maxwell ~ Blackwell
  - 2026年2月EOL
* - R535
  - 539.72
  - 2026-04
  - Kepler ~ Blackwell
  - 维护模式
```

**关键事件时间线**：

- **2021年**：R470最后支持Kepler架构（GTX 600/700系列）
- **2025年10月**：R580发布，最后一次Game Ready驱动更新支持Maxwell、Pascal、Volta
- **2025年Q4起**：上述架构转为季度安全更新（至2028年10月）
- **2026年2月**：R570分支正式EOL
- **2026年4月**：R595（游戏）/ R580（企业）为当前活跃分支

```{admonition} 如何选择驱动分支
:class: tip

- **游戏/开发机**：选R595 Game Ready，有最新功能优化
- **生产服务器**：选R580 Enterprise LTSB，稳定优先，支持到2028年
- **旧卡用户**：如果持有GTX 900/10系列或Titan V，最多只能用到R580系列的最后版本
```

## 一键安装：三行命令搞定

Debian/Ubuntu上装NVIDIA驱动最可靠的方式是通过官方APT源：

```bash
# 1. 检测你的显卡型号和推荐驱动
nvidia-detect  # 输出类似 "nvidia-driver-570"

# 2. 添加NVIDIA官方APT源（Ubuntu LTS用户）
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# 3. 安装推荐驱动（将570替换为detect输出的版本）
sudo apt install nvidia-driver-570

# 4. 重启
sudo reboot
```

重启后验证：

```bash
# 基本验证
nvidia-smi

# 输出应该是类似这样的：
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 570.86.15    Driver Version: 570.86.15    CUDA Version: 12.8     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  Tesla T4            Off  | 00000000:00:1E.0 Off |                    0 |
# | N/A   48C    P0    28W /  70W |      0MiB / 15360MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

```{admonition} Secure Boot警告
:class: warning

如果系统启用了Secure Boot（很多预装Ubuntu/Debian的机器默认开启），安装NVIDIA专有驱动后需要注册MOK（Machine Owner Key）：重启时会进入蓝色MOK管理界面，选择"Enroll MOK"→"Continue"→输入密码→重启。如果不做这一步，驱动不会加载，`nvidia-smi`会报错。
```

### 驱动装不上？常见原因

| 症状 | 原因 | 解决 |
|------|------|------|
| `nvidia-smi: command not found` | 驱动没装 | 先 `nvidia-detect` 确认，然后 `sudo apt install nvidia-driver-XXX` |
| `Failed to initialize NVML: Driver/library version mismatch` | 驱动更新后内核模块没重载 | 重启，或 `sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia` |
| `ERROR: You appear to be running an X server` | 有图形界面在运行 | `sudo service lightdm stop` 后重试安装 |
| 装完重启黑屏 | 驱动与显卡不匹配 或 Secure Boot 阻止 | 进恢复模式卸载驱动，换版本重试，或禁用 Secure Boot |

## nvidia-smi 深入解读

`nvidia-smi` 是你的GPU仪表盘。不只是看一眼温度和显存——它告诉你的信息远比表面多：

```bash
# 基本（一秒钟刷新一次）
watch -n 1 nvidia-smi

# 只看关键指标（适合监控训练）
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

# 输出示例：
# index, name, temperature.gpu, utilization.gpu, memory.used, memory.total
# 0, Tesla T4, 48, 0 %, 0 MiB, 15360 MiB
```

```{list-table} nvidia-smi 输出关键字段
:header-rows: 1

* - **字段**
  - **含义**
  - **关注点**
* - Temp
  - GPU核心温度（°C）
  - 超过85°C说明散热不足，会降频
* - Perf
  - 性能状态（P0-P12）
  - P0=最高性能，P8+表示降频了
* - Pwr:Usage/Cap
  - 当前功耗/最大功耗
  - 远低于Cap可能没在满负荷跑
* - Memory-Usage
  - 显存使用量
  - 接近上限说明模型太大，需降batch或换卡
* - GPU-Util
  - 计算单元利用率
  - 持续＜80%说明CPU/IO瓶颈
* - Volatile GPU-Util
  - 实际SM占用率（更准）
  - 和GPU-Util一起看
* - Compute M.
  - 计算模式
  - Default / Exclusive_Process / PROHIBITED
```

```{admonition} GPU-Util 低不代表有问题
:class: note

深度学习训练中GPU-Util低可能有多种原因：数据加载太慢（IO瓶颈）、CPU预处理跟不上、batch size过小、或者模型结构本身计算密度低（如小CNN）。可以尝试用 `nvidia-smi dmon` 查看更细粒度的指标。
```

### ECC与显存类型

企业级GPU（如A100、H100、V100）使用HBM/HBM2/HBM3显存，**默认开启ECC**。ECC纠错码会占用约6-12%的显存容量，但保证了计算正确性。可以用以下命令查询：

```bash
nvidia-smi -q -d ECC
nvidia-smi --query-gpu=ecc.mode.current --format=csv
```

消费级GPU（RTX系列）使用GDDR显存（GDDR6/GDDR6X/GDDR7），**没有ECC**。对大多数深度学习训练来说，GDDR的可靠性足够，但长时间大规模训练（数周级别的HPC任务）中，HBM+ECC是标配。

```{list-table} 显存类型对比
:header-rows: 1

* - **类型**
  - **显卡示例**
  - **带宽**
  - **ECC**
  - **适用场景**
* - GDDR6
  - RTX 3060~4060
  - 192~480 GB/s
  - 无
  - 训练/推理入门
* - GDDR6X
  - RTX 3090/4090
  - 700~1000 GB/s
  - 无
  - 消费级高性能
* - GDDR7
  - RTX 5090
  - 1792 GB/s
  - 无
  - 消费级旗舰
* - HBM2
  - V100/P100
  - 732~900 GB/s
  - 有（6-12%开销）
  - 遗留企业级
* - HBM2e
  - A100
  - 1935~2039 GB/s
  - 有
  - 上一代数据中心
* - HBM3
  - H100
  - 3350 GB/s
  - 有
  - 当前数据中心
* - HBM3e
  - H200/B200
  - 4800~8000 GB/s
  - 有
  - 最新数据中心
```

(cuda-compat)=
## CUDA版本管理

```{admonition} CUDA Toolkit版本 ≠ nvidia-smi显示的CUDA Version
:class: warning

`nvidia-smi` 输出的"CUDA Version"是**驱动支持的CUDA最大版本**（Driver API版本），不是实际安装的CUDA Toolkit版本。你可以装CUDA 11.8 Toolkit在CUDA 12.8的驱动上跑（向下兼容），但不能反过来。
```

快速确认你的深度学习框架在用哪个CUDA版本：

```bash
# PyTorch
python -c "import torch; print(torch.version.cuda)"

# TensorFlow
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cuda_version'])"

# 查看系统CUDA Toolkit版本
/usr/local/cuda/bin/nvcc --version
```

```{list-table} NVIDIA驱动与CUDA Toolkit的兼容关系
:header-rows: 1

* - **驱动分支**
  - **最大CUDA版本**
  - **最低CUDA版本**
  - **建议配对**
* - R595/R590
  - CUDA 13.x
  - CUDA 12.0
  - 最新项目用CUDA 12.8+
* - R580
  - CUDA 12.8
  - CUDA 12.0
  - 生产环境配CUDA 12.4~12.8
* - R570
  - CUDA 12.8
  - CUDA 11.8
  - 遗留项目
* - R535
  - CUDA 12.2
  - CUDA 11.0
  - 旧卡兼容
```

建议的CUDA安装方式（不用NVIDIA官网的runfile，而是用包管理器）：

```bash
# 先确认驱动版本够用
nvidia-smi  # 看顶部CUDA Version

# 用pip安装cuda-toolkit（推荐，不污染系统）
pip install nvidia-cuda-toolkit

# 或直接用框架自带的CUDA（PyTorch自带）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

```{admonition} 什么时候需要手动装CUDA Toolkit？
:class: tip

大多数情况下你**不需要**手动安装CUDA Toolkit。PyTorch/TensorFlow的预编译包已经带了它们需要的CUDA运行时库。只有在你需要自己编译CUDA扩展（如`flash-attn`）或者使用NVIDIA的底层库（如`cuBLAS`、`cuDNN`）的开发API时，才需要装完整的CUDA Toolkit。
```

(pytorch-gpu)=
## PyTorch的显卡支持淘汰节奏

驱动和CUDA Toolkit的版本只是第一层兼容。PyTorch有自己的**编译时支持的架构列表**——即使你的驱动和CUDA Toolkit够新，如果PyTorch二进制没编译你家显卡的架构代码，照样跑不了。

验证方法：

```python
import torch
print(torch.cuda.get_arch_list())
# 输出类似: ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
# sm_70=Volta，sm_75=Turing，sm_80=Ampere，sm_90=Hopper，sm_100=Ada，sm_120=Blackwell
```

PyTorch的显卡支持随版本演进不断抬升最低门槛：

```{list-table} PyTorch版本与CUDA架构支持
:header-rows: 1

* - **PyTorch版本**
  - **CUDA构建**
  - **支持的最小架构**
  - **不支持的显卡**
* - **2.6.0**
  - 12.4
  - Maxwell (sm_50)
  - 无
* - 2.7~2.10
  - 12.6
  - Maxwell (sm_50)
  - 无
* - 2.8~2.10
  - 12.8
  - Volta (sm_70)
  - GTX 900/10系列
* - **2.11+**
  - **12.8**
  - **Turing (sm_75)**
  - **V100、Titan V、GTX 900/10系列**
* - 2.12+
  - 13.0
  - Turing (sm_75)
  - 同上
```

```{admonition} 如果你的显卡被PyTorch最新版抛弃了怎么办？
:class: important

你有三条路可选：

1. **用旧版PyTorch**：V100、GTX 1080 Ti等Volta/Pascal显卡。值得考虑的版本是：
   ~~~bash
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   ~~~
2. **自行编译**：从源码编译PyTorch时指定 `TORCH_CUDA_ARCH_LIST="7.0"` 来包含你的架构。
3. **换卡**：一张二手RTX 3060 12GB（~1000~1500 RMB）支持sm_86，未来几年都不会被抛弃。

简单规则：**什么时候该考虑升级显卡？** 当PyTorch最新版不再编译你的架构（`get_arch_list()`里找不到你的sm版本），且你想用的新模型依赖新版PyTorch的新特性时。
```

当前（2026年4月）PyTorch最新版使用的CUDA 12.8构建已移除Volta（sm_70，即V100/Titan V）支持。V100曾经是{doc}`../../pytorch-practice/index`中最常用的GPU之一，现在也进入了"遗留硬件"行列。

## Compute Capability 速查

PyTorch用sm版本号（如sm_75）标识显卡架构。想知道你的卡对应哪个sm：

```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
# 输出示例: (8, 6) 表示 sm_86
```

| sm版本 | 架构 | 代表显卡 |
|--------|------|---------|
| sm_50 | Maxwell | GTX 750 Ti、GTX 960、GTX 980 |
| sm_52 | Maxwell | GTX 980 Ti、Titan X (Maxwell) |
| sm_60 | Pascal | GTX 1080、GTX 1070、GTX 1060 |
| sm_61 | Pascal | GTX 1080 Ti、Titan Xp |
| sm_70 | Volta | Titan V、Tesla V100 |
| sm_75 | Turing | RTX 2080 Ti、RTX 2080、RTX 2070、GTX 1660、T4 |
| sm_80 | Ampere | A100 |
| sm_86 | Ampere | RTX 3090、RTX 3080、RTX 3070、RTX 3060、A10、A40 |
| sm_89 | Ada | RTX 4090、RTX 4080、RTX 4070、RTX 4060、L40S |
| sm_90 | Hopper | H100、H200 |
| sm_120 | Blackwell | RTX 5090、RTX 5080、B200 |

```{admonition} sm版本有什么用？
:class: note

- 确认你的卡是否被当前PyTorch支持：`torch.cuda.get_arch_list()` 里有没有你的sm
- 自行编译CUDA扩展时，`TORCH_CUDA_ARCH_LIST` 需要填对
- 买卡前查sm版本就能预判这张卡未来几年的PyTorch支持窗口
```

(docker-gpu)=
## Docker GPU 支持：一劳永逸的环境隔离

驱动和CUDA Toolkit的版本兼容问题是深度学习环境配置中最容易出错的环节。**Docker + nvidia-container-toolkit** 可以一步解决这个问题：宿主机只需要装驱动，PyTorch、CUDA Toolkit、cuDNN这些都放在容器里，互不干扰。

安装配置：

```bash
# 1. 确保宿主机驱动已装好（nvidia-smi 能正常输出）
nvidia-smi

# 2. 安装 nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit

# 3. 配置 Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4. 测试
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

之后跑任何深度学习代码只需要拉一个带CUDA的PyTorch镜像：

```bash
# 使用PyTorch官方镜像（自带CUDA 12.4 + cuDNN）
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  python train.py

# 或使用NVIDIA的PyTorch容器（已预装所有优化）
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:24.12-py3 \
  python train.py
```

```{list-table} 裸机 vs Docker 的环境管理对比
:header-rows: 1

* - **维度**
  - **裸机直接装**
  - **Docker容器**
* - 环境隔离
  - 冲突时难以清理
  - 每个项目独立容器
* - 版本切换
  - 需要卸载重装CUDA Toolkit
  - 换一个镜像标签即可
* - 团队协作
  - 每个人装的不一样
  - 共享Dockerfile保证一致
* - 复现
  - 依赖系统包的版本
  - 镜像锁定所有依赖
* - 磁盘占用
  - 一套环境
  - 每个镜像几GB
```

```{admonition} 什么时候不用Docker？
:class: tip

- 单用户、单项目的开发机，Docker的隔离优势不明显
- 多个Docker镜像同时挂载大训练集可能浪费磁盘空间
- 需要访问宿主机特定硬件（USB设备、HCA网卡）时配置复杂

权衡：Docker最适合**多人共用服务器**或**生产环境部署**。个人开发机装裸机更直接，但要小心不要污染系统Python环境（{doc}`linux-basics`的uv小节已经讲了解决方案）。
```

(gpu-selection)=
## GPU选型指南

### 产品线定位

```{list-table} NVIDIA产品线
:header-rows: 1

* - **产品线**
  - **定位**
  - **显存**
  - **互连**
  - **典型价格**
  - **适合**
* - GTX
  - 入门消费
  - 4~12 GB GDDR
  - 无
   - <¥2,000
  - 学习、小模型
* - RTX x050/60
  - 主流消费
  - 8~16 GB GDDR
  - 无
   - ¥1,800~¥4,000
  - 微调、推理
* - RTX x070/80
  - 高端消费
  - 12~32 GB GDDR7
  - 无
   - ¥4,000~¥14,000
  - 训练中型模型
* - RTX Pro
  - 工作站
  - 24~48 GB GDDR6/7
  - 无
   - ¥22,000~¥72,000
  - 专业可视化+AI
* - Tesla/Data Center
  - 数据中心
  - 40~192 GB HBM
  - NVLink
   - ¥72,000~¥216,000+
  - 大规模训练/生产
```

### 消费级 vs 企业级：核心差异

消费级显卡（RTX）和企业级（A/H/B系列）在深度学习场景下的差异可能出乎你的意料：

1. **显存瓶颈**：RTX 5090有32 GB GDDR7，而P100（2016）有16 GB HBM2。十年过去，消费级旗舰的显存才翻了2倍，而模型大小翻了1000倍。**显存通常是第一瓶颈**。
2. **单精度算力倒挂**：RTX 5050（2025最低端）的FP32算力（约11 TFLOPS）和十年前的P100（10.6 TFLOPS）相当，但**5050有5代Tensor Core而P100一个都没有**——用FP16/INT8推理时5050甩P100几条街。
3. **FP64基本没变**：消费级显卡的FP64算力始终被锁定在FP32的1/64~1/32。如果你需要做高精度科学计算，必须上企业级（A100的FP64是RTX 5090的12倍）。
4. **多卡互联**：消费级没有NVLink，多卡通信只能走PCIe。数据中心卡通过NVLink可以达到600~1800 GB/s的卡间带宽，这是训练大模型的关键。
5. **显存带宽差距**：这是最容易被忽视的差异。HBM显存（企业级标配）的带宽远超同代GDDR，直接影响训练吞吐：

```{list-table} 显存带宽对比
:header-rows: 1

* - **显卡**
  - **显存类型**
  - **显存容量**
  - **带宽**
  - **总线位宽**
* - Tesla P100
  - HBM2
  - 16 GB
  - 732 GB/s
  - 4096-bit
* - Tesla V100
  - HBM2
  - 16/32 GB
  - 900 GB/s
  - 4096-bit
* - Tesla A100
  - HBM2e
  - 40/80 GB
  - 2039 GB/s
  - 5120-bit
* - RTX 3090
  - GDDR6X
  - 24 GB
  - 936 GB/s
  - 384-bit
* - RTX 4090
  - GDDR6X
  - 24 GB
  - 1008 GB/s
  - 384-bit
* - RTX 5090
  - GDDR7
  - 32 GB
  - 1792 GB/s
  - 512-bit
* - H100
  - HBM3
  - 80 GB
  - 3350 GB/s
  - 5120-bit
```

注意一个关键对比：**P100（2016）的732 GB/s HBM2带宽比RTX 4060（2024）的272 GB/s GDDR6快近3倍**。企业级显卡的HBM显存通过超宽总线（4096-bit起）实现高带宽，消费级GDDR则靠高频率和窄总线（128~512-bit）。这就是为什么二手P100/V100即使算力落后，在训练吞吐上依然能和一些入门消费卡抗衡——带宽够大。不过也别忘了{ref}`pytorch-gpu`中提到的框架兼容性问题——算力再高跑不了新版PyTorch也不行。

### 怎么选：两个核心场景

买卡之前先看清楚自己属于哪种情况：

**场景一：手里没有GPU，想低成本入门**

淘汰的数据中心卡是性价比极高的选择。它们退役后被大批量抛售到二手市场，算力和显存在今天依然能打：

| 显卡 | 显存 | 二手参考价（闲鱼） | Tensor Core | FP32 | 入手价值 |
|------|------|-------------------|-------------|------|---------|
| Tesla P100 16GB | 16 GB HBM2 | 300~600 RMB | 无 | 10.6 TFLOPS | 极致低价入门，适合跑传统CNN |
| Tesla V100 16GB | 16 GB HBM2 | ~1000 RMB | 第1代（FP16） | 15.7 TFLOPS | Tensor Core入门，跑小Transformer |
| Tesla V100 32GB | 32 GB HBM2 | ~3000 RMB | 第1代（FP16） | 15.7 TFLOPS | 大显存低成本方案 |

```{admonition} 买淘汰企业级显卡的注意事项
:class: warning

- 这些卡**没有主动散热风扇**（被动散热），需要服务器风道或自己加装风扇
- 通常需要**专用电源线**（EPS 8-pin，不是PCIE 8-pin）
- P100/V100基于Pascal/Volta架构，**PyTorch新版CUDA 12.8构建已放弃支持**——需要用PyTorch 2.10的CUDA 12.6构建，或自行编译
- V100的Tensor Core只支持FP16，不支持TF32/BF16/INT8等后续精度
- 驱动方面R580是最后支持分支，可用至2028年安全更新
```

**场景二：手上有卡想升级，或者预算充足**

RTX 20系及以上都值得考虑。深度学习场景下，**显存大小比算力重要得多**——一张老卡只要显存够大，跑大模型的能力就比新卡更强：

```{list-table} 消费级升级路径（先看显存，再看架构）
:header-rows: 1

* - **推荐优先级**
  - **显卡**
  - **显存**
  - **架构**
  - **二手参考价**
  - **适合做什么**
* - ⭐⭐⭐⭐⭐
  - RTX 3090
  - 24 GB GDDR6X
  - Ampere
  - ~3000~5000 RMB
  - **性价比之王**，24GB能微调7B模型
* - ⭐⭐⭐⭐⭐
  - RTX 4090
  - 24 GB GDDR6X
  - Ada
  - ~12000~15000 RMB
  - 消费级天花板，显存+算力都够
* - ⭐⭐⭐⭐
  - RTX 5090
  - 32 GB GDDR7
  - Blackwell
  - ~14000+ RMB
  - 最大显存消费卡
* - ⭐⭐⭐⭐
  - RTX 3080 20GB
  - 20 GB GDDR6X
  - Ampere
  - ~2000~3000 RMB
  - 20GB显存，实惠之选
* - ⭐⭐⭐
  - RTX 2080 Ti 22GB
  - 22 GB GDDR6
  - Turing
  - ~1500~2500 RMB
  - 改版22GB显存，魔改卡有风险
* - ⭐⭐⭐
  - RTX 4060 Ti 16GB
  - 16 GB GDDR6
  - Ada
  - ~2500~3000 RMB
  - 新卡省心，16GB够跑多数模型
* - ⭐⭐
  - RTX 3060 12GB
  - 12 GB GDDR6
  - Ampere
  - ~1000~1500 RMB
  - 入门首选，12GB够跑小型LLM
```

```{admonition} 显存决定你能跑什么
:class: important

这是最常被忽略的选卡标准。一个粗略参考：

- **6~8 GB**：可以跑BERT、ResNet、小CNN，LLM最多跑1~3B参数（量化后）
- **12~16 GB**：可以跑7B模型（4-bit量化QLoRA微调）
- **24 GB**：可以跑7B模型全参数微调，或13B模型QLoRA
- **32~48 GB**：可以跑13B全参数微调，或70B模型QLoRA
- **80 GB+**：可以跑70B模型全参数微调

**显存不够用 $\neq$ 完全不能做**。梯度累积、混合精度、LoRA/QLoRA、模型分片（model parallelism）都是小显存跑大模型的手段。但门槛是存在的——一张3090无论如何跑不了70B的全参数微调。
```

如果你在做购买决策，记住一个简单的公式：

> - **二手企业级（P100/V100）**  ≈ 极致性价比入门方案，有显存有算力但缺生态
> - **二手RTX 3090**              ≈ 个人用户的甜点卡，24GB显存+Ampere架构覆盖90%场景
> - **新的RTX 40/50系**           ≈ 省心省电，享受最新特性，代价是贵
> - **数据中心卡（A100/H100）**   ≈ 团队/生产环境的选择，个人用户租云实例更划算

### 一张显卡的"黄金年代"

从驱动支持和框架兼容性两个角度看，架构的生命周期大致分三个阶段：

- **淘汰期**（Maxwell/Pascal/Volta）：R580是最后支持的驱动分支，新版PyTorch也不再编译这些架构。还能用，但新特性与新卡无缘
- **成熟期**（Turing/Ampere）：驱动、框架、容器生态支持最完整，是当前最稳妥的选择
- **主力期**（Ada/Hopper/Blackwell）：享受所有新特性，但部分框架适配可能滞后，价格也最高

---

驱动装好了，CUDA配好了，接下来你需要知道怎么远程访问这台服务器。下一节{doc}`remote-access`。
