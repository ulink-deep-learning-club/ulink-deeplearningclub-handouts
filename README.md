# Deep Learning Club Learning Materials

深度学习社教学资料库——面向高中生，从理论到实践的深度学习课程。

## 📚 在线文档

完整课程文档地址：**https://ulink-deep-learning-club.github.io/ulink-deeplearningclub-handouts/**

## 📁 项目结构

```
lecture-material/
├── source/                          # 课程文档源码（Markdown）
│   ├── index.md                     # 课程主页
│   ├── preface.md                   # 写在前面：关于深度学习
│   ├── math-fundamentals/           # 数学基础
│   │   ├── computational-graph.md   # 计算图
│   │   ├── back-propagation.md      # 反向传播
│   │   ├── gradient-descent.md      # 梯度下降
│   │   ├── activation-functions.md  # 激活函数
│   │   └── loss-functions.md        # 损失函数
│   ├── neural-network-basics/       # 神经网络基础
│   │   ├── fc-layer-basics.md       # 全连接层
│   │   ├── cnn-basics.md            # 卷积神经网络
│   │   ├── le-net.md                # LeNet-5架构
│   │   ├── neural-training-basics.md # 训练基础
│   │   ├── exp-cmp.md               # 全连接 vs CNN实验对比
│   │   └── scaling-law.md           # 缩放定律
│   ├── pytorch-practice/            # PyTorch实践
│   │   ├── tensor-ops.md            # 张量操作
│   │   ├── neural-network-module.md # 神经网络模块
│   │   ├── auto-grad.md             # 自动微分
│   │   ├── optimiser.md             # 优化器
│   │   ├── train-workflow.md        # 训练流程
│   │   └── best-practices.md        # 工程最佳实践
│   ├── cnn-ablation-study/          # CNN消融研究
│   │   ├── experiment-design.md     # 实验设计
│   │   └── implementation.md        # 代码实现
│   ├── transfer-learning/           # 迁移学习
│   │   ├── part1-intro.md           # 迁移学习简介
│   │   ├── part2-taxonomy.md        # 分类体系
│   │   ├── part3-model-transfer.md  # 模型迁移技术
│   │   └── part4-practical-guide.md # 实践指南
│   ├── attention-mechanisms/        # 注意力机制
│   │   ├── se-net.md                # SE-Net通道注意力
│   │   ├── cbam.md                  # CBAM注意力
│   │   └── practice.md              # 实践指南
│   ├── unet-image-segmentation/     # U-Net图像分割
│   │   ├── u-net.md                 # U-Net架构
│   │   ├── core-impl.md             # 核心实现
│   │   └── practice.md              # 实践指南
│   ├── model-serving/               # 模型部署
│   │   ├── onnx-export.md           # ONNX模型导出
│   │   ├── serving-architecture.md  # 服务架构
│   │   └── deployment-practice.md   # 部署实践
│   ├── appendix/                    # 附录
│   │   └── environment-setup/       # 环境配置（按需查阅）
│   │       ├── index.md             # 附录概览
│   │       ├── linux-basics.md      # Linux基础
│   │       ├── nvidia-setup.md      # NVIDIA驱动与CUDA
│   │       ├── remote-access.md     # 远程访问与内网穿透
│   │       ├── server-management.md # 服务管理与进程守护
│   │       └── the-end.md
│   └── postscript.md                # 写在后面
├── mnist-helloworld-main/           # MNIST训练框架（Python）
├── ferrinx-main/                    # 模型推理服务（Rust）
├── legacy-doc/                      # 历史LaTeX文档（已归档）
└── _static/                         # 静态资源（图片、样式）
```

## 🎯 课程定位

**目标读者**：具备微积分基础的高中生AI社团成员

**前置要求**：
- 数学：微积分（导数、链式法则）、线性代数（矩阵乘法）、基础概率论
- 编程：Python基础（函数、类、NumPy数组操作）
- 深度学习：零基础，课程从零开始

**配套教材**：《深度学习入门》（斋藤康毅）——建议作为预习材料

## 🗺 学习路径

### 核心路径（必修）
1. **数学基础** → 计算图、反向传播、梯度下降
2. **神经网络基础** → 全连接层、CNN、LeNet-5架构
3. **PyTorch实践** → 张量操作、自动微分、完整训练流程

### 进阶路径（选修，任选方向）
- **CNN消融研究**（2-4周）——培养科学实验思维，理解各组件贡献
- **迁移学习**（2-3小时）——掌握小数据集训练大模型的核心技术
- **注意力机制**（2-3小时）——SE-Net、CBAM等模型改进技术
- **U-Net分割**（3-4小时）——图像分割实战，编码器-解码器架构
- **模型部署**（2-3小时）——ONNX导出、API服务化

**学习建议**：理论理解 40% + 代码实践 60%

## 🛠 本地构建

```bash
# 创建虚拟环境
uv venv
source .venv/bin/activate

# 安装依赖
uv sync

# 构建HTML文档
make build

# 启动本地服务器预览
cd build/html && python3 -m http.server 1200
# 访问 http://localhost:1200
```

## 📝 内容特点

- **理论到实践**：每个概念都配有PyTorch代码实现
- **详细注释**：代码中每行都有注释，包含维度变化和参数计算
- **可视化丰富**：TikZ图表、Mermaid流程图辅助理解
- **Cross-Reference**：章节间紧密关联，形成知识网络
- **历史背景**：包含经典论文引用（LeCun、He等）

## 🤝 参与贡献

欢迎提交PR改进课程内容：

1. Fork本仓库
2. 在对应Markdown文件中修改
3. 提交Pull Request并详细说明改动内容

## 📖 技术说明

- **构建工具**：Sphinx + MyST Markdown
- **数学渲染**：MathJax（LaTeX语法）
- **图表支持**：TikZ、Mermaid
- **代码高亮**：Pygments

---

**最后更新**：2026-04-29

> ⚠️ 本课程包含大量梯度下降、反向传播和玄学调参。副作用可能包括：对GPU产生依赖、对过拟合产生恐惧、以及半夜醒来突然想到"是不是学习率太大"。
