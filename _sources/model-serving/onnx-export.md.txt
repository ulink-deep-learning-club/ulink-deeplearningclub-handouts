(onnx-export)=
# ONNX：模型的中立语言

{ref}`model-serving-intro`中我们讨论了模型服务的四个核心挑战，第一个就是格式兼容性。PyTorch训练出的模型文件（`.pt`或`.pth`）本质上是一个包含模型结构和参数的Python pickle文件，它天然地和PyTorch运行时绑定在一起。如果要在一个没有PyTorch的环境（或者用不同框架）中加载它，就很困难。

ONNX（Open Neural Network Exchange，开放神经网络交换格式）正是为了解决这个问题而诞生的。你可以把它理解为深度学习的"通用语言"——无论训练时用的是PyTorch、TensorFlow还是其他框架，都可以导出为ONNX格式，然后在任何支持ONNX Runtime的环境中运行。

## 为什么需要ONNX？

ONNX 由微软和 Facebook 于 2017 年联合发起，初衷是打破框架壁垒——让研究员在 PyTorch 里训练，工程师用 ONNX Runtime 部署，互不依赖。2019 年 ONNX 被移交给 Linux Foundation 旗下的 LF AI & Data 基金会，成为一个真正中立的开放标准。

直觉上你可能觉得"多此一举"——既然已经在PyTorch里训练好了，为什么不直接在PyTorch里做推理？原因有几个。

第一是**性能优化**。ONNX Runtime会对计算图做一系列优化：算子融合（把多个连续的数学操作合并成一个）、常量折叠（把可以预先计算的表达式提前算好）、内存规划（复用内存减少分配开销）。这些优化在PyTorch的推理模式下不一定能自动启用。以 ResNet-50 为例，ONNX Runtime 的推理延迟通常比 PyTorch eager 模式快 20-40%。

第二是**部署自由度**。ONNX Runtime有C++、Python、C#、Java、JavaScript等多种语言的API，还可以运行在移动端和浏览器上（ONNX Runtime Mobile和ONNX Runtime Web）。这意味着你可以把一个ONNX模型同时部署到服务器、手机、网页三个平台，而训练它的框架是什么并不重要。

第三是**硬件加速的灵活性**。ONNX Runtime支持通过"执行提供者"（Execution Provider）机制切换不同的硬件后端——CPU、CUDA、TensorRT、CoreML、ROCm、WebGPU。同一个ONNX模型，在NVIDIA GPU上用TensorRT加速，在Apple Silicon上用CoreML加速，不需要修改模型本身。这对于{doc}`model-optimization`中讨论的量化和剪枝后的模型尤其重要——优化后的ONNX模型可以一键切换硬件后端以获得最佳加速。

## ONNX 计算图：一个 `.onnx` 文件里有什么

ONNX 文件的核心是一个**计算图**（Computation Graph），用 Google 的 Protocol Buffers（Protobuf）格式序列化。用一句话描述：**ONNX 图是一个有向无环图（DAG），节点是算子，边是张量，数据和计算逻辑分离存储。**

```{list-table} ONNX 计算图的核心组成
:header-rows: 1

* - **组件**
  - **Protobuf 字段**
  - **作用**
  - **举例（LeNet）**
* - 图（Graph）
  - `graph`
  - 顶层容器，包含所有节点和元信息
  - 整个 LeNet 前向计算
* - 节点（Node）
  - `graph.node`
  - 一个算子操作，定义计算逻辑
  - `Conv`, `Relu`, `MaxPool`, `Gemm`
* - 输入/输出（Value Info）
  - `graph.input` / `graph.output`
  - 图的入口和出口张量，含形状与数据类型
  - `input: float32[1,1,28,28]`, `output: float32[1,10]`
* - 初始化器（Initializer）
  - `graph.initializer`
  - 存储训练好的权重和偏置值
  - conv1.weight, conv1.bias, fc2.weight...
* - 算子集（OpSet）
  - `opset_import`
  - 声明图中使用的算子版本
  - `ai.onnx v18`
```

**操作直觉**：把 ONNX 图想象成一个工厂流水线——原材料（input）从入口流入，经过一系列工位（node，每个工位做一件事，比如卷积、激活、池化），最终成品（output）从出口流出。每个工位有自己固定的工具（initializer，即训练好的权重）。工厂的"操作规程"版本就是算子集（opset）。

每个算子节点有三个关键属性：

| 属性 | 含义 | LeNet 中 Conv 的例子 |
|------|------|---------------------|
| `op_type` | 算子类型 | `"Conv"` |
| `input` | 输入张量名列表 | `["input", "conv1.weight", "conv1.bias"]` |
| `output` | 输出张量名列表 | `["conv1_output"]` |
| `attribute` | 算子超参数 | `kernel_shape=[3,3]`, `strides=[1,1]`, `pads=[1,1,1,1]` |

这种"结构即文档"的设计意味着你用 Netron 打开一个 `.onnx` 文件，就能完整看到网络架构、每层的参数形状、数据流向——不需要原始的训练代码。这也是 ONNX 成为跨框架"通用语言"的底层原因。

### 算子集：版本的学问

`opset_version` 指定了 ONNX 算子集的版本。每个算子集定义了一组算子及其精确的行为规范。版本越高，支持的算子越新，表达能力越强——但可能不被旧版 ONNX Runtime 支持。

```{list-table} 常用算子集版本演变
:header-rows: 1

* - **版本**
  - **年份**
  - **代表性新增**
* - opset 11
  - 2019
  - 动态切片（Dynamic Slice）、Round
* - opset 13
  - 2020
  - Softmax 支持负轴、Sigmoid 改为 13 版本
* - opset 17
  - 2022
  - LayerNorm 成为标准算子
* - opset 18
  - 2023
  - GroupNorm、更完整的 Reduce 算子族
* - opset 21
  - 2024
  - 支持 INT4/UINT4 量化数据类型
```

{ref}`model-optimization`中讨论的 INT8/FP8 量化依赖较高版本的算子集（opset 18+），因为这些版本才定义了低精度数据类型。导出一个要量化的模型时，建议使用 `opset_version=18` 或更高。

## PyTorch到ONNX的导出

PyTorch从1.x版本开始内置了ONNX导出功能。从PyTorch 2.x开始，推荐的导出方式是基于 `torch.export` 引擎的新导出器（`dynamo=True` 模式，2.5+ 已是默认行为），它利用 Torch FX 和 `torch.export` 进行图捕获，生成的ONNX图更加精确。

导出一个模型的基本步骤如下：

```python
import torch
import torch.onnx

# 加载训练好的模型，设为评估模式
model = LeNetMNIST()
model.load_state_dict(torch.load("lenet_mnist.pth"))
model.eval()

# 创建一个示例输入
# 关键：输入的形状、数据类型必须和模型期望的一致
# LeNet期望输入：[batch, channel, height, width] = [1, 1, 28, 28]
dummy_input = torch.randn(1, 1, 28, 28)

# 导出为ONNX
torch.onnx.export(
    model,                          # 要导出的模型
    dummy_input,                    # 示例输入（用于追踪计算图）
    "lenet_mnist.onnx",             # 输出文件路径
    input_names=["input"],          # 输入层的名字
    output_names=["output"],        # 输出层的名字
    dynamic_axes={                  # 动态轴（可选）
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=18,              # ONNX算子集版本
)
```

这段代码有几个需要理解的要点。

`input_names` 和 `output_names` 为模型的输入输出张量起了名字。这个名字在服务端配置中非常重要——{doc}`deployment-practice` 中注册模型时，你需要知道模型输入层的具体名字，因为服务端通过名字匹配来传递数据。名字是自由的，但建议使用有意义的英文名（如 `"input"`、`"output"`），而不是默认的 `"onnx::Conv_0"` 这种内部名。

`dynamic_axes` 告诉导出器哪些维度是动态的（这里 batch 维度可以变化），这样导出的模型不仅限于 batch=1。如果不配置动态轴，ONNX Runtime 会拒绝接收与导出时尺寸不一致的输入。一个务实的做法是只将 batch 维度设为动态，其他维度固定——这样既支持 batch 推理，又避免了全动态形状带来的性能损失。

`opset_version` 指定了 ONNX 算子集的版本，越高版本支持的算子越新，但需要在目标部署环境中确认 ONNX Runtime 版本是否兼容。一般来说，ONNX Runtime 1.16+ 支持 opset 18，1.20+ 支持 opset 21。

## 导出后的验证

导出完成后，可以用 Netron（一个开源的神经网络模型可视化工具，在浏览器中运行）来查看 ONNX 模型的图结构——拖拽 `.onnx` 文件到 Netron 页面，就能看到完整的计算图、每层的参数形状、数据流向。

更重要的是，**一定要做数值验证**。导出成功只意味着 ONNX 图结构合法，不意味着计算正确：

```python
import onnx
import onnxruntime as ort
import numpy as np

# 验证ONNX模型结构
onnx_model = onnx.load("lenet_mnist.onnx")
onnx.checker.check_model(onnx_model)

# 用ONNX Runtime做推理，和PyTorch对比
ort_session = ort.InferenceSession("lenet_mnist.onnx")

# 准备一样的输入
test_input = np.random.randn(1, 1, 28, 28).astype(np.float32)

# PyTorch推理
with torch.no_grad():
    torch_output = model(torch.from_numpy(test_input)).numpy()

# ONNX Runtime推理
ort_input = {ort_session.get_inputs()[0].name: test_input}
ort_output = ort_session.run(None, ort_input)[0]

# 对比输出差异
diff = np.abs(torch_output - ort_output).max()
print(f"最大差异: {diff:.6f}")
# 如果导出正确，差异一般在 1e-5 ~ 1e-6 量级
```

数值差异过大通常意味着模型中含有 ONNX 不支持的算子，或者动态轴配置不正确。差异的来源有两种可能：一是浮点数运算顺序不同导致的微小误差（通常在 1e-5 量级），这可以接受；二是算子实现差异导致的系统性偏差，需要排查模型中的自定义操作。

## 导出时的常见问题

ONNX导出虽然看起来简单，但实际中会遇到一些容易踩坑的地方。

第一个问题是**动态控制流**。如果你的模型中有依赖于数据的条件分支（比如根据输入的值走不同的计算路径），这些分支在ONNX中可能无法正确捕获。原因是 ONNX 导出时需要一个静态的计算图，而 Python 的条件分支在导出时只能展开其中一个分支。解决方案是使用 `torch.where` 等可以表达为静态图的操作来替代条件语句。在 PyTorch 2.x 中，dynamo 导出器可以自动处理部分简单的控制流，但复杂分支仍需要手动改造。

第二个问题是**自定义算子**。如果你的模型使用了PyTorch中没有等价ONNX算子的自定义操作（比如一些特殊的注意力机制变体），导出时会报错。这时你需要实现一个自定义的ONNX算子。这在{doc}`../cnn-expedition/ablation-study/experiment-design`中的实验场景中可能不会遇到，但如果你在研究前沿架构，就需要了解。

第三个问题是**动态形状的配置**。如果你导出的模型需要在生产环境中处理不同大小的输入（比如目标检测模型需要处理不同分辨率的图片），就需要正确配置 `dynamic_axes`。配置错误会导致 ONNX Runtime 拒绝接收非固定尺寸的输入。

```{admonition} 数值验证的重要性
:class: warning

导出的ONNX模型一定要做数值验证。我们见过太多"导出成功但推理结果不对"的情况——导出成功只意味着ONNX图结构合法，不意味着计算正确。养成好习惯：每次都跑一遍PyTorch和ONNX Runtime的输出对比。
```

## 预处理与后处理：张量之外的世界

导出的ONNX模型本质上是"原始张量到原始张量"的映射。你的LeNet模型期望输入的是 `[1, 1, 28, 28]` 的归一化张量，输出的是 `[1, 10]` 的 logits——但在生产环境中，用户的输入是一张 JPEG 图片，你可能期望的输出是"数字3，置信度0.98"这样的结构化结果。

这就是预处理和后处理要做的事。

**预处理**把用户原始输入（图片文件、JSON 数据等）转换成模型期望的张量格式。一个典型的图片预处理流水线包括：缩放（Resize）→ 灰度转换（Grayscale）→ 标准化（Normalize）→ 转张量（To Tensor）。

**后处理**把模型的原始输出（logits、特征图）转换成用户能理解的结果。一个典型的分类后处理流水线包括：Softmax（转概率）→ Argmax（取最大类）→ 标签映射（数字→人类可读标签）。

预处理和后处理的设计需要与模型的训练流程严格对应。{doc}`../cnn-expedition/practice-peak/dataset-preparation` 中数据集的预处理步骤必须在推理时精确复现——否则模型看到的输入分布与训练时不一致，准确率会骤降。这和 {doc}`../cnn-expedition/ablation-study/experiment-design` 中"控制变量"的思维相通：确保推理环境与训练环境的预处理一致。

有关预处理/后处理如何在部署工具中落地，参见下文 {ref}`ferrinx-onnx-config`。

(ferrinx-onnx-config)=
## Ferrinx 中的 ONNX 模型配置

Ferrinx 通过 TOML 格式的模型配置文件来声明预处理和后处理流水线。以 LeNet 为例，模型配置文件 `model.toml` 如下：

```toml
[meta]
name = "lenet-mnist"
version = "1.0"
description = "MNIST digit classification model"

[model]
file = "lenet_mnist.onnx"

# 标签映射文件
labels = "labels.json"

[[inputs]]
name = "input"          # ONNX模型输入名（与导出时的input_names一致）
alias = "image"         # 用户友好的别名
shape = [-1, 1, 28, 28] # [batch, channel, height, width]
dtype = "float32"

# 预处理流水线：将用户图片转换为模型输入张量
[[inputs.preprocess]]
type = "resize"
size = [28, 28]

[[inputs.preprocess]]
type = "grayscale"

[[inputs.preprocess]]
type = "normalize"
mean = [0.1307]
std = [0.3081]

[[inputs.preprocess]]
type = "to_tensor"
dtype = "float32"
scale = 255.0

[[outputs]]
name = "output"         # ONNX模型输出名（与导出时的output_names一致）
alias = "prediction"
shape = [-1, 10]
dtype = "float32"

# 后处理流水线：将原始logits转换为分类结果
[[outputs.postprocess]]
type = "softmax"

[[outputs.postprocess]]
type = "argmax"
keep_prob = true

[[outputs.postprocess]]
type = "map_labels"
```

这个配置文件的思路和{doc}`../pytorch-practice/best-practices`中讨论的"将配置和代码分离"是一致的：Ferrinx读取这个配置文件，自动构建预处理和后处理流水线，用户只需关注模型本身。

预处理操作按顺序执行：先将用户上传的图片缩放到28×28，转为灰度图，用MNIST的mean和std做标准化，最后归一化到`[0,1]`区间并转为float32张量。后处理则将模型输出的10维logits通过Softmax转为概率分布，取最高概率对应的类别索引，再通过 `labels.json` 映射为具体的数字标签。

```json
{
  "labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
  "description": "MNIST handwritten digits"
}
```

### 模型目录结构

一个完整的模型目录结构如下：

```
models/
├── lenet-mnist/
│   ├── model.toml       # 配置文件
│   ├── lenet.onnx       # ONNX 模型
│   └── labels.json      # 标签映射
```

这种"模型即目录"的组织方式让模型的管理变得直观——复制一个目录就等于复制了一个完整的模型服务单元。

### Tensor格式：统一的API数据契约

当用户通过 REST API 调用模型推理时，输入输出数据需要一个标准化的格式。Ferrinx使用显式的Tensor结构来描述所有推理数据：

```json
{
  "dtype": "float32",
  "shape": [1, 1, 28, 28],
  "data": "<base64-encoded-binary>"
}
```

这个设计有三个意图。第一，**显式的shape**让API调用者不需要猜测模型期望的输入尺寸，shape信息就在数据中。第二，**base64编码的二进制数据**比嵌套JSON数组紧凑得多——对于224×224×3的图像，二进制编码比JSON数组能减少约50%的传输体积。第三，**显式的dtype**避免了类型混淆——float32和int64在JSON中看起来一样，但Tensor结构明确声明了数据类型。

在Python客户端中，构造Tensor输入的代码很直观：

```python
import base64
import numpy as np
import requests

# 创建numpy数组
input_array = np.random.randn(1, 1, 28, 28).astype(np.float32)

# 转为Tensor格式
tensor = {
    "dtype": "float32",
    "shape": list(input_array.shape),
    "data": base64.b64encode(input_array.tobytes()).decode("utf-8")
}

# 发送推理请求
response = requests.post(
    "http://localhost:8080/api/v1/inference/sync",
    headers={"Authorization": "Bearer frx_sk_..."},
    json={
        "model_id": "lenet-mnist-uuid",
        "inputs": {"input": tensor}
    }
)

# 解析返回结果
result = response.json()
output_tensor = result["data"]["outputs"]["output"]
output_array = np.frombuffer(
    base64.b64decode(output_tensor["data"]),
    dtype=np.float32
).reshape(output_tensor["shape"])
```

## 从导出到服务

掌握了ONNX导出之后，我们有了一个可以在不同平台运行的模型文件。但导出后的模型体积可能很大、推理速度可能不够快——在真正部署之前，还需要做一步"瘦身"。下一节{doc}`model-optimization`将介绍量化和剪枝两种核心技术，让模型变得"又轻又快"。
