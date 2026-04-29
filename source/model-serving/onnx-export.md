(onnx-export)=
# ONNX：模型的中立语言

{ref}`model-serving-intro`中我们讨论了模型服务的三个核心挑战，第一个就是格式兼容性。PyTorch训练出的模型文件（`.pt`或`.pth`）本质上是一个包含模型结构和参数的Python pickle文件，它天然地和PyTorch运行时绑定在一起。如果要在一个没有PyTorch的环境（或者用不同框架）中加载它，就很困难。

ONNX（Open Neural Network Exchange）正是为了解决这个问题而诞生的。你可以把它理解为深度学习的"通用语言"——无论训练时用的是PyTorch、TensorFlow还是其他框架，都可以导出为ONNX格式，然后在任何支持ONNX Runtime的环境中运行。

## 为什么需要ONNX？

直觉上你可能觉得"多此一举"——既然已经在PyTorch里训练好了，为什么不直接在PyTorch里做推理？原因有几个。

第一是**性能优化**。ONNX Runtime会对计算图做一系列优化：算子融合（把多个连续的数学操作合并成一个）、常量折叠（把可以预先计算的表达式提前算好）、内存规划（复用内存减少分配开销）。这些优化在PyTorch的推理模式下不一定能自动启用。

第二是**部署自由度**。ONNX Runtime有C++、Python、C#、Java、JavaScript等多种语言的API，还可以运行在移动端和浏览器上（ONNX Runtime Mobile和ONNX Runtime Web）。这意味着你可以把一个ONNX模型同时部署到服务器、手机、网页三个平台，而训练它的框架是什么并不重要。

第三是**硬件加速的灵活性**。ONNX Runtime支持通过"执行提供者"（Execution Provider）机制切换不同的硬件后端——CPU、CUDA、TensorRT、CoreML、ROCm、WebGPU。同一个ONNX模型，在NVIDIA GPU上用TensorRT加速，在Apple Silicon上用CoreML加速，不需要修改模型本身。

## PyTorch到ONNX的导出

PyTorch从1.x版本开始内置了ONNX导出功能。从PyTorch 2.x开始，推荐的导出方式是基于`torch.export`引擎的新导出器（`dynamo=True`模式），它利用Torch FX和torch.export进行图捕获，生成的ONNX图更加精确。

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
    dummy_input,                    # 示例输入
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

这段代码有几个需要理解的要点。`input_names`和`output_names`为模型的输入输出张量起了名字，这个名字在后续的服务端配置中非常重要——{doc}`deployment-practice`中注册模型时，你需要知道模型输入层的具体名字。`dynamic_axes`告诉导出器哪些维度是动态的（这里batch维度可以变化），这样导出的模型就不仅限于batch=1。`opset_version`指定了ONNX算子集的版本，越高版本支持的算子越新，但兼容性可能降低。

导出完成后，可以用Netron（一个神经网络模型可视化工具）来查看ONNX模型的图结构，也可以用ONNX Runtime来验证导出的正确性。

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

这个验证步骤非常重要。数值差异过大通常意味着模型中含有ONNX不支持的算子，或者动态轴配置不正确。差异的来源有两种可能：一是浮点数运算顺序不同导致的微小误差（通常在1e-5量级），这可以接受；二是算子实现差异导致的系统性偏差，需要排查模型中的自定义操作。

## 预处理和后处理

导出的ONNX模型本质上是"原始张量到原始张量"的映射。你的LeNet模型期望输入的是`[1, 1, 28, 28]`的归一化张量，输出的是`[1, 10]`的logits——但在生产环境中，用户的输入是一张JPEG图片，你期望的输出可能是"数字3，置信度0.98"这样的结构化结果。

这就是预处理和后处理要做的事。预处理把用户原始输入（图片文件、JSON数据等）转换成模型期望的张量格式；后处理把模型的原始输出（logits、特征图）转换成用户能理解的结果。

Ferrinx通过模型配置文件来声明预处理和后处理流水线。以LeNet为例，模型配置文件`model.toml`如下：

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
name = "input"          # ONNX模型输入名
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
name = "output"         # ONNX模型输出名
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

预处理操作按顺序执行：先将用户上传的图片缩放到28×28，转为灰度图，用MNIST的mean和std做标准化，最后归一化到`[0,1]`区间并转为float32张量。后处理则将模型输出的10维logits通过Softmax转为概率分布，取最高概率对应的类别索引，再通过`labels.json`映射为具体的数字标签。

```json
{
  "labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
  "description": "MNIST handwritten digits"
}
```

一个完整的模型目录结构如下：

```
models/
├── lenet-mnist/
│   ├── model.toml       # 配置文件
│   ├── lenet.onnx       # ONNX 模型
│   └── labels.json      # 标签映射
```

这种"模型即目录"的组织方式让模型的管理变得直观——复制一个目录就等于复制了一个完整的模型服务单元。

## Tensor格式：统一的API数据契约

当用户通过REST API调用模型推理时，输入输出数据需要一个标准化的格式。Ferrinx使用显式的Tensor结构来描述所有推理数据：

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

## 导出时的常见问题

ONNX导出虽然看起来简单，但实际中会遇到一些容易踩坑的地方。

第一个问题是**动态控制流**。如果你的模型中有依赖于数据的条件分支（比如根据输入的值走不同的计算路径），这些分支在ONNX中可能无法正确捕获。原因是ONNX导出时需要一个静态的计算图，而Python的条件分支在导出时只能展开其中一个分支。解决方案是使用`torch.where`等可以表达为静态图的操作来替代条件语句。

第二个问题是**自定义算子**。如果你的模型使用了PyTorch中没有等价ONNX算子的自定义操作（比如一些特殊的注意力机制变体），导出时会报错。这时你需要实现一个自定义的ONNX算子。这在{doc}`../cnn-ablation-study/experiment-design`中的实验场景中可能不会遇到，但如果你在研究前沿架构，就需要了解。

第三个问题是**动态形状的配置**。如果你导出的模型需要在生产环境中处理不同大小的输入（比如目标检测模型需要处理不同分辨率的图片），就需要正确配置`dynamic_axes`。配置错误会导致ONNX Runtime拒绝接收非固定尺寸的输入。一个务实的做法是：在配置文件中声明输入的形状时，将batch维度设为-1（表示动态），其他维度固定——这样既支持batch推理，又避免了全动态形状带来的性能损失。

```{admonition} 数值验证的重要性
:class: warning

导出的ONNX模型一定要做数值验证。我们见过太多"导出成功但推理结果不对"的情况——导出成功只意味着ONNX图结构合法，不意味着计算正确。养成好习惯：每次都跑一遍PyTorch和ONNX Runtime的输出对比。
```

## 从导出到服务

掌握了ONNX导出之后，我们有了一个可以在不同平台运行的模型文件。但文件本身不能对外提供服务，还需要一个运行环境来接收HTTP请求、加载模型、执行推理并返回结果。下一节{doc}`serving-architecture`将介绍模型服务的整体架构设计——包括同步和异步两种推理模式、分布式部署方案，以及模型路由等核心机制。
