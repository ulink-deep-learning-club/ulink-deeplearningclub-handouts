(deployment-practice)=
# 部署实践：用Ferrinx服务模型

{ref}`serving-architecture`中我们讨论了模型服务的架构设计——同步异步、简单分布式、模型路由、认证限流。现在我们把理论变成实践：用Ferrinx把{ref}`onnx-export`导出的LeNet模型部署成一个真正的API服务。

这一节的操作流程和{doc}`../pytorch-practice/using-framework`中框架的使用思路一致：先了解工具的基本用法，然后通过实际操作来掌握各项功能。不同的是，这里的工具不是用来训练模型，而是用来服务模型。

## 编译与启动

Ferrinx是一个Rust项目，在项目目录的`ferrinx-main/`下。首次使用需要编译：

```bash
# 编译所有二进制
cd ferrinx-main
cargo build --release

# 编译产物
# ./target/release/ferrinx-api   - API服务器
# ./target/release/ferrinx-worker - 推理Worker
# ./target/release/ferrinx       - CLI客户端
```

编译完成后，用最简单模式启动——不需要Redis，不需要PostgreSQL，一个二进制加一个SQLite文件就够了：

```bash
# 启动API服务器（SQLite自动创建）
./target/release/ferrinx-api
```

服务默认监听`127.0.0.1:8080`。访问`http://localhost:8080/api/v1/health`可以看到健康检查响应。

```{admonition} ONNX Runtime的链接方式
:class: note

Ferrinx默认使用静态链接（`download-binaries`特性），编译时会自动下载预编译的ONNX Runtime库。如果你的系统glibc版本较旧（Debian 13以下、Ubuntu 24.04以下），需要改用动态链接模式：

~~~bash
cargo build --release --features load-dynamic
~~~

同时需要自行安装ONNX Runtime动态库，并在配置文件中指定路径：

~~~toml
[onnx]
dynamic_lib_path = "/usr/local/lib/libonnxruntime.so"
~~~
```

## 配置文件

Ferrinx的配置文件采用TOML格式。在`config.example.toml`的基础上修改即可。最简配置只需要指定数据库后端为SQLite：

```toml
[server]
host = "0.0.0.0"
port = 8080
sync_inference_concurrency = 4
sync_inference_timeout = 30

[database]
backend = "sqlite"
url = "sqlite://./data/ferrinx.db"

[storage]
backend = "local"
path = "./models"

[onnx]
cache_size = 5
execution_provider = "CPU"
```

各配置项的含义和{doc}`../pytorch-practice/best-practices`中讨论的"配置管理"理念相通——将可变参数从代码中分离出来，让部署者可以根据环境灵活调整。例如，`sync_inference_concurrency`控制同时有多少个推理请求可以并发执行，在CPU核心多的机器上可以调大这个值。

## 初始化系统

系统首次启动后，需要执行Bootstrap操作——创建第一个管理员用户并生成API Key。Ferrinx有CLI和curl两种方式完成这个步骤。

CLI方式（推荐）：
```bash
./target/release/ferrinx bootstrap
```

这个命令会自动创建admin用户，生成API Key并保存到本地配置文件。第一次输出的密码是系统自动生成的安全随机密码——只会显示一次，请及时保存。

curl方式：
```bash
# Bootstrap
curl -X POST http://localhost:8080/api/v1/bootstrap \
  -H "Content-Type: application/json" \
  -d '{}'

# 设置API Key环境变量
export FERRINX_API_KEY="frx_sk_..."
```

## 模型注册

有了API Key，就可以注册模型了。Ferrinx支持两种注册方式：一种是模型文件在服务端本地，通过配置文件直接注册；另一种是模型文件在客户端，通过上传接口提交。

对于本地有模型文件的情况，使用`model register`命令，通过{ref}`onnx-export`中编写的`model.toml`配置文件来注册：

```bash
./target/release/ferrinx model register \
  --model-config ./models/lenet-mnist/model.toml
```

Ferrinx会读取配置文件中的元信息、模型路径、预处理/后处理流水线和标签映射，自动注册模型并验证其有效性。注册成功后返回模型的UUID，后续推理调用需要用到这个ID。

如果模型文件在另一台机器上，使用`upload`命令上传：

```bash
./target/release/ferrinx model upload ./lenet_mnist.onnx \
  --name lenet-mnist \
  --version 1.0
```

上传完成后，用`model list`查看已注册的模型列表：

```bash
./target/release/ferrinx model list
```

你应该能看到刚刚注册的LeNet模型，状态为valid，代表模型通过了验证。

```{admonition} 模型验证的两层检查
:class: note

Ferrinx在注册模型时会做两层验证。第一层检查ONNX文件的magic number（文件头是否合法），第二层尝试解析模型的输入输出元信息，提取层名、形状和数据类型。这些元信息会被保存，后续推理时用于自动匹配输入。可选的第三层验证会创建ONNX Session来确认模型可执行，在`model_validation.validate_session = true`时启用——这个选项会显著增加注册时间。
```

## 执行推理

模型注册完成后，下面来调用推理。最简单的验证方式是使用CLI的同步推理命令：

```bash
# 通过模型名和版本指定模型
./target/release/ferrinx infer sync \
  --name lenet-mnist --version 1.0 \
  --image ./path/to/digit.png
```

CLI会自动对图片做预处理（根据model.toml中的配置，缩放→灰度→归一化→转张量），发送推理请求，并解析返回结果：

```json
{
  "result": {
    "class_index": 3,
    "label": "3",
    "probability": 0.9898
  },
  "latency_ms": 10
}
```

如果要处理批量请求或集成到其他系统，可以直接调用REST API。Python客户端的写法在{doc}`onnx-export`中已经给出，关键步骤是：将输入数据编码为Tensor格式（dtype + shape + base64），构造HTTP请求，带上API Key认证头，发送到`/api/v1/inference/sync`端点。

对于需要较长时间处理的大模型，使用异步推理：

```bash
# 提交异步推理任务
./target/release/ferrinx infer async \
  --name lenet-mnist --version 1.0 \
  --image ./path/to/digit.png

# 查询任务状态
./target/release/ferrinx task status <task-id>
```

异步推理返回一个task_id，客户端可以轮询这个ID来获取结果。在{ref}`serving-architecture`中讨论过，异步推理依赖Redis——如果没有配置Redis，异步推理端点会返回503。

## 配置执行提供者

如果你的机器有GPU，可以通过配置执行提供者来加速推理。Ferrinx支持CPU、CUDA、CoreML、ROCm、WebGPU等多种后端，在配置文件中指定即可：

```toml
[onnx]
execution_provider = "CUDA"    # 可选: CPU, CUDA, CoreML, ROCm, WEBGPU
gpu_device_id = 0
```

不同的执行提供者需要不同的Feature编译：
- WebGPU：`cargo build --release --features webgpu`
- CUDA：`cargo build --release --features cuda`
- CoreML：`cargo build --release --features coreml`

```{admonition} GPU配置的注意事项
:class: warning

编译时启用的执行提供者Feature必须和配置文件中的`execution_provider`一致。如果在配置中设置了CUDA但编译时没有启用cuda feature，Ferrinx会在启动时报错。WebGPU是推荐的GPU加速选项——它通过Vulkan/DirectX/Metal适配各类GPU，不需要安装CUDA Toolkit。
```

## API Key管理

生产环境中，不同的客户端需要使用不同的API Key，而不是所有客户端共享bootstrap生成的管理员Key。创建新的API Key：

```bash
# 创建一个永久Key，带默认用户权限
./target/release/ferrinx api-keys create \
  --name "my-app-key"

# 创建一个30天后过期的Key
./target/release/ferrinx api-keys create \
  --name "trial-key" \
  --expires-days 30
```

创建时返回的Key明文只会显示一次，请立即保存。查看和管理已有的Key：

```bash
# 列出所有Key
./target/release/ferrinx api-keys list

# 撤销某个Key
./target/release/ferrinx api-keys revoke <key-id>
```

## 部署到生产环境的建议

从开发环境到生产环境，有几个关键的配置调整值得注意。

第一，**数据库切换**。开发时SQLite很方便，但生产环境面对多实例部署，应该切换到PostgreSQL。配置文件中修改`[database]`段即可，Ferrinx的Repository抽象层确保业务代码不需要修改。

第二，**日志配置**。开发时日志格式为text方便阅读，生产环境应该改为json格式，便于日志收集系统（如ELK）解析。同时配置日志文件轮转，避免磁盘被日志填满。

```toml
[logging]
level = "info"
format = "json"
file = "./logs/ferrinx.log"
max_file_size_mb = 100
max_files = 10
```

第三，**安全增强**。确保API使用HTTPS（在反向代理如Nginx层面配置TLS），设置合理的限流参数，定期轮换API Key。Bootstrap接口在生产环境中应该在完成初始化后禁用（Ferrinx在已有用户时会自动拒绝bootstrap请求）。

第四，**监控部署**。Ferrinx暴露了`/api/v1/metrics`端点和基本的健康检查接口，可以接入Prometheus等监控系统。Worker的健康状况可以通过Redis心跳监测，长时间无心跳的Worker说明需要人工介入。

```{admonition} 从开发到生产的检查清单
:class: important

- [ ] 将数据库从SQLite切换为PostgreSQL
- [ ] 配置HTTPS（反向代理层面）
- [ ] 设置合理的限流参数
- [ ] 创建独立的生产API Key，不要使用bootstrap Key
- [ ] 配置日志输出为JSON格式
- [ ] 启用模型预热（`[onnx].preload`）
- [ ] 部署监控告警
- [ ] 制定备份和恢复策略
```

## 从框架到Ferrinx的一体化流程

现在回顾整条链路：从PyTorch代码开始（{doc}`../pytorch-practice/index`），经过模型设计和训练（{doc}`../neural-network-basics/index`），用消融实验验证（{doc}`../cnn-ablation-study/index`），导出为ONNX格式（{doc}`onnx-export`），最后部署到Ferrinx服务（本节）——一条完整的"从研究到生产"的管线就这样建立起来了。

这种端到端的视角是理解深度学习工程化的关键。每一章学的技能不是孤立的，它们构成了一个完整的工具链：PyTorch让你能做实验，消融研究让你能验证设计，ONNX解决了格式壁垒，Ferrinx解决了部署问题。下一节{doc}`the-end`我们来回顾整条链路，并讨论未来可以深入的方向。

---

## 参与到社团项目的开发

Ferrinx 是UCS 深度学习社维护的开源项目，欢迎你贡献代码、报告问题或提出改进建议：

- [github.com/ulink-deep-learning-club/ferrinx](https://github.com/ulink-deep-learning-club/ferrinx)

**贡献的方式很简单**：发现问题 → 提 Issue → 讨论方案 → 提交 PR。即使只是修正一个文档错别字，也是对社区有意义的贡献。
