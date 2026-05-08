(serving-architecture)=
# 服务架构：从单机到分布式

{ref}`onnx-export`中我们把PyTorch模型导出成了ONNX格式，现在有了一个可以在不同平台上运行的标准模型文件。但有了模型文件不等于有了服务，还需要一个架构来接收外部请求、加载和执行模型、返回推理结果。

这一节我们用Ferrinx的架构设计作为案例，来理解模型服务中的核心架构决策。Ferrinx的设计围绕两个维度展开：**推理模式**（同步 vs 异步）和**部署拓扑**（简单模式 vs 分布式模式）。

## 两个核心维度

### 推理模式：同步 vs 异步

同步推理和异步推理的核心区别在于：调用者是否需要原地等待结果。

**同步推理**：客户端发送请求后，HTTP连接保持打开，服务器处理完推理后立即返回结果。就像打电话——你问一句，对方当场回答。对于轻量模型（几十毫秒内能完成的推理），这是最常见的选择。Ferrinx的同步推理在API进程内执行，通过Semaphore控制并发数，用`spawn_blocking`把CPU密集的推理任务移出tokio运行时，避免阻塞异步事件循环。

**异步推理**：客户端提交任务后立即获得一个task_id，稍后再通过这个ID查询结果。就像发邮件——你发出去，对方处理完再通知你。Ferrinx的异步推理通过Redis Streams实现任务队列，Worker进程从队列中消费并执行推理。适合大模型（推理时间以秒计）或者推理负载波动大的场景——任务队列起到缓冲作用，削峰填谷。

```{list-table} 同步推理 vs 异步推理
:header-rows: 1

* - **特性**
  - **同步推理**
  - **异步推理**
* - 调用方式
  - 阻塞等待
  - 提交→轮询
* - 执行位置
  - API进程内
  - Worker进程
* - 模型路由
  - 不适用（本地加载）
  - 智能路由到最优Worker
* - Redis依赖
  - 不需要
  - 必须
* - 典型延迟
  - < 100ms
  - 可变（秒级）
* - 适用场景
  - 轻量模型、实时响应
  - 大模型、批处理、高负载
```

### 部署拓扑：简单模式 vs 分布式模式

简单模式就是"一个二进制跑起来就能用"——一个API进程包含所有功能（HTTP服务、推理引擎、数据库），不需要任何外部依赖。适合开发测试、小规模部署，以及刚接触模型服务时快速上手。

分布式模式在简单模式的基础上添加了Redis。Redis承担三个角色：任务队列（Redis Streams，异步推理的核心）、API Key缓存（加速认证验证）、推理结果缓存（减少重复计算）。Worker进程独立部署，从Redis Streams消费任务执行推理，并将结果写回。当需要扩展推理能力时，多启动几个Worker即可。

```{mermaid}
graph TB
    subgraph "简单模式（无Redis）"
        A1[Client] --> B1[API Server<br/>axum]
        B1 --> C1[Sync Inference<br/>In-Process]
        B1 --> D1[SQLite DB]
        D1 --> E1[Models/Local Storage]
    end
    
    subgraph "分布式模式（有Redis）"
        A2[Client] --> B2[API Server]
        B2 --> C2[Sync Inference<br/>In-Process]
        B2 --> F2[Redis]
        F2 --> G2[Worker Pool]
        B2 --> D2[PostgreSQL DB]
        D2 --> E2[Models/Local Storage]
        G2 --> E2
    end
```

## 架构的组件级分析

Ferrinx的架构可以分为四个层次。

### 接入层（API Server）

API Server基于axum框架，提供RESTful接口。它包含了三个中间件，按从外到内的顺序依次生效：日志中间件记录所有请求信息（包括未认证的请求）；限流中间件控制请求速率，在认证之前拦截，避免认证过程被暴力请求压垮；认证中间件验证API Key，从Redis缓存或数据库查找记录。

```{list-table} API端点一览
:header-rows: 1

* - **端点**
  - **方法**
  - **说明**
* - `/api/v1/health`
  - GET
  - 健康检查
* - `/api/v1/bootstrap`
  - POST
  - 系统初始化（仅首次可用）
* - `/api/v1/auth/login`
  - POST
  - 用户登录（返回临时Key）
* - `/api/v1/models/upload`
  - POST
  - 上传模型文件
* - `/api/v1/models/register`
  - POST
  - 注册已有模型
* - `/api/v1/inference/sync`
  - POST
  - 同步推理
* - `/api/v1/inference`
  - POST
  - 异步推理（提交任务）
* - `/api/v1/inference/{id}`
  - GET
  - 查询任务结果
* - `/api/v1/api-keys`
  - POST
  - 创建API Key
```

### 推理引擎层（Core）

推理引擎是Ferrinx的核心组件。它管理着ONNX Runtime的Session实例，通过LRU缓存减少模型重复加载的开销。当API Server收到同步推理请求时，引擎首先从缓存中查找Session，如果缓存未命中则从磁盘加载模型文件、创建Session并写入缓存。

推理执行通过`spawn_blocking`放到独立的线程池中，避免占用tokio的异步线程。同步推理的并发数通过Semaphore控制——每接到一个推理请求，先尝试获取一个信号量许可，获取成功则执行推理，获取失败则立即返回并发超限的错误。这个设计防止了短时间大量推理请求涌入时耗尽系统内存。

### Worker层

Worker进程独立于API Server部署，负责消费异步推理任务。每个Worker启动时会扫描本地磁盘，盘点自己拥有哪些模型的文件，然后将模型状态（cached / available）上报到Redis。API Server在处理异步推理请求时，会查询Redis获取模型到Worker的映射，把任务路由到最优的Worker——优先选择模型已缓存的Worker，其次是模型文件存在的Worker，如果没有任何Worker拥有该模型则返回错误。

Worker的故障恢复通过Redis Streams的消费组机制实现。Worker在处理一个任务时如果宕机，Stream中的任务不会被确认（XACK），其他Worker可以在一段时间后通过XCLAIM认领这些超时未完成的任务。这个机制保证了单个Worker宕机不会导致任务丢失。

### 存储与缓存层

Ferrinx使用Repository模式抽象数据库操作。SQLite和PostgreSQL两种后端通过trait统一接口，业务代码不依赖具体的数据库实现。模型文件存储在本地文件系统（S3接口已预留但尚未实现）。

Redis在分布式模式下承担三个缓存角色：API Key缓存（加速验证，避免每次请求都查询数据库）、推理结果缓存（异步推理完成后，结果写入Redis，客户端轮询时直接从缓存读取）、Loader状态缓存（Worker上报模型状态）。如果Redis不可用，系统会优雅降级：API Key验证回退到数据库查询，同步推理不受影响，异步推理则不可用（因为没有任务队列）。

## 模型路由的设计

在分布式模式中，路由机制是理解系统如何扩展的关键。当一个异步推理请求到达API Server时，路由逻辑是这样工作的：

1. API Server从数据库中查找模型信息
2. 查询Redis中该模型到Worker的映射（`ferrinx:models:{model_id}:workers`是一个有序集合，分数越低优先级越高，cached为0，available为1）
3. 如果找到已缓存的Worker，将任务推送到该Worker的专属Stream
4. 否则如果有available的Worker，推送到其专属Stream
5. 如果没有任何Worker拥有该模型，返回"无可用Worker"错误

Worker定期（每10秒）刷新自己的模型状态到Redis，同时发送心跳（TTL 60秒）。如果Worker宕机，Redis中的心跳和模型状态自然过期，路由会自动将新任务分配给其他Worker。

## 认证与安全

模型服务必须面对的一个现实是：API一旦暴露在公网上，就面临被滥用的风险。Ferrinx通过三个层次来保障安全。

第一层是**API Key认证**。所有非公共端点（除了health、bootstrap和login）都需要在HTTP Header中携带`Authorization: Bearer frx_sk_...`。API Key本身不存储明文，数据库中只保存SHA-256哈希值。即使数据库被攻破，攻击者也无法逆向出原始的API Key。

第二层是**基于角色的权限控制**。每个API Key关联一组权限，定义了该Key能否执行模型管理、推理调用、管理员操作等。例如，一个只用于推理的API Key可以被配置为只能调用`/api/v1/inference/sync`，无法删除模型或管理用户。如果这个Key泄露，攻击者的破坏范围被限制在推理调用上。

第三层是**限流保护**。Ferrinx实现了滑动窗口和令牌桶两种限流算法，可以对不同端点设置不同的速率限制。例如，同步推理端点限制为每分钟30次，异步推理端点可能放宽到每分钟100次。限流基于API Key和IP的组合，防止单个用户通过多IP绕过限制。

```{admonition} 安全与便利的平衡
:class: tip

模型服务的安全设计需要权衡便利性。API Key比用户名密码更适合机器对机器的通信，但管理API Key本身也需要规范。在生产环境中，建议配置API Key的有效期、定期轮换，并将Key存储在环境变量或密钥管理服务中，而不是硬编码在代码里。
```

## 架构设计的原则

Ferrinx的架构设计体现了几条值得借鉴的原则。第一，**简单性优先**——无Redis的简单模式只用一条命令就能启动，降低了入门门槛。第二，**渐进式复杂**——需要分布式能力时，只需启动Redis和Worker，不需要修改代码或配置结构。第三，**优雅降级**——每个外部依赖（Redis、数据库）都有回退策略，系统在部分组件不可用时仍能提供核心功能。

```{admonition} 架构选择的建议
:class: important

对于个人项目或小团队：
1. 从简单模式开始，最快获得可用的服务
2. 当需要异步推理时，添加Redis
3. 当单个Worker不够时，横向扩展Worker数量

不要一开始就搭建完整的分布式系统——你很可能不需要它。
```

下一节{doc}`deployment-practice`将实际操作Ferrinx的部署流程，包括编译、配置、模型注册和推理调用，把这些架构概念落到实处。
