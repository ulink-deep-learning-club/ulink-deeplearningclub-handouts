(appendix-env-end)=
# 结语：现在你有一台真正能用的服务器了

这篇附录的目标很简单：**让你拿到一台GPU服务器后，能在一小时内让它开始跑深度学习代码**，而不是花一整天纠结驱动版本、SSH配置和Jupyter连不上。

如果你是按顺序读下来的，你已经走过了这样一条路：

```{mermaid}
graph LR
    A[拿到服务器] --> B[装Linux<br/>基础操作]
    B --> C[装NVIDIA驱动<br/>配CUDA]
    C --> D[远程访问<br/>连上去]
    D --> E[服务管理<br/>让它稳定跑]
    E --> F[可以开始炼丹了]
    
    style A fill:#f3e5f5
    style B fill:#e3f2fd
    style C fill:#e8f5e9
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#ffebee
```

这条路径上每一步都可能卡住新手：

- **Linux基础**卡在"不知道sudo是什么意思"、"rm -rf /删了系统"
- **NVIDIA驱动**卡在"Secure Boot阻止驱动加载"、"nvidia-smi报driver mismatch"
- **远程访问**卡在"服务器没公网IP"、"Jupyter不让我连"
- **服务管理**卡在"训练跑一半终端关了"、"服务器重启了怎么自动启动"

每篇内容都在解决这些具体的痛点，而不是抽象地介绍工具。

## 但这不是终点

这篇附录和主课程的关系很特别。主课程的知识是**线性的**——你从数学基础读到模型部署，一章接一章。但环境配置的知识是**放射状**的——你遇到什么问题，回来翻对应的那篇就行。

你不会"读完"这篇附录就一劳永逸。真正的使用方式是：

- 第一次配服务器时顺序读{doc}`linux-basics`和{doc}`nvidia-setup`
- 遇到远程连接问题时翻{doc}`remote-access`，找到tailscale或frp的配置抄过去
- 部署服务时再读{doc}`server-management`，用systemd把服务固定下来

```{admonition} 记住一件事
:class: important

环境配置的终极目标不是让你成为运维专家——而是让运维知识**不成为你做深度学习的障碍**。如果有一节内容你读完了觉得"暂时用不上"，那就对了，等你需要的时候它会在那里。
```

---

{doc}`../../index` 才是你的主战场。如果你是从这里开始的，现在可以回到{doc}`../../model-serving/index` 或者{doc}`../../pytorch-practice/index`选择你感兴趣的进阶方向了。训练脚本在等你，GPU在转，loss曲线在下降——这才是真正重要的事。
