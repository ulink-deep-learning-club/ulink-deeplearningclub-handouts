(remote-access)=
# 远程访问：从任何地方连上你的GPU服务器

GPU服务器很少在你手边——可能在机房，可能在云上，可能在朋友家阁楼里吃灰。你需要的是**在任何地方都能连上去跑训练、看loss曲线、查日志**的能力。

这一节不讲理论，只讲实操。从最简单的SSH隧道到全家桶式的组网方案，按需选读就行。

## SSH：不止是登上去敲命令

{doc}`linux-basics`教了你密钥配置和基本连接。但深度学习场景下，SSH真正的大招是**隧道转发**。

(ssh-tunnel)=
### 把远程服务"拉"到本地来

{ref}`ssh-setup`已经帮你配好了密钥和基本连接，但SSH的真正实力是**端口转发**。最常见的场景是：服务器上跑着Jupyter Notebook，你想在本地浏览器打开它。SSH隧道就是干这个的：

```bash
# 把服务器的8888端口映射到本地的8888端口
ssh -L 8888:localhost:8888 user@your-server

# 然后打开 http://localhost:8888 —— 实际上访问的是服务器上的Jupyter
```

三种隧道各有各的用处：

| 类型 | 参数 | 什么时候用 |
|------|------|-----------|
| 本地转发 | `-L 本地端口:目标:目标端口` | Jupyter、TensorBoard、Gradio——远程服务拉到本地看 |
| 远程转发 | `-R 远程端口:本地:本地端口` | 让公网服务器帮你转发流量到内网机器 |
| 动态转发 | `-D 本地端口` | 整台浏览器的流量全走服务器网络 |

把这些固定配置写进 `~/.ssh/config`，以后就不用每次敲长长一串了：

```
# ~/.ssh/config
Host gpu-server
    HostName your-server-ip
    User anson
    # 自动映射Jupyter和TensorBoard端口
    LocalForward 8888 localhost:8888
    LocalForward 6006 localhost:6006
    # 保持心跳，防止长时间训练后断开
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

```{admonition} 一个要命的细节
:class: warning

通过SSH隧道访问Jupyter时，Jupyter必须只监听 `localhost`（`--ip=127.0.0.1`），**绝对不能**监听 `0.0.0.0`。否则你的Notebook就直接暴露在公网上了，任何人都能访问。这个原则同样适用于TensorBoard、Gradio、Streamlit——只要是通过SSH隧道访问的服务，绑定到127.0.0.1就对了。
```

### VSCode Remote SSH

如果你用VSCode写代码，Remote SSH插件比Jupyter的体验好一个数量级——本地写代码，远程执行，文件自动同步。装了插件之后 `Ctrl+Shift+P` → `Remote-SSH: Connect to Host` 选服务器就行。它自动读 `~/.ssh/config` 里的配置，设好的隧道和别名都能用。

效果就是：代码在GPU服务器上跑，但编辑体验和在本地完全一样。断网了重连一下，编辑器状态全在。

### 过跳板机（ProxyJump）

很多实验室和云环境的GPU服务器躲在私有子网里，不能直接访问。你得先登上一台**跳板机**（堡垒机），再通过它跳到GPU服务器：

```bash
# 原始方案：两步走
ssh user@jump-server
# 在跳板机上再执行
ssh user@gpu-server

# 优雅方案：ProxyJump一步到位
ssh -J user@jump-server user@gpu-server

# 如果跳板机端口不是默认的22
ssh -J user@jump-server:2222 user@gpu-server
```

写进 `~/.ssh/config` 就更清爽了：

```
Host jump-server
    HostName jump.company.com
    User anson

Host gpu-server
    HostName 10.0.1.100
    User anson
    ProxyJump jump-server
    # 还可以同时配端口转发，穿透两层
    LocalForward 8888 localhost:8888
```

配置好后一句 `ssh gpu-server` 就自动经过跳板机直连GPU服务器，隧道也一并建立。如果不方便装OpenSSH 7.3以上版本（ProxyJump需要），也可以用老的 `ProxyCommand` 模式：

```
Host gpu-server
    HostName 10.0.1.100
    User anson
    ProxyCommand ssh jump-server -W %h:%p
```

(mosh-basics)=
### mosh：容忍你切WiFi、掉线、高延迟

SSH有一个天生的毛病：**网络一不稳定就断**。你在咖啡厅连服务器看loss曲线，站起来换了个位置，WiFi切了一下——SSH卡死了，只能重连。

mosh（Mobile Shell）就是来解决这个问题的。它在SSH的基础上加了一层UDP会话，能容忍IP地址变化、网络延迟波动、甚至电脑合盖再打开：

```bash
# 安装mosh（服务端和客户端都需要）
sudo apt install mosh

# 连接（和ssh用法基本一样）
mosh user@your-server

# 如果SSH端口不是22
mosh --ssh="ssh -p 2222" user@your-server
```

连接上之后试试：断开WiFi换成手机热点——mosh不会断，敲几个回车就恢复了。对于"在高铁上看loss曲线"这种场景来说，mosh比tmux续命更直接——tmux保证**会话**不丢，mosh保证**连接**不丢，两者搭配最佳。

```{admonition} mosh的限制
:class: note

mosh不支持SSH隧道转发（-L/-R/-D参数）。所以mosh适合日常查状态、敲命令、看日志。如果你需要端口转发（Jupyter、TensorBoard），还是得用SSH隧道。实际工作流通常是：mosh上去启动训练，SSH隧道开Jupyter看结果。
```

(tailscale-basics)=
## tailscale：零配置组网，像在同一局域网

tailscale是一个建立在WireGuard之上的组网工具。它的核心价值一句话就能说清楚：**让分布在世界各地的机器像在同一个局域网里一样通信**。

### 什么时候你需要它

| 场景 | 没有tailscale | 有tailscale |
|------|-------------|-------------|
| 服务器没有公网IP | 连都连不上 | tailscale IP直连 |
| 想从手机上看训练进度 | 折腾端口转发一整天 | 手机装个app就行 |
| 几个同学共用一台服务器 | 每个人都要配一遍 | 每人一个tailscale账号自动搞定 |
| 不想把服务暴露到公网 | 得写iptables防扫描 | 默认就在加密网络里 |

### 真·三步搞定

```bash
# 第1步：所有设备都装
curl -fsSL https://tailscale.com/install.sh | sh

# 第2步：启动并登录（会弹出浏览器让你认证）
sudo tailscale up

# 第3步：查看分配到的IP
tailscale status
# 100.x.x.x    server    username@  linux
# 100.x.x.y    laptop    username@  macOS
```

之后就简单了：`ssh anson@100.x.x.x` 代替 `ssh anson@公网IP`。tailscale自动处理NAT穿透、加密传输和身份认证。你不需要在服务器上开放任何端口，tailscale会自己想办法打洞连上你。

```{admonition} 免费套餐够用吗？
:class: tip

个人用户完全够。最多100台设备，支持子网路由和ACL访问控制。如果你只是连自己的一台GPU服务器加一台笔记本，免费套餐绰绰有余。
```

### 进阶：让你的服务器当"网桥"

如果GPU服务器在一个局域网里（比如学校实验室），旁边还有NAS、另一台服务器、共享存储——你想通过tailscale访问整个内网：

```bash
# 在服务器上通告整个192.168.1.0/24子网
sudo tailscale up --advertise-routes=192.168.1.0/24

# 然后在 tailscale 管理后台（https://login.tailscale.com）开启子网路由
# 之后不管你在哪里，都可以通过192.168.1.x访问实验室的任意设备
```

(frp-basics)=
## frp：没有公网IP？找个"中转站"

tailscale虽然好，但有些场景它搞不定：网络环境把UDP封了、你需要**对外暴露HTTP服务**（比如给别人用模型API）、或者就是不能用tailscale。这时候frp上场。

frp需要一个**有公网IP的机器**当中转。你的请求先发到中转服务器，它再转发给你的GPU服务器：

```{mermaid}
flowchart LR
    A[你的笔记本] -->|请求| B["公网中转服务器<br/>frps"]
    B -->|转发| C["GPU服务器<br/>frpc"]

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#e8f5e9
```

### 中转端配置（公网服务器）

```ini
# frps.toml —— 就这么几行
bindPort = 7000
```

```bash
# 启动
./frps -c frps.toml
```

### GPU服务器配置（没公网IP的那台）

```ini
# frpc.toml
serverAddr = "你的公网服务器IP"
serverPort = 7000

# 把SSH暴露到公网的6000端口
[[proxies]]
name = "ssh"
type = "tcp"
localIP = "127.0.0.1"
localPort = 22
remotePort = 6000

# 把Jupyter暴露到公网的8888端口
[[proxies]]
name = "jupyter"
type = "tcp"
localIP = "127.0.0.1"
localPort = 8888
remotePort = 8888
```

```bash
# 启动
./frpc -c frpc.toml
```

之后 `ssh anson@公网服务器IP -p 6000` 就连上了你的GPU服务器，Jupyter在浏览器打开 `公网服务器IP:8888` 就能用。

```{list-table} tailscale vs frp 一句话版
:header-rows: 1

* - **对比项**
  - **tailscale**
  - **frp**
* - 要不要公网服务器
  - 不要，机器之间直连
  - 必须有一台
* - 流量怎么走
  - P2P直连，不绕路
  - 全经过中转服务器
* - 对外暴露服务
  - 不太擅长
  - 天生适合
* - 配置麻烦程度
  - 装完就跑
  - 得写配置文件
```

```{admonition} 推荐策略
:class: tip

- **自己用** → tailscale。免费、简单、安全，不用管端口和防火墙
- **给别人用**（API服务、演示）→ frp。配合Nginx反向代理还能加域名和HTTPS
- **网络限制太严** → frp的WebSocket隧道可以尝试绕过一些防火墙
```

(iptables-basics)=
## iptables：给你的服务器上个锁
结合{ref}`tailscale-basics`的"默认就在加密网络里"，加上iptables的精细控制，你的服务器基本安全就有了保障。

前面说的都是"怎么连进来"的问题。还有一个反面问题：**怎么不让不该连的人连进来**。深度学习服务器放在公网上，半小时内就会收到各种扫描和爆破尝试。

iptables是Linux最底层的防火墙，你不需要成为专家，但掌握几条规则能让服务器安全很多：

```bash
# 先看看当前有什么规则
sudo iptables -L -n -v

# 只允许你的IP连SSH（其他人都滚蛋）
sudo iptables -A INPUT -p tcp --dport 22 -s 你的IP地址 -j ACCEPT

# 允许你实验室的整个网段访问模型API
sudo iptables -A INPUT -p tcp --dport 8080 -s 192.168.1.0/24 -j ACCEPT

# 除此以外所有访问8080的请求都拒绝
sudo iptables -A INPUT -p tcp --dport 8080 -j DROP

# 但已经连上的连接不要断（不然你正在跑的SSH会被自己踢掉）
sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# 默认策略：进门全关，出门全开
sudo iptables -P INPUT DROP
sudo iptables -P FORWARD DROP
sudo iptables -P OUTPUT ACCEPT
```

如果你想把同一个端口换个映射（比如外网8080对应内网Jupyter的8888）：

```bash
sudo iptables -t nat -A PREROUTING -p tcp --dport 8080 -j REDIRECT --to-port 8888
```

**保存规则让重启后还在**：

```bash
sudo apt install iptables-persistent
sudo netfilter-persistent save
```

```{warning} 别把自己锁在外面
:class: danger

配置防火墙时，**千万不要关掉当前SSH连接的窗口**。开一个新终端测试新规则——确认能连上再关旧的。如果不小心把自己锁了，只能指望服务器有IPMI/iDRAC或者你认识机房管理员。
```

---

远程连接的问题解决了。但还有一件事：**训练跑到一半终端关了怎么办？服务怎么开机自启？** 最后一节{doc}`server-management`来收尾。
