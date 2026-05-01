(why-linux)=
# Linux基础：为什么炼丹用Linux

你刚拿到一台深度学习服务器，装好了系统（或者商家预装了），打开终端后看见一个黑乎乎的界面——然后呢？

在开始配置驱动、安装CUDA之前，先回答一个根本问题：**为什么整个深度学习生态都建在Linux上？**

## 为什么是Linux

```{list-table} 三大桌面系统的深度学习支持对比
:header-rows: 1

* - **维度**
  - **Linux (Debian/Ubuntu)**
  - **Windows**
  - **macOS**
* - 驱动支持
  - 官方驱动，内核级集成
  - 依赖系统更新，时有兼容问题
  - Apple Silicon MPS 表现不错
* - GPU加速
  - NVIDIA CUDA 完整生态
  - CUDA 支持，Docker 透传麻烦
  - MPS 后端可用，性能可观
* - 包管理
  - apt一键装全家桶
  - 手动下载安装包
  - brew，但生态不如Linux
* - 无头运行
  - 不需要图形界面，SSH就能管
  - 需要远程桌面，带宽消耗大
  - 可以但别扭
* - 生产环境
  - 95%以上服务器用Linux
  - 极少用于生产
  - 几乎不用
* - 适合场景
  - 训练+部署全链路
  - 入门学习
  - 开发调试+轻量训练
```

Linux在深度学习领域的统治地位不是偶然的。NVIDIA的CUDA生态在Linux上最成熟，Docker容器化部署在Linux上是原生体验，大多数深度学习框架的CI/CD只针对Linux和macOS做测试。简单说：**你用Linux，遇到问题Google一下就有答案；你不用Linux，遇到问题可能连Google都救不了你。**

## 选择哪个发行版

深度学习领域最常见的两个选择是 **Debian** 和 **Ubuntu**。两者都是Debian系，用同一个包管理器（apt），命令完全兼容。

| 特性 | Debian | Ubuntu |
|------|--------|--------|
| 稳定性 | 极其稳定，包版本较旧 | 较新，但可能不稳定 |
| CUDA支持 | 需要手动添加non-free源 | 官方APT源直接支持 |
| 社区资源 | 深度学习相关内容较少 | 教程最多，遇到问题好搜 |
| 推荐场景 | 生产服务器 | 开发/学习用机器 |

```{admonition} 推荐选择
:class: tip

如果你拿不准，选 **Ubuntu LTS**（如22.04或24.04）。这是深度学习领域事实上的标准系统，NVIDIA的文档和教程都以它为例。等你熟悉了Linux再换Debian不迟。
```

(apt-basics)=
## 包管理基础

Debian系用 `apt` 管理软件包。以下命令能覆盖90%的日常需求：

```bash
# 更新软件源（换源后或首次使用必须执行）
sudo apt update

# 升级所有已安装的包
sudo apt upgrade

# 安装软件（以安装Python为例）
sudo apt install python3 python3-pip

# 卸载软件
sudo apt remove python3

# 搜索软件
apt search nvidia-driver

# 查看已安装的包
apt list --installed | grep nvidia
```

```{admonition} 为什么要 sudo？
:class: note

普通用户没有权限安装系统级软件。`sudo` 让你临时以root身份执行命令（{ref}`user-permissions`会详细讲用户和权限）。首次使用 `sudo` 时需要输入当前用户的密码——注意：**终端输入密码时不会显示任何字符（不回显）。** 这是正常行为，输完按回车就行。
```

一个常见的陷阱：**不要用 `sudo pip install`**。这会把Python包装到系统目录，和apt管理的包冲突。正确的做法是用虚拟环境隔离项目依赖。

### uv：现代Python包管理

传统上Python用 `pip` 装包、`venv` 管环境、`pip-tools` 管依赖锁定——三个工具各管一事。**uv** 是一个用Rust写的Python包管理器，把这套东西全包了，而且快了一个数量级：

```bash
# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并激活
uv venv
source .venv/bin/activate

# 安装包（自动创建虚拟环境，无需手动 venv + activate）
uv pip install torch torchvision

# 从 requirements.txt 安装
uv pip install -r requirements.txt

# 锁定依赖版本（生成 uv.lock）
uv pip compile requirements.txt -o requirements.txt.lock

# 根据 lock 文件安装（保证团队环境一致）
uv pip sync requirements.txt.lock
```

```{list-table} pip + venv vs uv
:header-rows: 1

* - **操作**
  - **pip + venv**
  - **uv**
* - 创建环境
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `uv venv`（自动激活）
* - 安装依赖
  - `pip install -r requirements.txt`
  - `uv pip install -r requirements.txt`（快10-100倍）
* - 依赖解析
  - 无内置，需 `pip-tools`
  - 内置 `uv pip compile`
* - Python版本管理
  - 手动安装不同版本
  - `uv python install 3.12`
* - 底层语言
  - Python
  - Rust（无GIL，全局解释器锁问题）
```

最重要的优势是**速度**和**确定性**：`uv pip compile` 会解析出完整的依赖树并锁定版本，你和队友拿到同一个 lock 文件安装出来的环境完全一致，不会出现"我机器上能跑你机器上报错"的情况。

```{admonition} uv 和 conda 的关系
:class: note

uv 不管理非Python包（如CUDA驱动、系统库）。如果你需要 conda 那种跨语言包管理（Python + C++库 + 系统工具），可以继续用 conda 或 mamba。但纯Python依赖管理，uv 是更好的选择。
```

## 系统管理基础

### systemd：管理服务

深度学习服务器上，很多组件（SSH、Docker、监控）都是作为系统服务运行的。systemd是管理这些服务的标准工具：

```bash
# 查看服务状态
systemctl status sshd

# 启动/停止/重启服务
sudo systemctl start sshd
sudo systemctl stop sshd
sudo systemctl restart sshd

# 设置开机自启
sudo systemctl enable sshd
```

(user-permissions)=
### 用户和权限

```bash
# 创建新用户
sudo adduser alice

# 给用户sudo权限
sudo usermod -aG sudo alice

# 切换用户
su - alice

# 修改文件/目录所有者
sudo chown alice:alice /data/datasets

# 修改权限（r=4, w=2, x=1）
chmod 755 script.sh    # 所有者可读写执行，其他人可读执行
chmod 600 id_rsa       # 私钥文件必须600权限，否则SSH会报错
```

### 磁盘和存储

训练数据集动辄几十GB，了解磁盘使用情况是常备技能：

```bash
# 查看磁盘分区和使用情况
df -h

# 查看当前目录下各子目录的大小
du -sh *

# 查看当前目录总大小
du -sh .

# 挂载新硬盘
sudo mount /dev/sdb1 /mnt/data

# 开机自动挂载（编辑 /etc/fstab）
# UUID=xxxx-xxxx  /mnt/data  ext4  defaults  0  2
```

### 进程管理

训练一个模型通常要跑几小时甚至几天，掌握进程管理能帮你避免"训练跑了一半终端关了"的惨案：

```bash
# 查看所有进程
ps aux

# 按CPU或内存排序
ps aux --sort=-%cpu
ps aux --sort=-%mem

# 实时监控（类似任务管理器）
htop

# 终止进程
kill -9 PID

# 让进程在退出终端后继续运行
nohup python train.py > train.log 2>&1 &
```

```{admonition} nohup 和 tmux 的选择
:class: tip

`nohup` 是最简单的后台运行方式，但管理多个进程不方便。生产环境建议用 tmux（{ref}`tmux-basics`会详细介绍），它让你能创建多个终端会话，随时断开和重新连接。
```

(ssh-setup)=
## SSH：远程连接服务器

深度学习服务器很少在你手边——可能在机房，可能在云上。SSH是你和服务器之间的桥梁。本节讲基础配置，{ref}`ssh-tunnel`会深入到端口转发等高级用法。

### 密钥认证（推荐，不要用密码）

```bash
# 在本地机器上生成密钥对
# -t rsa: 算法类型
# -b 4096: 密钥长度
# -C "注释，通常是邮箱"
ssh-keygen -t rsa -b 4096 -C "your@email.com"

# 查看公钥内容（把这个给服务器管理员）
cat ~/.ssh/id_rsa.pub

# 将公钥复制到服务器（如果有密码权限）
ssh-copy-id user@your-server-ip
```

```{admonition} 为什么要用密钥而不是密码？
:class: important

密码认证有两个大问题：一是弱密码容易被暴力破解（服务器放在公网上半小时内就会收到扫描）；二是每次登录都要输入密码。密钥认证用一对公钥-私钥文件代替了密码输入，私钥存在本地，别人拿不到就不可能登录你的服务器。
```

### SSH配置文件

`~/.ssh/config` 可以让你不用每次都输入完整的连接信息：

```
# ~/.ssh/config
Host gpu-server
    HostName 123.123.123.123
    User anson
    Port 22
    IdentityFile ~/.ssh/id_rsa

Host lab-machine
    HostName 192.168.1.100
    User alice
    Port 2222

# 配置好后就可以用短名称连接了
# ssh gpu-server   # 等价于 ssh anson@123.123.123.123 -p 22 -i ~/.ssh/id_rsa
```

### 安全加固

服务器放在公网上不需要太多配置，但以下几个措施值得做：

```bash
# 1. 禁止root直接登录
# 编辑 /etc/ssh/sshd_config
PermitRootLogin no

# 2. 禁止密码认证
PasswordAuthentication no

# 3. 修改默认端口（减少被扫描的概率）
Port 2222

# 修改后重启SSH服务
sudo systemctl restart sshd
```

```{warning}
修改SSH配置前，确保你已经用**另一个终端窗口**测试过新配置能连上。如果配错了把自己锁在外面，就只能去机房或者找运维帮忙了。测试方法：新开一个终端，用新的配置尝试连接，确认能登录后再关掉旧的窗口。
```

## 常用命令速查

| 命令 | 作用 | 示例 |
|------|------|------|
| `ls -lh` | 列出文件（人类可读大小） | `ls -lh /data` |
| `cd` | 切换目录 | `cd /mnt/data/datasets` |
| `cp -r` | 递归复制目录 | `cp -r /data/raw /data/backup` |
| `mv` | 移动/重命名 | `mv old_name new_name` |
| `rm -r` | 删除目录（慎用！） | `rm -r /tmp/cache` |
| `rm` | 删除文件（慎用！） | `rm file_path` |
| `grep` | 搜索文本 | `grep "error" train.log` |
| `wc -l` | 统计行数 | `wc -l dataset.csv` |
| `tar -xzf` | 解压 tar.gz | `tar -xzf imagenet.tar.gz` |
| `wget` | 下载文件 | `wget https://example.com/file.zip` |
| `curl` | HTTP请求 | `curl http://localhost:8080/health` |

```{admonition} rm -rf 的警告
:class: danger

`rm -rf /` 会删除你系统上的所有文件。不要在 **任何** 以 `/` 开头的路径上运行 `rm -rf`，除非你100%确定自己在做什么。一个安全习惯：删除前先用 `ls` 确认路径。
```

---

掌握了这些基础操作，你就可以开始配置深度学习环境了。下一节 {doc}`nvidia-setup` 会带你安装NVIDIA驱动和CUDA，让你的GPU真正派上用场。
