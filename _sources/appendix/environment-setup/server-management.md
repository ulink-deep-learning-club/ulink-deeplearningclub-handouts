(server-management)=
# 服务管理：让训练一直跑，服务一直开

驱动装好了，远程能连上了——但这只是开始。真正的考验是：**你关掉笔记本电脑回家的路上，训练还在跑吗？**

这一节解决两个核心问题：一是让任务在你离线后继续运行（tmux），二是让服务在服务器重启后自动恢复（systemd）。顺便聊聊面板和监控。

(tmux-basics)=
## tmux：终端"续命"神器

这是深度学习最实用的工具，没有之一。{ref}`ssh-setup`和{ref}`mosh-basics`帮你连上服务器，但tmux让你的训练在你断开后继续跑。

没有tmux的时候，你的工作流是这样的：

```bash
python train.py  # 训练开始，预计跑6小时
# ... 你去吃午饭了 ...
# 回来发现：终端超时断开了，训练白跑了6小时
```

有了tmux：

```bash
tmux new -s training  # 创建名为 training 的会话
python train.py        # 在tmux里启动训练
# Ctrl+B 然后按 D 断开（detach），会话还在跑
# 几小时后重新连上：
tmux attach -t training  # 回到训练现场，一切正常
```

### 日常操作就记四组快捷键

tmux的学习曲线平缓——日常只用记四组快捷键就够了：

| 操作 | 快捷键 | 说明 |
|------|--------|------|
| 断开会话 | `Ctrl+B` → `D` | 断开但不终止，任务后台继续跑 |
| 新建窗口 | `Ctrl+B` → `C` | 在同一个会话里开新终端 |
| 切换窗口 | `Ctrl+B` → `窗口数字` | 在多个任务间切换 |
| 上下分屏 | `Ctrl+B` → `"` | 一个窗口拆成上下两半 |
| 左右分屏 | `Ctrl+B` → `%` | 一个窗口拆成左右两半 |
| 在分屏间跳转 | `Ctrl+B` → `方向键` | 切换焦点到另一个分屏 |

```{admonition} tmux vs nohup vs screen
:class: tip

- **nohup**：最简单，`nohup python train.py &`，但管理多个进程不方便，不能重连到控制台
- **screen**：tmux的前辈，功能类似但默认不装、界面丑、分屏弱
- **tmux**：现在是主流选择，配置漂亮、分屏强大、会话管理灵活

三句话：**nohup解决"有没有"的问题，tmux解决"好不好用"的问题。** 有条件就用tmux。
```

### 真实世界的训练工作流

结合mosh（{doc}`remote-access`）和tmux，你的日常会变成这样：

```bash
# 1. 用mosh连服务器（不怕切WiFi断连）
mosh user@server

# 2. 创建tmux会话
tmux new -s experiment-0429

# 3. 启动训练
python train.py --config configs/experiment.toml

# 4. Ctrl+B → D 断开，关掉笔记本电脑走人

# 5. 晚上回家，重新连上服务器
mosh user@server
tmux attach -t experiment-0429  # 训练还在跑，loss曲线实时更新
```

```{warning} 一个tmux新手最容易踩的坑
:class: warning

不要在tmux里启动Jupyter Notebook。因为Jupyter会打开浏览器，而tmux没有浏览器——它会卡住。正确的做法是在普通SSH窗口里开Jupyter（绑定127.0.0.1），然后通过SSH隧道访问。tmux只用来跑训练脚本、预处理数据这些不需要界面的任务。
```

### 自定义tmux配置

如果你觉得默认的tmux界面太素，可以加个简单的配置：

```
# ~/.tmux.conf
# 让Ctrl+B更顺手（连续按两次更快）
set -g escape-time 0

# 开启鼠标支持（可以用鼠标点窗口、拖分屏）
set -g mouse on

# 窗口列表显示在底部
set -g status-position bottom

# 每60秒自动刷新状态栏
set -g status-interval 60
```

改完配置后，在tmux里按 `Ctrl+B` → `:` → 输入 `source-file ~/.tmux.conf` 重载。

(systemd-basics)=
## systemd：让服务开机自己跑

tmux解决的是"我走了训练别停"的问题。还有一个更根本的问题：**服务器重启了，服务还能自己起来吗？**

系统管理员不可能每次重启都手动去启动服务。systemd就是Linux的标准答案——它是一个"服务管家"，负责在开机时启动你指定的程序，并在程序崩溃后自动重启。

### 写一个service文件

假设你想让Jupyter Notebook开机自启，可以写一个systemd service：

```
# /etc/systemd/system/jupyter.service
[Unit]
Description=Jupyter Notebook Server
After=network.target

[Service]
Type=simple
User=anson
WorkingDirectory=/home/anson/projects
ExecStart=/home/anson/.local/bin/jupyter notebook --ip=127.0.0.1 --port=8888 --no-browser
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

各字段的含义：

| 字段 | 意思 | 为什么重要 |
|------|------|-----------|
| `After=network.target` | 等网络就绪再启动 | 有些服务依赖网络，不等就会报错 |
| `User=anson` | 以哪个用户身份运行 | 不要用root跑服务，有安全风险 |
| `Restart=on-failure` | 失败了自动重试 | 程序崩溃了systemd会自动拉起来 |
| `RestartSec=5` | 重试前等5秒 | 防止频繁重启把系统搞崩 |
| `WantedBy=multi-user.target` | 在多用户模式下启动 | 一般服务都写这个 |

### 管理service

```bash
# 重新加载配置文件（改了service文件后必须执行）
sudo systemctl daemon-reload

# 启用开机自启
sudo systemctl enable jupyter

# 立即启动
sudo systemctl start jupyter

# 查看状态
sudo systemctl status jupyter

# 查看日志
sudo journalctl -u jupyter -f

# 停止
sudo systemctl stop jupyter

# 不想开机自启了
sudo systemctl disable jupyter
```

```{admonition} 一个排错技巧
:class: tip

`systemctl status` 显示的日志通常只保留最后几行。如果服务启动失败且原因不明显，用 `journalctl -u 服务名 -f` 查看完整日志。`-f` 会持续跟踪新日志，你可以同时重启服务观察报错。
```

### 训练脚本也适合用systemd吗？

不适合。systemd适合**服务类程序**（Jupyter、API服务、监控Agent），不适合**一次性任务**（训练脚本）。训练脚本用{ref}`tmux-basics`管理更灵活。

为什么？因为训练脚本可能需要你手动中断（调参、看loss曲线、改代码），而systemd设计的是"崩溃了就悄悄重试"——你想要的停止可能被systemd误解为崩溃，然后自动重启。

```{list-table} tmux vs systemd 的分工
:header-rows: 1

* - **场景**
  - **用什么**
  - **理由**
* - 跑训练脚本
  - tmux
  - 需要手动交互、调试、中断
* - 开Jupyter/TensorBoard
  - systemd
  - 开机自启，不用每次手动
* - 部署模型API
  - systemd
  - 崩溃自动恢复，日志统一管理
* - 数据预处理
  - tmux
  - 一次性任务，跑完就结束
```

## 定时任务与日志管理

### cron：按计划跑脚本

如果你需要"每天早上8点备份一次数据"或者"每周清理一次缓存"，cron是最简单的方式：

```bash
# 编辑当前用户的crontab
crontab -e

# 格式：分钟 小时 日 月 星期 命令
# 每天凌晨3点执行备份
0 3 * * * /home/anson/scripts/backup.sh

# 每周日晚上10点清理日志
0 22 * * 0 find /home/anson/logs -mtime +7 -delete

# 每半小时检查一次GPU温度（如果超过阈值就发通知）
*/30 * * * * /home/anson/scripts/check_gpu_temp.sh
```

### logrotate：日志别撑爆磁盘

训练日志如果不做管理，几个月就能把磁盘塞满。logrotate是Linux自带的日志轮转工具：

```
# /etc/logrotate.d/training-logs
/home/anson/logs/train.log {
    daily          # 每天轮转一次
    rotate 7       # 保留最近7天的日志
    compress       # 压缩旧日志
    missingok      # 日志文件不存在也不报错
    notifempty     # 空日志不轮转
    copytruncate   # 复制后截断原文件（不影响正在写的进程）
}
```

### 磁盘监控

```bash
# 快速查看磁盘使用情况
df -h

# 找出当前目录下最大的文件/目录
du -sh * | sort -rh | head -5

# 安装ncdu，交互式的磁盘分析工具（比du好用）
sudo apt install ncdu
ncdu /home/anson
```

## 管理面板：图形化监控

如果你不喜欢纯命令行操作，或者想让团队里的非技术成员也能查看服务器状态，可以装一个面板。

### 1Panel

[1Panel](https://github.com/1Panel-dev/1Panel) 是一个开源的Linux服务器管理面板，安装简单：

```bash
curl -sSL https://resource.fit2cloud.com/1panel/package/quick_start.sh -o quick_start.sh
sudo bash quick_start.sh
```

装完后通过浏览器访问 `http://服务器IP:端口`，可以：

- 查看CPU、内存、磁盘、网络使用情况
- 查看GPU温度和显存占用
- 管理Docker容器
- 设置防火墙规则
- 查看系统日志

### GPU专用监控

如果你只需要看GPU的状态，有几个轻量的选择：

```bash
# nvtop：htop风格的GPU监控
sudo apt install nvtop
nvtop

# 用watch + nvidia-smi（最轻量，不需要额外装）
watch -n 2 nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
```

```{admonition} 监控建议
:class: note

- **个人用**：`watch -n 2 nvidia-smi` 就够了，零依赖
- **喜欢图形界面**：nvtop，和htop操作逻辑一样
- **团队用**：1Panel或Prometheus + Grafana，可以历史回溯和告警
- **生产环境**：配置Prometheus exporter + Grafana仪表盘，配合企业微信/钉钉告警
```

---

这一节是附录的最后一篇。从{ref}`why-linux`到{ref}`nvidia-arch`，从{ref}`ssh-tunnel`到{ref}`tmux-basics`——你现在已经拥有了一台"真正能用"的深度学习服务器。回到{doc}`../../index`选择你感兴趣的章节继续学习吧。
