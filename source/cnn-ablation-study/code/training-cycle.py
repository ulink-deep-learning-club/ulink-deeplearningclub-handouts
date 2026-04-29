"""
消融实验训练入口。

将模型注册到框架的 ModelRegistry，通过 --model / --config 参数切换实验。
"""
from base_model import BaselineCNN
from batch_normal import CNNWithBN
from dropout import CNNWithDropout

# 注册到框架 ModelRegistry（在 src/models/registry.py 中添加）：
#   ModelRegistry.register("baseline_cnn", BaselineCNN)
#   ModelRegistry.register("cnn_with_bn", CNNWithBN)
#   ModelRegistry.register("cnn_with_dropout", CNNWithDropout)

# 运行实验：
#   python train.py --model baseline_cnn --dataset cifar10
#   python train.py --model cnn_with_bn --dataset cifar10
#   python train.py --model cnn_with_dropout --dataset cifar10
