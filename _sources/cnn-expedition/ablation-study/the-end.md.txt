(cnn-ablation-end)=

# 结语

拆完了。每件装备的数据摆在眼前。

从{doc}`introduction` 的方法论，到{doc}`experiment-design` 的实验方案，再到{doc}`implementation` 的代码实现——你走完了"提出假设→设计实验→收集数据→分析结论"的完整闭环。

---

## 装备清单

| 装备 | 不带它跌多少 | 重要性 | 作用 |
| ---------- | ---------- | ---------- | ---------- |
| 卷积层 | 12.6% | ⭐⭐⭐ 必须带 | 冰镐——没有它根本爬不了 |
| ReLU 激活 | 15.6% | ⭐⭐⭐ 必须带 | 呼吸方式——高海拔还能保持效率 |
| 池化层 | 6.2% | ⭐⭐ 尽量带 | 减轻负重——不牺牲太多精度 |
| 批归一化 | +2.9% | ⭐⭐ 尽量带 | 稳定步伐——加速收敛，允许更大学习率 |
| Dropout | 1.2% | ⭐ 看情况 | 备用装备——数据少时有用 |

> "Extraordinary claims require extraordinary evidence." —— Carl Sagan

消融研究的本质不是"证明哪个组件好"，而是**用数据回答"为什么好"**。{doc}`../practice-peak/index` 给了你搭建网络的菜单，{doc}`../../pytorch-practice/index` 给了你实现代码的工具，本章给了你**验证和优化设计的实验方法**。三个原则：**控制变量，数据说话，科学即迭代**。

---

## 拆完了。然后呢？

装备清单到手了——卷积层是冰镐，ReLU 是呼吸方式，Dropout 可带可不带。

但这份清单是在练习峰上测的。28×28 的灰度山。真正的山——ImageNet，224×224 彩色照片，1,000 个类别——比练习峰大 1,000 倍。在 MNIST 上"Dropout 影响不大"的结论，放到 ImageNet 上会不会刚好相反？

拆装备是为了理解。理解是为了去爬更高的山。

{doc}`../image-net-era/index`——走。2012 年，炮声响了。

---

## 延伸

**想继续拆？** 
- 对 ResNet / EfficientNet / ViT 做消融，验证练习峰上的结论在更大的架构上是否成立
- 研究多因素交互效应——组件 A 和 B 同时移除的效果是否等于各自效果之和
- 用 Optuna / NNI 做自动化消融，贝叶斯优化替代手动逐一实验

**本章相关章节**：{doc}`../practice-peak/cnn-basics`（组件原理）、{doc}`../practice-peak/neural-training-basics`（正则化）、{doc}`../../pytorch-practice/using-framework`（实验管理）

**经典消融研究**：ResNet（残差连接）、Dropout（比例）、Network In Network（GAP vs FC）

---

```{bibliography}
:filter: docname in docnames
```
