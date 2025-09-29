#!/bin/bash

# 编译LaTeX文档
echo "正在编译LaTeX文档..."
xelatex -interaction=nonstopmode mnist.tex

# 如果需要多次编译以解决交叉引用
echo "第二次编译以解决交叉引用..."
xelatex -interaction=nonstopmode mnist.tex

echo "编译完成！"
