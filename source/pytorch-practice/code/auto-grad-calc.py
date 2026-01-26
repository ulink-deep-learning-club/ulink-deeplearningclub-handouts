"""
自动微分梯度计算示例

本示例演示PyTorch如何自动计算复杂函数的梯度。
函数：z = x·y + ||x||²
理论梯度：∂z/∂x = y + 2x, ∂z/∂y = x
"""

import torch

def main():
    """主函数：演示自动微分梯度计算"""
    
    # 创建需要梯度的张量
    # requires_grad=True 告诉PyTorch跟踪这些张量的操作历史
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    
    print("=" * 60)
    print("自动微分梯度计算示例")
    print("=" * 60)
    print(f"输入张量 x: {x}")
    print(f"输入张量 y: {y}")
    print()
    
    # 定义复杂函数：z = x·y + ||x||²
    # torch.dot(x, y): 向量点积
    # torch.norm(x): 向量范数（L2范数）
    z = torch.dot(x, y) + torch.norm(x) ** 2
    
    print(f"计算函数值 z = x·y + ||x||²")
    print(f"z = {z.item():.4f}")
    print()
    
    # 计算梯度
    # backward() 方法自动计算所有requires_grad=True的张量的梯度
    z.backward()
    
    print("梯度计算结果：")
    print(f"∂z/∂x (x.grad) = {x.grad}")  # 理论值：y + 2x = [4,5,6] + 2*[1,2,3] = [6,9,12]
    print(f"∂z/∂y (y.grad) = {y.grad}")  # 理论值：x = [1,2,3]
    print()
    
    # 验证梯度计算正确性
    expected_x_grad = y + 2 * x
    expected_y_grad = x
    
    print("梯度验证：")
    print(f"理论 ∂z/∂x: {expected_x_grad}")
    print(f"理论 ∂z/∂y: {expected_y_grad}")
    print()
    
    # 检查梯度计算是否正确
    x_grad_correct = torch.allclose(x.grad, expected_x_grad, rtol=1e-5)
    y_grad_correct = torch.allclose(y.grad, expected_y_grad, rtol=1e-5)
    
    if x_grad_correct and y_grad_correct:
        print("✅ 梯度计算正确！")
    else:
        print("❌ 梯度计算有误！")
    
    print("=" * 60)
    
    return x, y, z

if __name__ == "__main__":
    # 运行示例
    x, y, z = main()
    
    # 额外信息：展示计算图相关信息
    print("\n计算图信息：")
    print(f"z.grad_fn: {z.grad_fn}")
    print(f"z.is_leaf: {z.is_leaf}")
    print(f"x.is_leaf: {x.is_leaf}")
    print(f"y.is_leaf: {y.is_leaf}")
    
    # 清理梯度（在实际训练中很重要）
    x.grad.zero_()
    y.grad.zero_()
    print("\n梯度已清零，准备下一次计算。")
