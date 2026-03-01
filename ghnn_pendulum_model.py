import torch
import torch.nn as nn

# 铁律一：必须是“可分离的哈密顿量”（分离出独立的势能网络和动能网络）
class SeparableHamiltonianNet(nn.Module):
    def __init__(self, hidden_dim=25, hidden_layers=2):
        super().__init__()
        # 势能网络 U(q)
        layers_U = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers_U += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers_U += [nn.Linear(hidden_dim, 1)]
        self.U_net = nn.Sequential(*layers_U)
        
        # 动能网络 T(p)
        layers_T = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers_T += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers_T += [nn.Linear(hidden_dim, 1)]
        self.T_net = nn.Sequential(*layers_T)

    def forward(self, q, p):
        # 升维输入，计算得到分离的势能和动能
        U = self.U_net(q.unsqueeze(-1)).squeeze(-1)
        T = self.T_net(p.unsqueeze(-1)).squeeze(-1)
        return U, T

class GHNN(nn.Module):
    def __init__(self, hidden_dim=25, hidden_layers=2):
        super().__init__()
        # 替换为可分离网络
        self.H = SeparableHamiltonianNet(hidden_dim, hidden_layers)

    def forward(self, x, step=0.01):
        q = x[:, 0].clone().requires_grad_(True)
        p = x[:, 1].clone().requires_grad_(True)
        
        # 铁律二：严格遵守交错的“辛欧拉积分”（Symplectic Euler）
        
        # 1. 先用当前的 q 计算势能 U(q) 并求导，更新得到 p_next
        U, _ = self.H(q, p)
        grad_q = torch.autograd.grad(outputs=U.sum(), inputs=q, create_graph=True)[0]
        p_next = p - step * grad_q
        
        # 2. 核心修复：必须用【刚更新出来的 p_next】计算动能 T(p_next) 并求导
        _, T_next = self.H(q, p_next)
        grad_p = torch.autograd.grad(outputs=T_next.sum(), inputs=p_next, create_graph=True)[0]
        q_next = q + step * grad_p  # 用新动量更新位置
        
        return torch.stack([q_next, p_next], dim=1)