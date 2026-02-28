import torch
import torch.nn as nn

class HamiltonianNet(nn.Module):
    def __init__(self, hidden_dim=60, hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, q, p):
        x = torch.stack([q, p], dim=-1)  # shape [batch, 2]
        return self.net(x).squeeze(-1)   # shape [batch]

class GHNN(nn.Module):
    def __init__(self, hidden_dim=60, hidden_layers=2):
        super().__init__()
        self.H = HamiltonianNet(hidden_dim, hidden_layers)
    def forward(self, x, step=0.01):
        # x: [batch, 2] (q, p) with requires_grad=True
        q = x[:, 0]
        p = x[:, 1]
        q = q.clone().detach().requires_grad_(True)
        p = p.clone().detach().requires_grad_(True)
        H = self.H(q, p)
        grad_q, grad_p = torch.autograd.grad(
            outputs=H.sum(), inputs=[q, p], create_graph=True
        )
        q_next = q + step * grad_p
        p_next = p - step * grad_q
        return torch.stack([q_next, p_next], dim=1)