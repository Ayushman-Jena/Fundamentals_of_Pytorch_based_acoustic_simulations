import torch
import torch.nn as nn
import numpy as np

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Neural Network
# -----------------------------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# -----------------------------
# Parameters
# -----------------------------
c = 1.0  # wave speed
model = PINN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# Training Data (collocation points)
# -----------------------------
N = 1000

x = torch.rand(N, 1).to(device)
t = torch.rand(N, 1).to(device)

x.requires_grad = True
t.requires_grad = True

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(2000):

    optimizer.zero_grad()

    # Prediction
    u = model(x, t)

    # First derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

    # Second derivatives
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    # PDE residual
    residual = u_tt - c**2 * u_xx

    loss_pde = torch.mean(residual**2)

    # -----------------------------
    # Initial condition: u(x,0) = sin(pi x)
    # -----------------------------
    t0 = torch.zeros_like(x)
    u0 = model(x, t0)
    loss_ic = torch.mean((u0 - torch.sin(np.pi * x))**2)

    # -----------------------------
    # Boundary conditions: u(0,t)=0, u(1,t)=0
    # -----------------------------
    x0 = torch.zeros_like(t)
    x1 = torch.ones_like(t)

    u_left = model(x0, t)
    u_right = model(x1, t)

    loss_bc = torch.mean(u_left**2) + torch.mean(u_right**2)

    # Total loss
    loss = loss_pde + loss_ic + loss_bc

    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# -----------------------------
# Testing / Visualization
# -----------------------------
x_test = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)
t_test = torch.full_like(x_test, 0.5).to(device)

u_pred = model(x_test, t_test).detach().cpu().numpy()

import matplotlib.pyplot as plt

plt.plot(x_test.cpu(), u_pred)
plt.title("Wave at t = 0.5")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.show()
