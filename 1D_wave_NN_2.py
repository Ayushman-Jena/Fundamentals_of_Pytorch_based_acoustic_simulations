import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Neural Network
# -----------------------------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

model = PINN().to(device)

# Wave speed
c = 1.0

# -----------------------------
# Training Data
# -----------------------------
def generate_points(N_f=10000, N_bc=2000, N_ic=2000):
    # Collocation points
    x_f = torch.rand(N_f, 1)
    t_f = torch.rand(N_f, 1)

    # Boundary points (x=0 and x=1)
    t_bc = torch.rand(N_bc, 1)
    x_bc0 = torch.zeros(N_bc, 1)
    x_bc1 = torch.ones(N_bc, 1)

    # Initial condition (t=0)
    x_ic = torch.rand(N_ic, 1)
    t_ic = torch.zeros(N_ic, 1)

    return x_f, t_f, x_bc0, x_bc1, t_bc, x_ic, t_ic

# -----------------------------
# Loss Function
# -----------------------------
def loss_function():
    x_f, t_f, x_bc0, x_bc1, t_bc, x_ic, t_ic = generate_points()

    x_f = x_f.to(device).requires_grad_(True)
    t_f = t_f.to(device).requires_grad_(True)

    # PDE residual
    u = model(x_f, t_f)

    u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u),
                             retain_graph=True, create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_f, grad_outputs=torch.ones_like(u_t),
                              retain_graph=True, create_graph=True)[0]

    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u),
                             retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x),
                              retain_graph=True, create_graph=True)[0]

    f = u_tt - c**2 * u_xx
    loss_pde = torch.mean(f**2)

    # Boundary conditions
    t_bc = t_bc.to(device)
    x_bc0 = x_bc0.to(device)
    x_bc1 = x_bc1.to(device)

    u_bc0 = model(x_bc0, t_bc)
    u_bc1 = model(x_bc1, t_bc)

    loss_bc = torch.mean(u_bc0**2) + torch.mean(u_bc1**2)

    # Initial condition
    x_ic = x_ic.to(device).requires_grad_(True)
    t_ic = t_ic.to(device).requires_grad_(True)

    u_ic = model(x_ic, t_ic)
    u_ic_t = torch.autograd.grad(u_ic, t_ic, grad_outputs=torch.ones_like(u_ic),
                                retain_graph=True, create_graph=True)[0]

    u_true = torch.sin(np.pi * x_ic)

    loss_ic = torch.mean((u_ic - u_true)**2) + torch.mean(u_ic_t**2)

    return loss_pde + loss_bc + loss_ic

# -----------------------------
# Training
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 5000

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = loss_function()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# -----------------------------
# Visualization
# -----------------------------
x = np.linspace(0, 1, 200)
t = 0.5 * np.ones_like(x)

x_torch = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)
t_torch = torch.tensor(t, dtype=torch.float32).unsqueeze(1).to(device)

u_pred = model(x_torch, t_torch).detach().cpu().numpy()

plt.plot(x, u_pred, label="PINN Prediction")
plt.xlabel("x")
plt.ylabel("u(x,t=0.5)")
plt.title("1D Wave Solution using PINN")
plt.legend()
plt.show()