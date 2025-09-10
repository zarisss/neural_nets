##SOLVING 1D HEAT EQUATION USING PINN
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
    def forward(self, t, x):
        input_tensor = torch.cat([t, x], dim=1)
        return self.net(input_tensor)

alpha = 0.1
def loss(model, t_ic, x_ic, t_bc, x_bc, t_phys, x_phys):
    #L_IC --> Loss for initial condition
    #Equation--> L_IC = MSE(U_nn - (-sin(pi(x))))
    u_pred_ic = model(t_ic, x_ic)
    u_true_ic = -torch.sin(torch.pi * x_ic)
    loss_ic = torch.mean((u_pred_ic - u_true_ic)**2)

    #L_BC --> loss for boundary condition
    #equation --> L_bc = MSE(U_nn(t_i, -1), theta) + MSE((U_nn(t_i, 1), theta2)
    u_pred_bc = model(t_bc, x_bc)
    loss_bc = torch.mean((u_pred_bc - 0.0)**2)

    #Physics loss
    t_phys.requires_grad_(True)
    x_phys.requires_grad_(True)

    u = model(t_phys, x_phys)

    #calculating the first derivative using autograd.grad
    grad_u = torch.autograd.grad(u, [t_phys, x_phys], grad_outputs=torch.ones_like(u), create_graph=True)
    u_t = grad_u[0]
    u_x = grad_u[1]
    u_xx = torch.autograd.grad(u_x, x_phys, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    residual = u_t - alpha * u_xx
    physics_loss = torch.mean(residual**2)

    total_loss = loss_ic + loss_bc + physics_loss

    return total_loss

# Instantiate the model
pinn_model = PINN().to(device)
# Instantiate the optimizer (Adam is a great choice)
optimizer = torch.optim.Adam(pinn_model.parameters(), lr=1e-3)

# --- Training Loop ---
for epoch in range(10000):
    # In a real implementation, you'd resample these points each epoch
    # For simplicity, we define them once here.
    # N_ic, N_bc, N_phys would be the number of points for each loss component.
    t_ic = torch.zeros((100, 1)).to(device)
    x_ic = (torch.rand((100, 1)) * 2 - 1).to(device) # x in [-1, 1]

    t_bc = torch.rand((100, 1)).to(device) # t in [0, 1]
    x_bc = torch.cat([torch.ones(50, 1) * -1, torch.ones(50, 1)]).to(device) # x = -1 and x = 1

    t_phys = torch.rand((1000, 1)).to(device) # t in [0, 1]
    x_phys = (torch.rand((1000, 1)) * 2 - 1).to(device) # x in [-1, 1]
    
    # This is the core optimization step
    optimizer.zero_grad()
    losss = loss(pinn_model, t_ic, x_ic, t_bc, x_bc, t_phys, x_phys)
    losss.backward() # This computes d(L_Total)/d(theta)
    optimizer.step() # This updates theta based on the gradients
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {losss.item()}")
pinn_model.eval()

# Create a grid of points to evaluate the solution
t = np.linspace(0, 1, 100)
x = np.linspace(-1, 1, 100)
t_grid, x_grid = np.meshgrid(t, x)

# Convert grid to PyTorch tensors
t_tensor = torch.tensor(t_grid.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
x_tensor = torch.tensor(x_grid.flatten(), dtype=torch.float32).unsqueeze(1).to(device)

# Get the model's prediction
with torch.no_grad():
    u_pred = pinn_model(t_tensor, x_tensor)

# Reshape the prediction to match the grid and move to CPU for plotting
u_pred_grid = u_pred.reshape(t_grid.shape).cpu().numpy()

# Plot the result
plt.figure(figsize=(8, 6))
plt.pcolormesh(t_grid, x_grid, u_pred_grid, shading='auto', cmap='hot')
plt.colorbar(label='Temperature u(t,x)')
plt.xlabel('Time (t)')
plt.ylabel('Position (x)')
plt.title('PINN Solution of the 1D Heat Equation')
plt.show()
