# -*- coding: utf-8 -*-
"""
Enhanced PINN Implementation without Adaptive Loss Weights and No DataLoader

This script trains a Physics-Informed Neural Network (PINN) to solve the Navier-Stokes equations
for aneurysmal flow dynamics. It avoids data splitting and DataLoader, uses fixed loss weights,
measures training time, evaluates the model, saves training parameters, and plots loss and prediction comparisons.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time  # For tracking training time
import matplotlib.pyplot as plt  # For plotting
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm  # For progress bars
import os  # For file path operations
import json  # For saving parameters

###################################### Helper functions for loading and saving trained models
def load_model(path, model, optimizer=None, scheduler=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{path}' not found.")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    training_params = checkpoint.get('training_params', None)  # Retrieve training parameters if available
    return model, optimizer, scheduler, training_params

def save_model(path, model, optimizer=None, scheduler=None, training_params=None):
    data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'training_params': training_params  # Include training parameters
    }
    torch.save(data, path)

###################################### Helper functions for retrieving CSV data (mesh, CFD results)
def parse_csv_data(path, category, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file '{path}' not found.")
    
    if category == 'mesh':
        data = np.loadtxt(path, delimiter=',', unpack=True)
        if data.shape[0] != 3:
            raise ValueError(f"Expected 3 columns for 'mesh', got {data.shape[0]} columns.")
        x, y, z = data
        x = torch.tensor(x, dtype=torch.float32, device=device).reshape(-1,1)
        y = torch.tensor(y, dtype=torch.float32, device=device).reshape(-1,1)
        z = torch.tensor(z, dtype=torch.float32, device=device).reshape(-1,1)
        return x, y, z
    
    elif category == 'cfd':
        data = np.loadtxt(path, delimiter=',', unpack=True)
        if data.shape[0] != 7:
            raise ValueError(f"Expected 7 columns for 'cfd', got {data.shape[0]} columns.")
        x, y, z, p, u, v, w = data
        x = torch.tensor(x, dtype=torch.float32, device=device).reshape(-1,1)
        y = torch.tensor(y, dtype=torch.float32, device=device).reshape(-1,1)
        z = torch.tensor(z, dtype=torch.float32, device=device).reshape(-1,1)
        p = torch.tensor(p, dtype=torch.float32, device=device).reshape(-1,1)
        u = torch.tensor(u, dtype=torch.float32, device=device).reshape(-1,1)
        v = torch.tensor(v, dtype=torch.float32, device=device).reshape(-1,1)
        w = torch.tensor(w, dtype=torch.float32, device=device).reshape(-1,1)
        return x, y, z, p, u, v, w
    
    else:
        raise ValueError(f"Unknown category: {category}")

###################################### The neural network with configurable parameters
class NSNeuralNet(nn.Module):
    def __init__(self, input_size=3, output_size=4, num_layers=12, units_per_layer=256, activation='silu'):
        super().__init__()
        layers = []
        activation_function = self.get_activation_function(activation)
        # Input layer
        layers.append(nn.Linear(input_size, units_per_layer))
        layers.append(activation_function)
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(units_per_layer, units_per_layer))
            layers.append(activation_function)
        # Output layer
        layers.append(nn.Linear(units_per_layer, output_size))
        self.main = nn.Sequential(*layers)
        
    def forward(self, x, y, z):
        return self.main(torch.cat([x, y, z], axis=1))
        
    def get_activation_function(self, activation):
        activation = activation.lower()
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'silu':
            return nn.SiLU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

###################################### Define the loss functions
loss_func = nn.MSELoss()

def loss_physics(net, x, y, z):
    """
    Physics-informed loss enforcing Navier-Stokes equations.
    """
    # Ensure inputs require gradients for derivative computations
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = z.requires_grad_(True)
    
    qq = net(x, y, z)
    p = qq[:,0].reshape(-1,1)
    u = qq[:,1].reshape(-1,1)
    v = qq[:,2].reshape(-1,1)
    w = qq[:,3].reshape(-1,1)
    del qq

    # First-order derivatives
    ones = torch.ones_like(p)
    p_x = torch.autograd.grad(p, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

    w_x = torch.autograd.grad(w, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

    # Second-order derivatives
    ones_u = torch.ones_like(u_x)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]

    w_xx = torch.autograd.grad(w_x, x, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]

    rho = 1050  # density
    mu = 0.0035  # viscosity

    # Navier-Stokes equations
    eqn_x = p_x + rho * (u*u_x + v*u_y + w*u_z) - mu * (u_xx + u_yy + u_zz)
    eqn_y = p_y + rho * (u*v_x + v*v_y + w*v_z) - mu * (v_xx + v_yy + v_zz)
    eqn_z = p_z + rho * (u*w_x + v*w_y + w*w_z) - mu * (w_xx + w_yy + w_zz)
    eqn_c = u_x + v_y + w_z  # Continuity equation

    # Concatenate all residuals
    dt = torch.cat([eqn_x, eqn_y, eqn_z, eqn_c], axis=1)
    return loss_func(dt, torch.zeros_like(dt))

def loss_data(net, x, y, z, p, u, v, w):
    """
    Data loss comparing network predictions with CFD data.
    """
    predictions = net(x, y, z)
    targets = torch.cat([p, u, v, w], axis=1)
    return loss_func(predictions, targets)

###################################### CONFIGURABLE PARAMETERS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 3  # x, y, z
output_size = 4  # p, u, v, w
num_layers = 12
units_per_layer = 256
activation_function = 'silu'  # Options: 'relu', 'tanh', 'silu', 'elu', 'leaky_relu'

epochs = 1000

load_previous = False
save_end = True
load_fn = 'ns_34_3'
save_fn = 'ns_34_3'

print_every = 50
save_every = 100

learning_rate = 1e-3
lr_steps = 100
lr_gamma = 0.5

file_mesh = 'mesh_1_s.csv'
file_cfd = 'train_1.csv'

batch_mesh = 500 # Batch size for mesh data reduced to 500 from 5000
batch_cfd = 250 # Batch size for CFD data reduced to 250 from 2500

patience = 20  # For early stopping

# Fixed loss weights
# lambda_physics = 1.0  # Weight for physics loss
lambda_data = 10.0     # Weight for data loss

# Collect and save training parameters
training_params = {
    'device': str(device),
    'input_size': input_size,
    'output_size': output_size,
    'num_layers': num_layers,
    'units_per_layer': units_per_layer,
    'activation_function': activation_function,
    'epochs': epochs,
    'load_previous': load_previous,
    'save_end': save_end,
    'load_fn': load_fn,
    'save_fn': save_fn,
    'print_every': print_every,
    'save_every': save_every,
    'learning_rate': learning_rate,
    'lr_steps': lr_steps,
    'lr_gamma': lr_gamma,
    # 'lambda_physics': lambda_physics,
    'lambda_data': lambda_data,
    'file_mesh': file_mesh,
    'file_cfd': file_cfd,
    'batch_mesh': batch_mesh,
    'batch_cfd': batch_cfd,
    'patience': patience,
}
    
# Save the parameters to a JSON file
params_filename = f'{save_fn}_params.json'
with open(params_filename, 'w') as f:
    json.dump(training_params, f, indent=4)
    
print(f'Training parameters saved to {params_filename}')

###################################### Initialize the Neural Network
model = NSNeuralNet(
    input_size=input_size,
    output_size=output_size,
    num_layers=num_layers,
    units_per_layer=units_per_layer,
    activation=activation_function
).to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)

model.apply(init_weights)

###################################### Initialize Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=lr_steps, gamma=lr_gamma)

# Load previous model if specified
if load_previous:
    model, optimizer, scheduler, loaded_params = load_model(f'{load_fn}.pt', model, optimizer, scheduler)
    if loaded_params is not None:
        print(f"Loaded training parameters from {load_fn}.pt")
else:
    loaded_params = None

###################################### TRAIN THE MODEL
print(f'Starting training on {device}...')

# Load datasets
x_mesh_full, y_mesh_full, z_mesh_full = parse_csv_data(file_mesh, 'mesh', device)
x_cfd_full, y_cfd_full, z_cfd_full, p_cfd_full, u_cfd_full, v_cfd_full, w_cfd_full = parse_csv_data(file_cfd, 'cfd', device)

num_mesh = x_mesh_full.shape[0]
num_cfd = x_cfd_full.shape[0]

# Start timer
start_time = time.time()  # Record start time

best_val_loss = float('inf')
patience_counter = 0

# Initialize lists to store loss values for plotting
train_losses = []
physics_losses = []
data_losses = []
val_losses = []

for epoch in tqdm(range(1, epochs + 1), desc="Training Progress"):
    model.train()
    
    # Shuffle indices for batching
    mesh_perm = torch.randperm(num_mesh)
    cfd_perm = torch.randperm(num_cfd)
    
    # Create batches by slicing indices
    mesh_batches = [mesh_perm[i:i+batch_mesh] for i in range(0, num_mesh, batch_mesh)]
    cfd_batches = [cfd_perm[i:i+batch_cfd] for i in range(0, num_cfd, batch_cfd)]
    
    # Initialize accumulators for losses
    total_loss = 0.0
    total_physics_loss = 0.0
    total_data_loss = 0.0
    total_batches = 0

    for mesh_batch in mesh_batches:
        for cfd_batch in cfd_batches:
            # Get batch data
            xm = x_mesh_full[mesh_batch]
            ym = y_mesh_full[mesh_batch]
            zm = z_mesh_full[mesh_batch]

            xd = x_cfd_full[cfd_batch]
            yd = y_cfd_full[cfd_batch]
            zd = z_cfd_full[cfd_batch]
            pd = p_cfd_full[cfd_batch]
            ud = u_cfd_full[cfd_batch]
            vd = v_cfd_full[cfd_batch]
            wd = w_cfd_full[cfd_batch]

            # Ensure requires_grad=True for xm, ym, zm
            xm = xm.clone().detach().requires_grad_(True)
            ym = ym.clone().detach().requires_grad_(True)
            zm = zm.clone().detach().requires_grad_(True)

            # Zero gradients
            optimizer.zero_grad()

            # Compute losses
            l_phys = loss_physics(model, xm, ym, zm)
            l_data = loss_data(model, xd, yd, zd, pd, ud, vd, wd)

            # Total loss with fixed weighting
            l_total = l_phys + lambda_data * l_data

            # Backward pass and optimization
            l_total.backward()
            optimizer.step()

            # Accumulate losses
            total_loss += l_total.item()
            total_physics_loss += l_phys.item()
            total_data_loss += l_data.item()
            total_batches += 1

    # Scheduler step
    scheduler.step()

    # Compute average losses
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_physics_loss = total_physics_loss / total_batches if total_batches > 0 else 0
    avg_data_loss = total_data_loss / total_batches if total_batches > 0 else 0

    # Since we're not splitting the data, use avg_data_loss as validation loss
    avg_val_loss = avg_data_loss

    # Store loss values for plotting
    train_losses.append(avg_loss)
    physics_losses.append(avg_physics_loss)
    data_losses.append(avg_data_loss)
    val_losses.append(avg_val_loss)

    # Check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save the best model
        save_model(f'{save_fn}_best.pt', model, optimizer, scheduler, training_params)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Print progress
    if epoch % print_every == 0 or epoch == 1:
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Avg Train Loss : {avg_loss:.6f}")
        print(f"    Physics Loss : {avg_physics_loss:.6f}")
        print(f"    Data Loss    : {avg_data_loss:.6f}")
        # print(f"  Lambda Physics : {lambda_physics:.6f}")
        print(f"  Lambda Data    : {lambda_data:.6f}")
        print(f"  Val Loss       : {avg_val_loss:.6f}")

    # Save model periodically
    if epoch % save_every == 0:
        save_model(f'{save_fn}_epoch_{epoch}.pt', model, optimizer, scheduler, training_params)
        print(f"Model saved to {save_fn}_epoch_{epoch}.pt")

# End timer
end_time = time.time()  # Record end time

# Calculate and display total training time
total_training_time = end_time - start_time
print(f"\nTotal training time: {total_training_time:.2f} seconds")

print('Training phase complete.')

###################################### SAVE THE TRAINED MODEL
if save_end:
    save_model(f'{save_fn}.pt', model, optimizer, scheduler, training_params)
    print(f"Model saved to {save_fn}.pt")

###################################### EVALUATE THE MODEL AND PLOT PRESSURE FIELD COMPARISON

from sklearn.metrics import mean_squared_error, r2_score

# Load the best saved model
model, _, _, _ = load_model(f'{save_fn}_best.pt', model)
model.eval()

# Evaluate the model on the entire CFD data
with torch.no_grad():
    qq_pred = model(x_cfd_full, y_cfd_full, z_cfd_full)
    p_pred = qq_pred[:, 0].reshape(-1, 1)

# Move data to CPU and convert to numpy
x_val_cpu = x_cfd_full.cpu().numpy().flatten()
y_val_cpu = y_cfd_full.cpu().numpy().flatten()
z_val_cpu = z_cfd_full.cpu().numpy().flatten()
p_val_cpu = p_cfd_full.cpu().numpy().flatten()
p_pred_cpu = p_pred.cpu().numpy().flatten()

# Compute evaluation metrics for pressure
mse_p = mean_squared_error(p_val_cpu, p_pred_cpu)
r2_p = r2_score(p_val_cpu, p_pred_cpu)

print(f"\nMean Squared Error on Validation Set (Pressure): {mse_p:.6f}")
print(f"R^2 Score on Validation Set (Pressure): {r2_p:.6f}")

# Prepare data for full simulation visualization

# CFD Data
x_cfd_val = x_val_cpu
y_cfd_val = y_val_cpu
z_cfd_val = z_val_cpu
p_cfd_val = p_val_cpu

# PINN Predictions
x_pinn_val = x_val_cpu
y_pinn_val = y_val_cpu
z_pinn_val = z_val_cpu
p_pinn_val = p_pred_cpu

# Plotting
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), constrained_layout=True)

# CFD Pressure Field in x-y plane
sc0 = ax[0, 0].scatter(x_cfd_val, y_cfd_val, c=-p_cfd_val, cmap='RdBu', rasterized=True, marker='o')
ax[0, 0].set(aspect='equal', xlabel='x', ylabel='y', title='CFD Pressure Field (x-y Plane)')
fig.colorbar(sc0, ax=ax[0, 0], fraction=0.046, pad=0.04)

# PINN Pressure Field in x-y plane
sc1 = ax[1, 0].scatter(x_pinn_val, y_pinn_val, c=-p_pinn_val, cmap='RdBu', rasterized=True, marker='o')
ax[1, 0].set(aspect='equal', xlabel='x', ylabel='y', title='PINN Predicted Pressure Field (x-y Plane)')
fig.colorbar(sc1, ax=ax[1, 0], fraction=0.046, pad=0.04)

# CFD Pressure Field in x-z plane
sc2 = ax[0, 1].scatter(x_cfd_val, z_cfd_val, c=-p_cfd_val, cmap='RdBu', rasterized=True, marker='o')
ax[0, 1].set(aspect='equal', xlabel='x', ylabel='z', title='CFD Pressure Field (x-z Plane)')
fig.colorbar(sc2, ax=ax[0, 1], fraction=0.046, pad=0.04)

# PINN Pressure Field in x-z plane
sc3 = ax[1, 1].scatter(x_pinn_val, z_pinn_val, c=-p_pinn_val, cmap='RdBu', rasterized=True, marker='o')
ax[1, 1].set(aspect='equal', xlabel='x', ylabel='z', title='PINN Predicted Pressure Field (x-z Plane)')
fig.colorbar(sc3, ax=ax[1, 1], fraction=0.046, pad=0.04)

# Set the case number for the title and filename
sc = 1  # Case 1

fig.suptitle(f'Pressure Field Comparison: Case {sc}', fontsize=16)
plt.savefig(f'p_case_{sc}.png')
plt.show()
