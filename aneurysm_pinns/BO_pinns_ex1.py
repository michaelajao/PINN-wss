# -*- coding: utf-8 -*-
"""
Enhanced PINN Implementation for Aneurysmal Flow Dynamics

This script trains a Physics-Informed Neural Network (PINN) to solve the Navier-Stokes equations
for aneurysmal flow dynamics. It handles the absence of boundary data by enforcing boundary conditions
through network architecture modifications, measures training time, evaluates the model, saves
training parameters, and plots loss and prediction comparisons with enhanced visualizations.

Author: Your Name
Date: YYYY-MM-DD
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm.auto import tqdm  # For progress bars
import time  # For tracking training time
import matplotlib.pyplot as plt  # For plotting
import os  # For file path operations
import json  # For saving parameters
from sklearn.metrics import mean_squared_error, r2_score  # For performance metrics

###################################### Set Random Seeds for Reproducibility
def set_seed(seed=42):
    """
    Sets the random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Ensures that CUDA convolution operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # You can change the seed value as needed

###################################### Helper functions for loading and saving trained models
def load_model(path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Loads the model state, optimizer state, scheduler state, and training parameters from a checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{path}' not found.")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    training_params = checkpoint.get('training_params', None)  # Retrieve training parameters if available
    return model, optimizer, scheduler, training_params

def save_model(path, model, optimizer=None, scheduler=None, training_params=None):
    """
    Saves the model state, optimizer state, scheduler state, and training parameters to a checkpoint.
    """
    data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'training_params': training_params  # Include training parameters
    }
    torch.save(data, path)

###################################### Helper functions for retrieving CSV data (mesh, CFD results)
def parse_csv_data(path, category, device):
    """
    Parses CSV data for 'mesh' or 'cfd' categories and returns a TensorDataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file '{path}' not found.")
    
    if category == 'mesh':
        data = np.loadtxt(path, delimiter=',', unpack=True)
        if data.shape[0] != 3:
            raise ValueError(f"Expected 3 columns for 'mesh', got {data.shape[0]} columns.")
        x, y, z = data
        x = torch.tensor(x, dtype=torch.float32, device=device).reshape(-1,1)  # requires_grad=False
        y = torch.tensor(y, dtype=torch.float32, device=device).reshape(-1,1)  # requires_grad=False
        z = torch.tensor(z, dtype=torch.float32, device=device).reshape(-1,1)  # requires_grad=False
        dataset = TensorDataset(x, y, z)
        return dataset
    
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
        dataset = TensorDataset(x, y, z, p, u, v, w)
        return dataset
    
    else:
        raise ValueError(f"Unknown category: {category}")

###################################### The neural network with boundary condition enforcement
class NSNeuralNet(nn.Module):
    def __init__(self, input_size=3, output_size=4, num_layers=6, units_per_layer=128, activation='tanh'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.units_per_layer = units_per_layer
        self.activation = self.get_activation_function(activation)
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, units_per_layer))
        layers.append(self.activation)
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(units_per_layer, units_per_layer))
            layers.append(self.activation)
        # Output layer
        layers.append(nn.Linear(units_per_layer, output_size))
        self.main = nn.Sequential(*layers)
    
    def forward(self, x, y, z):
        # Combine inputs
        inputs = torch.cat([x, y, z], axis=1)
        # Normalize inputs
        inputs = 2.0 * (inputs - self.input_min) / (self.input_max - self.input_min) - 1.0
        # Forward pass
        outputs = self.main(inputs)
        # Enforce boundary conditions via transformation
        p = outputs[:, 0:1]
        u = outputs[:, 1:2]
        v = outputs[:, 2:3]
        w = outputs[:, 3:4]
        # Apply transformation to enforce zero velocity at boundaries (no-slip condition)
        # Using (1 - x^2)(1 - y^2)(1 - z^2) ensures velocities are zero at the boundaries
        u = (1 - x**2) * (1 - y**2) * (1 - z**2) * u
        v = (1 - x**2) * (1 - y**2) * (1 - z**2) * v
        w = (1 - x**2) * (1 - y**2) * (1 - z**2) * w
        return torch.cat([p, u, v, w], axis=1)
    
    def get_activation_function(self, activation):
        """
        Returns the activation function based on the provided name.
        """
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
    
    def set_normalization(self, input_min, input_max):
        """
        Sets the normalization parameters for input scaling.
        """
        self.input_min = input_min
        self.input_max = input_max

###################################### Define the loss functions
loss_func = nn.MSELoss()

def loss_physics(net, x, y, z):
    """
    Computes the physics-informed loss by enforcing the Navier-Stokes equations.
    """
    # Ensure inputs require gradients for derivative computations
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)
    
    # Physics-informed loss enforcing Navier-Stokes equations
    qq = net(x, y, z)
    p = qq[:,0].reshape(-1,1)
    u = qq[:,1].reshape(-1,1)
    v = qq[:,2].reshape(-1,1)
    w = qq[:,3].reshape(-1,1)
    del qq

    # First-order derivatives
    ones_p = torch.ones_like(p)
    p_x = torch.autograd.grad(p, x, grad_outputs=ones_p, create_graph=True, retain_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=ones_p, create_graph=True, retain_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=ones_p, create_graph=True, retain_graph=True)[0]

    ones_u = torch.ones_like(u)
    u_x = torch.autograd.grad(u, x, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]

    w_x = torch.autograd.grad(w, x, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]

    # Second-order derivatives
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
    eqn_x = p_x + rho * (u * u_x + v * u_y + w * u_z) - mu * (u_xx + u_yy + u_zz)
    eqn_y = p_y + rho * (u * v_x + v * v_y + w * v_z) - mu * (v_xx + v_yy + v_zz)
    eqn_z = p_z + rho * (u * w_x + v * w_y + w * w_z) - mu * (w_xx + w_yy + w_zz)
    eqn_c = u_x + v_y + w_z  # Continuity equation

    # Concatenate all residuals
    dt = torch.cat([eqn_x, eqn_y, eqn_z, eqn_c], axis=1)
    return loss_func(dt, torch.zeros_like(dt))

def loss_data(net, x, y, z, p, u, v, w):
    """
    Computes the data loss by comparing network predictions with actual CFD data.
    """
    predictions = net(x, y, z)
    targets = torch.cat([p, u, v, w], dim=1)
    return loss_func(predictions, targets)

###################################### CONFIGURABLE PARAMETERS
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

input_size = 3  # x, y, z
output_size = 4  # p, u, v, w
num_layers = 12   # Number of layers in the neural network
units_per_layer = 256  # Number of units per layer
activation_function = 'silu'  # Activation function: 'relu', 'tanh', 'silu', 'elu', 'leaky_relu'

epochs = 1000

load_previous = False  # Whether to load a previously saved model
save_end = True        # Whether to save the model at the end of training
load_fn = 'ns_improved'  # Filename to load the model from
save_fn = 'ns_improved'  # Filename to save the model to

print_every = 50  # Print training progress every 'print_every' epochs
save_every = 100  # Save the model every 'save_every' epochs

learning_rate = 1e-4  # Reduced learning rate
lr_steps = 200
lr_gamma = 0.5

# Dynamic loss weights
lambda_phys_initial = 1.0
lambda_phys_final = 1.0
lambda_data_initial = 0.1  # Start with lower data loss weight
lambda_data_final = 1.0    # Gradually increase data loss weight

file_mesh = 'mesh_1_s.csv'  # Path to mesh CSV file
file_cfd = 'train_1.csv'    # Path to CFD CSV file

batch_mesh = 500  # Reduced batch size for mesh data
batch_cfd = 250    # Reduced batch size for CFD data

patience = 50  # Increased early stopping patience

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
    'lambda_phys_initial': lambda_phys_initial,
    'lambda_phys_final': lambda_phys_final,
    'lambda_data_initial': lambda_data_initial,
    'lambda_data_final': lambda_data_final,
    'file_mesh': file_mesh,
    'file_cfd': file_cfd,
    'batch_mesh': batch_mesh,
    'batch_cfd': batch_cfd,
    'patience': patience,
    'seed': 42,
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
    """
    Initializes weights using Xavier (Glorot) Normal initialization.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(init_weights)

###################################### Initialize Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight_decay for L2 regularization
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)  # Changed scheduler

# Load previous model if specified
if load_previous:
    model, optimizer, scheduler, loaded_params = load_model(f'{load_fn}.pt', model, optimizer, scheduler, device=device)
    if loaded_params is not None:
        print(f"Loaded training parameters from {load_fn}.pt")
else:
    loaded_params = None

###################################### TRAIN THE MODEL
print(f'Starting training on {device}...')

# Load datasets
mesh_dataset = parse_csv_data(file_mesh, 'mesh', device)
cfd_dataset = parse_csv_data(file_cfd, 'cfd', device)

# Normalize inputs and outputs
x_all = torch.cat([mesh_dataset[:][0], cfd_dataset[:][0]], dim=0)
y_all = torch.cat([mesh_dataset[:][1], cfd_dataset[:][1]], dim=0)
z_all = torch.cat([mesh_dataset[:][2], cfd_dataset[:][2]], dim=0)

input_min = torch.min(torch.cat([x_all, y_all, z_all], dim=1), dim=0)[0].to(device)
input_max = torch.max(torch.cat([x_all, y_all, z_all], dim=1), dim=0)[0].to(device)

model.set_normalization(input_min, input_max)

# Normalize CFD outputs
p_all = cfd_dataset[:][3]
u_all = cfd_dataset[:][4]
v_all = cfd_dataset[:][5]
w_all = cfd_dataset[:][6]

# Compute min and max for each variable as scalars
p_min = p_all.min().item()
p_max = p_all.max().item()
u_min = u_all.min().item()
u_max = u_all.max().item()
v_min = v_all.min().item()
v_max = v_all.max().item()
w_min = w_all.min().item()
w_max = w_all.max().item()

def normalize_output(tensor, min_value, max_value):
    """
    Normalizes the tensor using the provided min and max values.
    """
    return 2.0 * (tensor - min_value) / (max_value - min_value) - 1.0

def denormalize_output(tensor, min_value, max_value):
    """
    Denormalizes the tensor using the provided min and max values.
    """
    return 0.5 * (tensor + 1.0) * (max_value - min_value) + min_value

# Apply normalization to CFD outputs
normalized_cfd_dataset = TensorDataset(
    cfd_dataset[:][0],  # x
    cfd_dataset[:][1],  # y
    cfd_dataset[:][2],  # z
    normalize_output(cfd_dataset[:][3], p_min, p_max),  # p
    normalize_output(cfd_dataset[:][4], u_min, u_max),  # u
    normalize_output(cfd_dataset[:][5], v_min, v_max),  # v
    normalize_output(cfd_dataset[:][6], w_min, w_max)   # w
)

# Verify shapes after normalization
print(f"Normalized p shape: {normalized_cfd_dataset[:][3].shape}")  # Should be [N, 1]
print(f"Normalized u shape: {normalized_cfd_dataset[:][4].shape}")  # Should be [N, 1]
print(f"Normalized v shape: {normalized_cfd_dataset[:][5].shape}")  # Should be [N, 1]
print(f"Normalized w shape: {normalized_cfd_dataset[:][6].shape}")  # Should be [N, 1]

# Split CFD data into training and validation sets
train_size = int(0.8 * len(normalized_cfd_dataset))
val_size = len(normalized_cfd_dataset) - train_size
cfd_train_dataset, cfd_val_dataset = random_split(normalized_cfd_dataset, [train_size, val_size])

# Create DataLoaders with num_workers=0 to avoid multiprocessing issues
mesh_loader = DataLoader(mesh_dataset, batch_size=batch_mesh, shuffle=True, num_workers=0)
cfd_train_loader = DataLoader(cfd_train_dataset, batch_size=batch_cfd, shuffle=True, num_workers=0)
cfd_val_loader = DataLoader(cfd_val_dataset, batch_size=batch_cfd, shuffle=False, num_workers=0)

best_val_loss = float('inf')
patience_counter = 0

# Initialize lists to store loss values for plotting
train_losses = []
physics_losses = []
data_losses = []
val_losses = []

# Function to adjust lambda values adaptively
def adjust_lambda(epoch, total_epochs, initial_value, final_value):
    """
    Linearly adjusts the lambda value from initial to final over the total epochs.
    """
    return initial_value + (final_value - initial_value) * (epoch / total_epochs)

# Start timer
start_time = time.time()  # Record start time

for epoch in range(1, epochs + 1):
    model.train()
    # Adjust dynamic weights
    lambda_phys = adjust_lambda(epoch, epochs, lambda_phys_initial, lambda_phys_final)
    lambda_data = adjust_lambda(epoch, epochs, lambda_data_initial, lambda_data_final)

    # Initialize accumulators for losses
    total_loss = 0.0
    total_physics_loss = 0.0
    total_data_loss = 0.0
    total_batches = 0

    # Create iterator for mesh data
    mesh_iter = iter(mesh_loader)

    # Loop over CFD training data
    for xd, yd, zd, pd, ud, vd, wd in tqdm(cfd_train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
        # Get next batch from mesh_loader
        try:
            xm, ym, zm = next(mesh_iter)
        except StopIteration:
            mesh_iter = iter(mesh_loader)
            xm, ym, zm = next(mesh_iter)

        # No need to move data to device again since parse_csv_data already did it

        # Zero gradients
        optimizer.zero_grad()

        # Compute losses
        l_phys = loss_physics(model, xm, ym, zm)
        l_data = loss_data(model, xd, yd, zd, pd, ud, vd, wd)

        # Total loss with dynamic weighting
        l_total = lambda_phys * l_phys + lambda_data * l_data

        # Backward pass and optimization
        l_total.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += l_total.item()
        total_physics_loss += l_phys.item()
        total_data_loss += l_data.item()
        total_batches += 1

    # Scheduler step based on validation loss
    scheduler.step(best_val_loss)

    # Compute average losses
    avg_loss = total_loss / total_batches
    avg_physics_loss = total_physics_loss / total_batches
    avg_data_loss = total_data_loss / total_batches

    # Validation
    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for xd_val, yd_val, zd_val, pd_val, ud_val, vd_val, wd_val in cfd_val_loader:
            l_val = loss_data(model, xd_val, yd_val, zd_val, pd_val, ud_val, vd_val, wd_val)
            val_loss += l_val.item()
            val_batches += 1
    avg_val_loss = val_loss / val_batches

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
        print(f"  Lambda Physics : {lambda_phys:.6f}")
        print(f"  Lambda Data    : {lambda_data:.6f}")
        print(f"  Avg Val Loss   : {avg_val_loss:.6f}")

        # Optional: Print memory usage
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1e6  # in MB
            reserved = torch.cuda.memory_reserved(device) / 1e6  # in MB
            print(f"  Memory Allocated: {allocated:.2f} MB; Memory Reserved: {reserved:.2f} MB")

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
    print(f'Model saved to {save_fn}.pt')

###################################### PLOT TRAINING AND VALIDATION LOSS CURVES
# Plot individual loss components
plt.figure(figsize=(10, 6))
epochs_range = range(1, len(train_losses) + 1)

plt.plot(epochs_range, train_losses, label='Total Training Loss')
plt.plot(epochs_range, physics_losses, label='Physics Loss')
plt.plot(epochs_range, data_losses, label='Data Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{save_fn}_loss_curves.png', dpi=300)
plt.show()

# Additionally, plot each loss component separately
loss_components = {
    'Total Training Loss': train_losses,
    'Physics Loss': physics_losses,
    'Data Loss': data_losses,
    'Validation Loss': val_losses
}

for name, losses in loss_components.items():
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, losses, label=name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{name} Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f'{save_fn}_{name.replace(" ", "_").lower()}.png'
    plt.savefig(filename, dpi=300)
    plt.show()

###################################### EVALUATE THE MODEL AND PLOT PREDICTIONS

# Load the best saved model
model, _, _, _ = load_model(f'{save_fn}_best.pt', model)
model.eval()

# Prepare data for full simulation visualization
cfd_full_dataset = parse_csv_data(file_cfd, 'cfd', device)
x_cfd_full = cfd_full_dataset[:][0]
y_cfd_full = cfd_full_dataset[:][1]
z_cfd_full = cfd_full_dataset[:][2]
p_cfd_full = cfd_full_dataset[:][3]
u_cfd_full = cfd_full_dataset[:][4]
v_cfd_full = cfd_full_dataset[:][5]
w_cfd_full = cfd_full_dataset[:][6]

# Evaluate the model on the entire CFD data
with torch.no_grad():
    p_pred, u_pred, v_pred, w_pred = [], [], [], []
    batch_size_eval = 10000  # Adjust based on your GPU memory
    for i in range(0, len(x_cfd_full), batch_size_eval):
        x_batch = x_cfd_full[i:i+batch_size_eval]
        y_batch = y_cfd_full[i:i+batch_size_eval]
        z_batch = z_cfd_full[i:i+batch_size_eval]
        outputs = model(x_batch, y_batch, z_batch)
        outputs = outputs.cpu()
        # Denormalize each output variable separately
        p_pred_batch = denormalize_output(outputs[:, 0:1], p_min, p_max)
        u_pred_batch = denormalize_output(outputs[:, 1:2], u_min, u_max)
        v_pred_batch = denormalize_output(outputs[:, 2:3], v_min, v_max)
        w_pred_batch = denormalize_output(outputs[:, 3:4], w_min, w_max)
        p_pred.append(p_pred_batch)
        u_pred.append(u_pred_batch)
        v_pred.append(v_pred_batch)
        w_pred.append(w_pred_batch)
    p_pred = torch.cat(p_pred).numpy()
    u_pred = torch.cat(u_pred).numpy()
    v_pred = torch.cat(v_pred).numpy()
    w_pred = torch.cat(w_pred).numpy()

# Convert tensors to numpy arrays for plotting
x_val = x_cfd_full.cpu().numpy().flatten()
y_val = y_cfd_full.cpu().numpy().flatten()
z_val = z_cfd_full.cpu().numpy().flatten()
p_actual = p_cfd_full.cpu().numpy().flatten()
u_actual = u_cfd_full.cpu().numpy().flatten()
v_actual = v_cfd_full.cpu().numpy().flatten()
w_actual = w_cfd_full.cpu().numpy().flatten()

###################################### COMPUTE PERFORMANCE METRICS

def compute_metrics(actual, predicted, variable_name):
    """
    Computes and prints MSE and R² Score for a given variable.
    """
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    print(f"{variable_name} Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  R² Score: {r2:.6f}")
    print()

# Pressure Metrics
compute_metrics(p_actual, p_pred, "Pressure")

# Velocity Components Metrics
compute_metrics(u_actual, u_pred, "Velocity U-component")
compute_metrics(v_actual, v_pred, "Velocity V-component")
compute_metrics(w_actual, w_pred, "Velocity W-component")

###################################### IMPROVED VISUALIZATION

# High-Resolution Pressure Field Comparison
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
cmap = 'viridis'

# Actual Pressure Field using contourf for smoothness
contour0 = ax[0].tricontourf(x_val, y_val, p_actual, levels=100, cmap=cmap)
ax[0].set_title('Actual Pressure Field')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
fig.colorbar(contour0, ax=ax[0], label='Pressure [Units]')

# Predicted Pressure Field using contourf for smoothness
contour1 = ax[1].tricontourf(x_val, y_val, p_pred, levels=100, cmap=cmap)
ax[1].set_title('Predicted Pressure Field')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
fig.colorbar(contour1, ax=ax[1], label='Pressure [Units]')

plt.tight_layout()
plt.savefig(f'{save_fn}_pressure_field_comparison.png', dpi=300)
plt.show()

# High-Resolution Velocity Magnitude Comparison
velocity_actual = np.sqrt(u_actual**2 + v_actual**2 + w_actual**2)
velocity_pred = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
cmap = 'plasma'

# Actual Velocity Magnitude using contourf for smoothness
contour2 = ax[0].tricontourf(x_val, y_val, velocity_actual, levels=100, cmap=cmap)
ax[0].set_title('Actual Velocity Magnitude')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
fig.colorbar(contour2, ax=ax[0], label='Velocity [Units]')

# Predicted Velocity Magnitude using contourf for smoothness
contour3 = ax[1].tricontourf(x_val, y_val, velocity_pred, levels=100, cmap=cmap)
ax[1].set_title('Predicted Velocity Magnitude')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
fig.colorbar(contour3, ax=ax[1], label='Velocity [Units]')

plt.tight_layout()
plt.savefig(f'{save_fn}_velocity_field_comparison.png', dpi=300)
plt.show()

# Additional Smoothness with Grid Interpolation (Optional)
from scipy.interpolate import griddata

# Define grid.
grid_x, grid_y = np.mgrid[min(x_val):max(x_val):500j, min(y_val):max(y_val):500j]

# Interpolate actual pressure
grid_p_actual = griddata((x_val, y_val), p_actual, (grid_x, grid_y), method='cubic')
# Interpolate predicted pressure
grid_p_pred = griddata((x_val, y_val), p_pred, (grid_x, grid_y), method='cubic')

# Plot interpolated pressure fields
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
cmap = 'viridis'

# Actual Pressure Field
c0 = ax[0].imshow(grid_p_actual.T, extent=(min(x_val), max(x_val), min(y_val), max(y_val)),
               origin='lower', cmap=cmap, aspect='auto')
ax[0].set_title('Actual Pressure Field (Interpolated)')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
fig.colorbar(c0, ax=ax[0], label='Pressure [Units]')

# Predicted Pressure Field
c1 = ax[1].imshow(grid_p_pred.T, extent=(min(x_val), max(x_val), min(y_val), max(y_val)),
               origin='lower', cmap=cmap, aspect='auto')
ax[1].set_title('Predicted Pressure Field (Interpolated)')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
fig.colorbar(c1, ax=ax[1], label='Pressure [Units]')

plt.tight_layout()
plt.savefig(f'{save_fn}_pressure_field_interpolated.png', dpi=300)
plt.show()

# Similarly, you can create interpolated plots for velocity magnitude if needed.

