# -*- coding: utf-8 -*-
"""
Enhanced PINN Implementation with Adaptive Loss Weights, Boundary Conditions,
Learnable Physical Parameters (rho and mu), Data Normalization,
Additional Evaluation Metrics, and Improved Initialization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time  # For tracking training time
import matplotlib.pyplot as plt  # For plotting
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # For progress bars
import os  # For file path operations
import json  # For saving parameters
import random  # For setting seeds
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

###################################### Set Random Seeds for Reproducibility
def set_seed(seed=42):
    """
    Sets the random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensures that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # You can change the seed value as needed

###################################### Helper functions for loading and saving trained models
def load_model(path, model, optimizer=None, scheduler=None):
    """
    Loads a saved model checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{path}' not found.")
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    training_params = checkpoint.get("training_params", None)  # Retrieve training parameters if available
    return model, optimizer, scheduler, training_params

def save_model(path, model, optimizer=None, scheduler=None, training_params=None):
    """
    Saves the model checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
    data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (optimizer.state_dict() if optimizer is not None else None),
        "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
        "training_params": training_params,  # Include training parameters
    }
    torch.save(data, path)

###################################### Helper functions for retrieving CSV data (mesh, CFD results)
def parse_csv_data(path, category):
    """
    Parses CSV data for mesh or CFD results.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file '{path}' not found.")

    data = np.loadtxt(path, delimiter=",", unpack=True)

    if category == "mesh":
        if data.shape[0] != 3:
            raise ValueError(f"Expected 3 columns for 'mesh', got {data.shape[0]} columns.")
        x, y, z = data
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
        y = torch.tensor(y, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
        z = torch.tensor(z, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
        return x, y, z

    elif category == "cfd":
        if data.shape[0] != 7:
            raise ValueError(f"Expected 7 columns for 'cfd', got {data.shape[0]} columns.")
        x, y, z, p, u, v, w = data
        x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        z = torch.tensor(z, dtype=torch.float32).reshape(-1, 1)
        p = torch.tensor(p, dtype=torch.float32).reshape(-1, 1)
        u = torch.tensor(u, dtype=torch.float32).reshape(-1, 1)
        v = torch.tensor(v, dtype=torch.float32).reshape(-1, 1)
        w = torch.tensor(w, dtype=torch.float32).reshape(-1, 1)
        return x, y, z, p, u, v, w

    else:
        raise ValueError(f"Unknown category: {category}")

###################################### Data Normalization
def normalize_data(x, y, z, p=None, u=None, v=None, w=None):
    """
    Normalizes input and output data.
    Returns normalized tensors and normalization parameters.
    """
    # Normalize inputs to [-1, 1]
    x_mean = x.mean()
    x_std = x.std()
    y_mean = y.mean()
    y_std = y.std()
    z_mean = z.mean()
    z_std = z.std()

    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    z_norm = (z - z_mean) / z_std

    norm_params = {
        "x_mean": x_mean.item(),
        "x_std": x_std.item(),
        "y_mean": y_mean.item(),
        "y_std": y_std.item(),
        "z_mean": z_mean.item(),
        "z_std": z_std.item(),
    }

    if p is not None and u is not None and v is not None and w is not None:
        # Normalize outputs to zero mean and unit variance
        p_mean = p.mean()
        p_std = p.std()
        u_mean = u.mean()
        u_std = u.std()
        v_mean = v.mean()
        v_std = v.std()
        w_mean = w.mean()
        w_std = w.std()

        p_norm = (p - p_mean) / p_std
        u_norm = (u - u_mean) / u_std
        v_norm = (v - v_mean) / v_std
        w_norm = (w - w_mean) / w_std

        norm_params.update({
            "p_mean": p_mean.item(),
            "p_std": p_std.item(),
            "u_mean": u_mean.item(),
            "u_std": u_std.item(),
            "v_mean": v_mean.item(),
            "v_std": v_std.item(),
            "w_mean": w_mean.item(),
            "w_std": w_std.item(),
        })

        return x_norm, y_norm, z_norm, p_norm, u_norm, v_norm, w_norm, norm_params
    else:
        return x_norm, y_norm, z_norm, norm_params

def denormalize_output(p_norm, u_norm, v_norm, w_norm, norm_params):
    """
    Denormalizes the output data.
    """
    p = p_norm * norm_params["p_std"] + norm_params["p_mean"]
    u = u_norm * norm_params["u_std"] + norm_params["u_mean"]
    v = v_norm * norm_params["v_std"] + norm_params["v_mean"]
    w = w_norm * norm_params["w_std"] + norm_params["w_mean"]
    return p, u, v, w

###################################### Neural Network Definition with Learnable rho and mu
class FullyConnectedNeuralNet(nn.Module):
    def __init__(
        self,
        input_size=3,
        output_size=4,
        num_layers=12,
        units_per_layer=256,
        activation="tanh",
    ):
        super(FullyConnectedNeuralNet, self).__init__()
        self.num_layers = num_layers
        self.units_per_layer = units_per_layer
        self.activation = self.get_activation_function(activation)

        layers = []
        layers.append(nn.Linear(input_size, units_per_layer))
        layers.append(self.activation)

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(units_per_layer, units_per_layer))
            layers.append(self.activation)

        layers.append(nn.Linear(units_per_layer, output_size))

        self.network = nn.Sequential(*layers)

        # Define rho and mu as trainable parameters
        self.rho = nn.Parameter(torch.tensor(1050.0, requires_grad=True))  # Density
        self.mu = nn.Parameter(torch.tensor(0.0035, requires_grad=True))   # Viscosity

    def forward(self, x, y, z):
        out = torch.cat([x, y, z], dim=1)
        output = self.network(out)
        return output  # Outputs: p, u, v, w

    def get_activation_function(self, activation):
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "silu":
            return nn.SiLU()
        elif activation == "elu":
            return nn.ELU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

###################################### Define the loss functions
loss_func = nn.MSELoss()

def loss_physics(net, x, y, z):
    """
    Physics-informed loss enforcing Navier-Stokes equations with boundary conditions.
    """
    # Ensure inputs require gradients for derivative computations
    # (Already set in data loading)

    qq = net(x, y, z)
    p = qq[:, 0].reshape(-1, 1)
    u = qq[:, 1].reshape(-1, 1)
    v = qq[:, 2].reshape(-1, 1)
    w = qq[:, 3].reshape(-1, 1)
    del qq

    # Apply transformation to enforce zero velocity at boundaries (no-slip condition)
    # Using (1 - x^2)(1 - y^2)(1 - z^2) ensures velocities are zero at the boundaries
    T = (1 - x**2) * (1 - y**2) * (1 - z**2)
    u = T * u
    v = T * v
    w = T * w

    # First-order derivatives
    ones = torch.ones_like(p)
    p_x = torch.autograd.grad(p, x, grad_outputs=ones, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=ones, create_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=ones, create_graph=True)[0]

    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=ones, create_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=ones, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=ones, create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=ones, create_graph=True)[0]

    w_x = torch.autograd.grad(w, x, grad_outputs=ones, create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=ones, create_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=ones, create_graph=True)[0]

    # Second-order derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=ones, create_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=ones, create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=ones, create_graph=True)[0]

    w_xx = torch.autograd.grad(w_x, x, grad_outputs=ones, create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=ones, create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=ones, create_graph=True)[0]

    # Retrieve rho and mu from the network
    rho = net.rho
    mu = net.mu

    # Navier-Stokes equations (assuming incompressible flow)
    eqn_x = p_x + rho * (u * u_x + v * u_y + w * u_z) - mu * (u_xx + u_yy + u_zz)
    eqn_y = p_y + rho * (u * v_x + v * v_y + w * v_z) - mu * (v_xx + v_yy + v_zz)
    eqn_z = p_z + rho * (u * w_x + v * w_y + w * w_z) - mu * (w_xx + w_yy + w_zz)
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
print(f"Using device: {device}")

input_size = 3  # x, y, z
output_size = 4  # p, u, v, w
num_layers = 12  # Increased number of layers
units_per_layer = 256  # Increased number of neurons per layer
activation_function = "relu"  # Changed to 'tanh' other options: 'relu', 'silu', 'elu', 'leaky_relu'

epochs = 500  # Increased number of epochs
load_previous = False
save_end = True
load_fn = "ns_34_3"
save_fn = "ns_34_3"

print_every = 10  # Adjusted frequency
save_every = 100  # Adjusted frequency

learning_rate = 1e-4  # Reduced learning rate for stability
weight_decay = 1e-5  # Added weight decay for regularization
lr_patience = 20  # Patience for ReduceLROnPlateau
lr_factor = 0.5  # Factor to reduce LR
min_lr = 1e-6  # Minimum learning rate

file_mesh = "./data/mesh_1_s.csv"
file_cfd = "./data/train_1.csv"

batch_mesh = int(5000 / 5)  # Adjusted batch size for mesh data
batch_cfd = int(2500 / 5)  # Adjusted batch size for CFD data

# Dynamic loss weights
lambda_phys_initial = 1.0  # Start with higher physics loss weight
lambda_phys_final = 1.0
lambda_data_initial = 0.1  # Start with lower data loss weight
lambda_data_final = 1.0  # Gradually increase data loss weight

###################################### Collect and Save Training Parameters
training_params = {
    "device": str(device),
    "input_size": input_size,
    "output_size": output_size,
    "num_layers": num_layers,
    "units_per_layer": units_per_layer,
    "activation_function": activation_function,
    "epochs": epochs,
    "load_previous": load_previous,
    "save_end": save_end,
    "load_fn": load_fn,
    "save_fn": save_fn,
    "print_every": print_every,
    "save_every": save_every,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "lr_patience": lr_patience,
    "lr_factor": lr_factor,
    "min_lr": min_lr,
    "lambda_phys_initial": lambda_phys_initial,
    "lambda_phys_final": lambda_phys_final,
    "lambda_data_initial": lambda_data_initial,
    "lambda_data_final": lambda_data_final,
    "file_mesh": file_mesh,
    "file_cfd": file_cfd,
    "batch_mesh": batch_mesh,
    "batch_cfd": batch_cfd,
    "seed": 42,  # Seed for reproducibility
}

# Save the parameters to a JSON file
params_filename = f"models/{save_fn}_params.json"
with open(params_filename, "w") as f:
    json.dump(training_params, f, indent=4)

print(f"Training parameters saved to {params_filename}")

###################################### Initialize the Neural Network
model = FullyConnectedNeuralNet(
    input_size=input_size,
    output_size=output_size,
    num_layers=num_layers,
    units_per_layer=units_per_layer,
    activation=activation_function,
).to(device)

def init_weights(m):
    """
    Initializes weights using Xavier Normal initialization.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(activation_function))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(init_weights)

###################################### Initialize Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, verbose=True, min_lr=min_lr)

###################################### Load a Pretrained Model (if applicable)
if load_previous:
    try:
        model, optimizer, scheduler, loaded_params = load_model(f"models/{save_fn}.pt", model, optimizer, scheduler)
        if loaded_params is not None:
            print(f"Loaded training parameters from models/{save_fn}.pt")
    except FileNotFoundError:
        print(f"Pretrained model 'models/{save_fn}.pt' not found. Proceeding without loading.")

###################################### TRAIN THE MODEL
print(f"Starting the training phase on {device} ...")

# Load datasets
x_mesh_full, y_mesh_full, z_mesh_full = parse_csv_data(file_mesh, "mesh")
x_cfd_full, y_cfd_full, z_cfd_full, p_cfd_full, u_cfd_full, v_cfd_full, w_cfd_full = parse_csv_data(file_cfd, "cfd")

# Normalize data
x_mesh_norm, y_mesh_norm, z_mesh_norm, mesh_norm_params = normalize_data(x_mesh_full, y_mesh_full, z_mesh_full)
x_cfd_norm, y_cfd_norm, z_cfd_norm, p_cfd_norm, u_cfd_norm, v_cfd_norm, w_cfd_norm, cfd_norm_params = normalize_data(
    x_cfd_full, y_cfd_full, z_cfd_full, p_cfd_full, u_cfd_full, v_cfd_full, w_cfd_full)

# Create TensorDatasets
mesh_dataset = TensorDataset(x_mesh_norm, y_mesh_norm, z_mesh_norm)
cfd_dataset = TensorDataset(
    x_cfd_norm, y_cfd_norm, z_cfd_norm, p_cfd_norm, u_cfd_norm, v_cfd_norm, w_cfd_norm
)

# Create DataLoaders with increased batch sizes and num_workers=0 to avoid multiprocessing issues
mesh_loader = DataLoader(
    mesh_dataset, batch_size=batch_mesh, shuffle=True, num_workers=0, drop_last=True
)
cfd_loader = DataLoader(
    cfd_dataset, batch_size=batch_cfd, shuffle=True, num_workers=0, drop_last=True
)

# Start timer
start_time = time.time()  # Record start time

best_val_loss = float("inf")
patience_counter = 0
patience = 20  # For early stopping

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

# Training Loop with tqdm progress bars
for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
    model.train()

    # Adjust dynamic weights
    lambda_phys = adjust_lambda(epoch, epochs, lambda_phys_initial, lambda_phys_final)
    lambda_data = adjust_lambda(epoch, epochs, lambda_data_initial, lambda_data_final)

    total_loss = 0.0
    total_physics_loss = 0.0
    total_data_loss = 0.0
    total_batches = 0

    # Iterate over mesh_loader and cfd_loader simultaneously
    mesh_iterator = iter(mesh_loader)
    cfd_iterator = iter(cfd_loader)
    num_batches = min(len(mesh_loader), len(cfd_loader))

    for _ in tqdm(range(num_batches), desc=f"Epoch {epoch} Batches", leave=False):
        try:
            mesh_batch = next(mesh_iterator)
            cfd_batch = next(cfd_iterator)
        except StopIteration:
            break

        xm, ym, zm = [tensor.to(device).requires_grad_(True) for tensor in mesh_batch]
        xd, yd, zd, pd, ud, vd, wd = [tensor.to(device) for tensor in cfd_batch]

        optimizer.zero_grad()
        # Compute losses
        l_phys = loss_physics(model, xm, ym, zm)
        l_data = loss_data(model, xd, yd, zd, pd, ud, vd, wd)
        l_total = lambda_phys * l_phys + lambda_data * l_data

        # Backward and optimize
        l_total.backward(retain_graph=True)
        optimizer.step()

        # Accumulate losses
        total_loss += l_total.item()
        total_physics_loss += l_phys.item()
        total_data_loss += l_data.item()
        total_batches += 1

    # Scheduler step based on validation loss
    avg_val_loss = total_data_loss / total_batches if total_batches > 0 else 0
    scheduler.step(avg_val_loss)

    # Compute average losses
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_physics_loss = total_physics_loss / total_batches if total_batches > 0 else 0
    avg_data_loss = total_data_loss / total_batches if total_batches > 0 else 0

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
        save_model(
            f"best_models/{save_fn}_best.pt",
            model,
            optimizer,
            scheduler,
            training_params,
        )
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
        print(f"  Val Loss       : {avg_val_loss:.6f}")
        # Optional: Print memory usage
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(device) / 1e6  # in MB
            reserved = torch.cuda.memory_reserved(device) / 1e6  # in MB
            print(f"  Memory Allocated: {allocated:.2f} MB; Memory Reserved: {reserved:.2f} MB")

    # Save model periodically
    if epoch % save_every == 0:
        save_model(
            f"checkpoints/{save_fn}_epoch_{epoch}.pt",
            model,
            optimizer,
            scheduler,
            training_params,
        )
        print(f"Model saved to checkpoints/{save_fn}_epoch_{epoch}.pt")

print("Training phase complete.")

###################################### SAVE THE TRAINED MODEL
if save_end:
    save_model(f"models/{save_fn}.pt", model, optimizer, scheduler, training_params)
    print(f"Model saved to models/{save_fn}.pt")

###################################### EVALUATE THE MODEL AND PLOT PRESSURE & Velocity Field Comparison

# Load the best saved model
best_model_path = f"best_models/{save_fn}_best.pt"
try:
    model, _, _, _ = load_model(best_model_path, model)
    print(f"Loaded the best model from {best_model_path}")
except FileNotFoundError:
    print(f"Best model file '{best_model_path}' not found. Proceeding with current model.")

model.eval()

# Evaluate the model on the entire CFD data
with torch.no_grad():
    x_cfd_norm_gpu = x_cfd_norm.to(device)
    y_cfd_norm_gpu = y_cfd_norm.to(device)
    z_cfd_norm_gpu = z_cfd_norm.to(device)
    qq_pred_norm = model(x_cfd_norm_gpu, y_cfd_norm_gpu, z_cfd_norm_gpu)
    p_pred_norm = qq_pred_norm[:, 0].reshape(-1, 1)
    u_pred_norm = qq_pred_norm[:, 1].reshape(-1, 1)
    v_pred_norm = qq_pred_norm[:, 2].reshape(-1, 1)
    w_pred_norm = qq_pred_norm[:, 3].reshape(-1, 1)

# Denormalize predictions
p_pred, u_pred, v_pred, w_pred = denormalize_output(p_pred_norm, u_pred_norm, v_pred_norm, w_pred_norm, cfd_norm_params)

# Denormalize true values
p_true, u_true, v_true, w_true = denormalize_output(p_cfd_norm, u_cfd_norm, v_cfd_norm, w_cfd_norm, cfd_norm_params)

# Move data to CPU and convert to numpy
x_val_cpu = x_cfd_full.cpu().numpy().flatten()
y_val_cpu = y_cfd_full.cpu().numpy().flatten()
z_val_cpu = z_cfd_full.cpu().numpy().flatten()
p_val_cpu = p_true.cpu().numpy().flatten()
p_pred_cpu = p_pred.cpu().numpy().flatten()
u_val_cpu = u_true.cpu().numpy().flatten()
v_val_cpu = v_true.cpu().numpy().flatten()
w_val_cpu = w_true.cpu().numpy().flatten()
u_pred_cpu = u_pred.cpu().numpy().flatten()
v_pred_cpu = v_pred.cpu().numpy().flatten()
w_pred_cpu = w_pred.cpu().numpy().flatten()

# Compute evaluation metrics for pressure
mse_p = mean_squared_error(p_val_cpu, p_pred_cpu)
rmse_p = np.sqrt(mse_p)
nrmse_p = rmse_p / (np.max(p_val_cpu) - np.min(p_val_cpu))
mae_p = mean_absolute_error(p_val_cpu, p_pred_cpu)
r2_p = r2_score(p_val_cpu, p_pred_cpu)
rmae_p = mae_p / np.mean(np.abs(p_val_cpu))

# Compute evaluation metrics for velocity components
def compute_metrics(true, pred, component_name=""):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(true) - np.min(true))
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    rmae = mae / np.mean(np.abs(true))
    print(f"\n{component_name} Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"  Normalized RMSE (NRMSE): {nrmse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  Relative MAE (RMAE): {rmae:.6f}")
    print(f"  R^2 Score: {r2:.6f}")

print(f"\nPressure Metrics:")
print(f"  Mean Squared Error (MSE): {mse_p:.6f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_p:.6f}")
print(f"  Normalized RMSE (NRMSE): {nrmse_p:.6f}")
print(f"  Mean Absolute Error (MAE): {mae_p:.6f}")
print(f"  Relative MAE (RMAE): {rmae_p:.6f}")
print(f"  R^2 Score: {r2_p:.6f}")

compute_metrics(u_val_cpu, u_pred_cpu, "Velocity U-component")
compute_metrics(v_val_cpu, v_pred_cpu, "Velocity V-component")
compute_metrics(w_val_cpu, w_pred_cpu, "Velocity W-component")

# Prepare data for full simulation visualization

# CFD Data
x_cfd_val = x_val_cpu
y_cfd_val = y_val_cpu
z_cfd_val = z_val_cpu
p_cfd_val = p_val_cpu
u_cfd_val = u_val_cpu
v_cfd_val = v_val_cpu
w_cfd_val = w_val_cpu

# PINN Predictions
x_pinn_val = x_val_cpu
y_pinn_val = y_val_cpu
z_pinn_val = z_val_cpu
p_pinn_val = p_pred_cpu
u_pinn_val = u_pred_cpu
v_pinn_val = v_pred_cpu
w_pinn_val = w_pred_cpu

# Ensure the plots directory exists
os.makedirs("./plots", exist_ok=True)

plt.rcParams.update(
    {
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.dpi": 600,
        "font.size": 12,
        "axes.titlesize": 16,
    }
)

# Plotting
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(22, 10), constrained_layout=True)

# Pressure Field in x-y plane (CFD)
sc0 = ax[0, 0].scatter(
    x_cfd_val, y_cfd_val, c=-p_cfd_val, cmap="RdBu", rasterized=True, marker="o", s=10
)
ax[0, 0].set(
    aspect="equal", xlabel="x", ylabel="y", title="CFD Pressure Field (x-y Plane)"
)
fig.colorbar(sc0, ax=ax[0, 0], fraction=0.02, pad=0.005)

# Pressure Field in x-y plane (PINN)
sc1 = ax[0, 1].scatter(
    x_pinn_val, y_pinn_val, c=-p_pinn_val, cmap="RdBu", rasterized=True, marker="o", s=10
)
ax[0, 1].set(
    aspect="equal",
    xlabel="x",
    ylabel="y",
    title="PINN Predicted Pressure Field (x-y Plane)",
)
fig.colorbar(sc1, ax=ax[0, 1], fraction=0.02, pad=0.005)

# Pressure Error Field in x-y plane
error_p = np.abs(p_cfd_val - p_pinn_val)
sc2 = ax[0, 2].scatter(
    x_cfd_val, y_cfd_val, c=error_p, cmap="viridis", rasterized=True, marker="o", s=10
)
ax[0, 2].set(
    aspect="equal",
    xlabel="x",
    ylabel="y",
    title=f"Pressure Error Field (x-y Plane)\nNRMSE: {nrmse_p:.4f}",
)
fig.colorbar(sc2, ax=ax[0, 2], fraction=0.02, pad=0.005)

# Velocity U-component in x-y plane (CFD)
sc3 = ax[1, 0].scatter(
    x_cfd_val, y_cfd_val, c=u_cfd_val, cmap="viridis", rasterized=True, marker="o", s=10
)
ax[1, 0].set(
    aspect="equal", xlabel="x", ylabel="y", title="CFD Velocity U-component (x-y Plane)"
)
fig.colorbar(sc3, ax=ax[1, 0], fraction=0.02, pad=0.005)

# Velocity U-component in x-y plane (PINN)
sc4 = ax[1, 1].scatter(
    x_pinn_val, y_pinn_val, c=u_pinn_val, cmap="viridis", rasterized=True, marker="o", s=10
)
ax[1, 1].set(
    aspect="equal",
    xlabel="x",
    ylabel="y",
    title="PINN Predicted Velocity U-component (x-y Plane)",
)
fig.colorbar(sc4, ax=ax[1, 1], fraction=0.02, pad=0.005)

# Velocity U-component Error Field in x-y plane
error_u = np.abs(u_val_cpu - u_pred_cpu)
nrmse_u = rmse_u = np.sqrt(mean_squared_error(u_val_cpu, u_pred_cpu)) / (np.max(u_val_cpu) - np.min(u_val_cpu))
sc5 = ax[1, 2].scatter(
    x_cfd_val, y_cfd_val, c=error_u, cmap="viridis", rasterized=True, marker="o", s=10
)
ax[1, 2].set(
    aspect="equal",
    xlabel="x",
    ylabel="y",
    title=f"Velocity U-component Error Field (x-y Plane)\nNRMSE: {nrmse_u:.4f}",
)
fig.colorbar(sc5, ax=ax[1, 2], fraction=0.02, pad=0.005)

# Velocity V-component in x-z plane (CFD)
sc6 = ax[2, 0].scatter(
    x_cfd_val, z_cfd_val, c=v_cfd_val, cmap="viridis", rasterized=True, marker="o", s=10
)
ax[2, 0].set(
    aspect="equal", xlabel="x", ylabel="z", title="CFD Velocity V-component (x-z Plane)"
)
fig.colorbar(sc6, ax=ax[2, 0], fraction=0.02, pad=0.005)

# Velocity V-component in x-z plane (PINN)
sc7 = ax[2, 1].scatter(
    x_pinn_val, z_pinn_val, c=v_pinn_val, cmap="viridis", rasterized=True, marker="o", s=10
)
ax[2, 1].set(
    aspect="equal",
    xlabel="x",
    ylabel="z",
    title="PINN Predicted Velocity V-component (x-z Plane)",
)
fig.colorbar(sc7, ax=ax[2, 1], fraction=0.02, pad=0.005)

# Velocity V-component Error Field in x-z plane
error_v = np.abs(v_val_cpu - v_pred_cpu)
nrmse_v = rmse_v = np.sqrt(mean_squared_error(v_val_cpu, v_pred_cpu)) / (np.max(v_val_cpu) - np.min(v_val_cpu))
sc8 = ax[2, 2].scatter(
    x_cfd_val, z_cfd_val, c=error_v, cmap="viridis", rasterized=True, marker="o", s=10
)
ax[2, 2].set(
    aspect="equal",
    xlabel="x",
    ylabel="z",
    title=f"Velocity V-component Error Field (x-z Plane)\nNRMSE: {nrmse_v:.4f}",
)
fig.colorbar(sc8, ax=ax[2, 2], fraction=0.02, pad=0.005)

# Set the case number for the title and filename
sc = 1  # Case 1

fig.suptitle(f"Pressure and Velocity Field Comparison: Case {sc}", fontsize=20)
plt.savefig(f"./plots/fields_comparison_case_{sc}.png", bbox_inches='tight')
plt.show()

# Plot Total Loss Over Epochs
plt.figure(figsize=(8, 5))  # Adjusted figsize
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Total Loss", color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Total Loss Over Epochs")
plt.yscale('log')  # Logarithmic scale
plt.legend()
plt.grid(False)  # Remove grid
plt.savefig(f"./plots/total_loss_case_{sc}.png", bbox_inches='tight')
plt.show()

# Plot Physics Loss Over Epochs
plt.figure(figsize=(8, 5))  # Adjusted figsize
plt.plot(range(1, len(physics_losses) + 1), physics_losses, label="Physics Loss", color='green')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Physics Loss Over Epochs")
plt.yscale('log')  # Logarithmic scale
plt.legend()
plt.grid(False)  # Remove grid
plt.savefig(f"./plots/physics_loss_case_{sc}.png", bbox_inches='tight')
plt.show()

# Plot Data Loss Over Epochs
plt.figure(figsize=(8, 5))  # Adjusted figsize
plt.plot(range(1, len(data_losses) + 1), data_losses, label="Data Loss", color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Data Loss Over Epochs")
plt.yscale('log')  # Logarithmic scale
plt.legend()
plt.grid(False)  # Remove grid
plt.savefig(f"./plots/data_loss_case_{sc}.png", bbox_inches='tight')
plt.show()

# Plot Validation Loss Over Epochs
plt.figure(figsize=(8, 5))  # Adjusted figsize
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", color='purple')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss Over Epochs")
plt.yscale('log')  # Logarithmic scale
plt.legend()
plt.grid(False)  # Remove grid
plt.savefig(f"./plots/validation_loss_case_{sc}.png", bbox_inches='tight')
plt.show()
