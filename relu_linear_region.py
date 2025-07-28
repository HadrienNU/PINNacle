import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Define the ReLU network (as before)
class ReLUModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[4, 4]):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))  # Output layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Create and evaluate the model
model = ReLUModel()

# Example 2D input grid
x_range = torch.linspace(-2, 2, 750)
y_range = torch.linspace(-2, 2, 750)
xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

# === Step 1: Register Hooks ===
activation_storage = []

def get_relu_hook():
    def hook(module, input, output):
        # Store binary ReLU activation pattern (1 = active, 0 = inactive)
        activation_storage.append((output > 0).int().cpu())
    return hook

# Assume `model` is your pre-existing network
relu_hooks = []
for module in model.modules():
    if isinstance(module, torch.nn.ReLU):
        relu_hooks.append(module.register_forward_hook(get_relu_hook()))

# === Step 2: Run model on input grid ===
with torch.no_grad():
    model.eval()
    _ = model(grid_points)

# === Step 3: Process activation pattern ===
# Combine all recorded ReLU layer activations into one binary vector per input
print([act.shape for act in activation_storage])
activation_patterns = torch.cat(activation_storage, dim=1)  # shape: [num_points, total_relus]
activation_storage.clear()  # Clean up

# Convert each activation pattern to a unique ID
pattern_ids = activation_patterns.numpy().dot(1 << np.arange(activation_patterns.shape[1]))
unique_ids, remapped_ids = np.unique(pattern_ids, return_inverse=True)
print("Number of region", len(unique_ids))
remapped_ids += 1  # So it starts from 1 instead of 0
pattern_image = remapped_ids.reshape(xx.shape)

# === Step 4: Plot the regions ===
plt.figure(figsize=(8, 6))
plt.imshow(pattern_image.T, extent=(-2, 2, -2, 2), origin='lower', cmap='tab20')
plt.colorbar(label='Linear Region ID (Activation Pattern)')
plt.title('Linear Regions via ReLU Activation Patterns (Hook-Based)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.tight_layout()
plt.show()
