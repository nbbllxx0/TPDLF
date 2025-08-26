import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# -----------------------------
# Batched Reconstruction Module
# -----------------------------
class BatchedSplineReconstruction(nn.Module):
    def __init__(self, degree):
        super(BatchedSplineReconstruction, self).__init__()
        self.degree = degree  # Now actually used for higher-order splines

    def get_knot_vector(self, n_cp, degree, device):
        """
        Returns a clamped uniform knot vector of length n_cp + degree + 1.
        The first (degree+1) knots are 0 and the last (degree+1) knots are 1.
        """
        p = degree
        if n_cp - p - 1 > 0:
            internal = torch.linspace(0, 1, steps=n_cp - p + 1, device=device)[1:-1]
            knots = torch.cat([
                torch.zeros(p + 1, device=device),
                internal,
                torch.ones(p + 1, device=device)
            ])
        else:
            knots = torch.cat([
                torch.zeros(p + 1, device=device),
                torch.ones(p + 1, device=device)
            ])
        return knots

    def bspline_basis(self, t, knots, degree):
        """
        Computes all B-Spline basis functions N_{i,degree}(t) for the given t values.
        """
        num_points = t.shape[0]
        n_cp = len(knots) - degree - 1
        device = t.device

        # Degree 0 basis functions
        N = torch.zeros(n_cp, num_points, device=device)
        for i in range(n_cp):
            left = knots[i].item()
            right = knots[i + 1].item()
            if i == n_cp - 1:
                N[i] = ((t >= left) & (t <= right)).float()
            else:
                N[i] = ((t >= left) & (t < right)).float()

        # Recursive definition for higher degrees.
        for d in range(1, degree + 1):
            N_new = torch.zeros(n_cp, num_points, device=device)
            for i in range(n_cp):
                denom1 = knots[i + d] - knots[i]
                term1 = 0.0
                if denom1 != 0:
                    term1 = ((t - knots[i]) / denom1) * N[i]
                term2 = 0.0
                if i + 1 < n_cp:
                    denom2 = knots[i + d + 1] - knots[i + 1]
                    if denom2 != 0:
                        term2 = ((knots[i + d + 1] - t) / denom2) * N[i + 1]
                N_new[i] = term1 + term2
            N = N_new

        return N  # shape: [n_cp, num_points]

    def forward_bspline(self, control_points, num_points):
        """
        Reconstruct a smooth closed B-Spline curve using higher-degree basis functions.
        """
        B, n_cp, _ = control_points.size()
        p = self.degree
        device = control_points.device

        # Generate knot vector.
        knots = self.get_knot_vector(n_cp, p, device)
        t = torch.linspace(knots[p].item(), knots[-p-1].item(), steps=num_points, device=device)
        basis = self.bspline_basis(t, knots, p)
        curve = torch.einsum('in,bid->bnd', basis, control_points)
        return curve.transpose(1, 2)  # [B, 2, num_points]

    def forward_nurbs(self, control_points, weights, num_points):
        """
        Computes a smooth closed NURBS curve.
        """
        B, n_cp, _ = control_points.size()
        p = self.degree
        device = control_points.device

        knots = self.get_knot_vector(n_cp, p, device)
        t = torch.linspace(knots[p].item(), knots[-p-1].item(), steps=num_points, device=device)
        basis = self.bspline_basis(t, knots, p)
        basis_exp = basis.unsqueeze(0).expand(B, -1, -1)
        weighted_basis = weights.unsqueeze(2) * basis_exp
        
        numerator = torch.einsum('bin,bij->bjn', weighted_basis, control_points)
        denominator = weighted_basis.sum(dim=1, keepdim=True)
        curve = numerator / (denominator + 1e-8)
        return curve  # [B, 2, num_points]

    def forward_bezier(self, control_points, num_points):
        """
        Standard Bernstein polynomial based Bezier reconstruction for closed curves.
        """
        B, num_cp, _ = control_points.size()
        n = num_cp - 1
        t = torch.linspace(0, 1, steps=num_points, device=control_points.device)
        t = t.view(1, num_points, 1)
        curve = torch.zeros(B, num_points, 2, device=control_points.device)
        for i in range(n + 1):
            coeff = torch.exp(torch.lgamma(torch.tensor(n + 1., device=control_points.device)) -
                              torch.lgamma(torch.tensor(i + 1., device=control_points.device)) -
                              torch.lgamma(torch.tensor(n - i + 1., device=control_points.device)))
            bernstein = coeff * (t ** i) * ((1 - t) ** (n - i))
            cp = control_points[:, i, :].unsqueeze(1)
            curve = curve + bernstein * cp
        return curve.transpose(1, 2)

    def forward(self, bspline_cp, nurbs_cp, nurbs_weights, bezier_cp, num_points):
        bspline_curve = self.forward_bspline(bspline_cp, num_points)
        bezier_curve  = self.forward_bezier(bezier_cp, num_points)
        nurbs_curve   = self.forward_nurbs(nurbs_cp, nurbs_weights, num_points)
        return bspline_curve, nurbs_curve, bezier_curve

# -----------------------------
# Autoencoder with Noise
# -----------------------------
class AirfoilAutoencoder(nn.Module):
    def __init__(self, num_control_points=10, degree=3, noise_dim=32, latent_dim=128, 
                 weights_enabled=True, sample_points=192):
        """
        Args:
            num_control_points (int): number of control points per curve.
            degree (int): spline degree.
            noise_dim (int): dimension of noise.
            latent_dim (int): latent representation dimension.
            weights_enabled (bool): if True, generate trainable weights for NURBS.
            sample_points (int): number of points per airfoil sample.
        """
        super(AirfoilAutoencoder, self).__init__()
        self.num_control_points = num_control_points
        self.weights_enabled = weights_enabled
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.sample_points = sample_points
        
        input_dim = 2 * sample_points
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim),
            nn.ReLU()
        )
        combined_dim = latent_dim + noise_dim

        # Decoders for each curve type.
        self.bspline_decoder = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_control_points * 2)
        )
        self.bezier_decoder = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_control_points * 2)
        )
        self.nurbs_decoder = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_control_points * 3)
        )
        self.reconstruction = BatchedSplineReconstruction(degree)

    def force_closed(self, cp):
        """
        Forces a set of control points to represent a closed curve by setting the last point equal to the first.
        Args:
            cp: Tensor of shape [B, num_control_points, 2]
        Returns:
            cp: Tensor with cp[:, -1, :] == cp[:, 0, :]
        """
        cp = cp.clone()
        cp[:, -1, :] = cp[:, 0, :]
        return cp

    def forward(self, x, noise):
        """
        Args:
            x: airfoil sample [B, 2, sample_points]
            noise: noise vector [B, noise_dim]
        Returns:
            Reconstructed curves [B, 2, sample_points]
        """
        B = x.size(0)
        latent = self.encoder(x)
        latent_noise = torch.cat([latent, noise], dim=1)
        
        # Decode control points.
        bspline_params = self.bspline_decoder(latent_noise)
        bspline_cp = bspline_params.view(B, self.num_control_points, 2)
        bspline_cp = self.force_closed(bspline_cp)
        
        bezier_params = self.bezier_decoder(latent_noise)
        bezier_cp = bezier_params.view(B, self.num_control_points, 2)
        bezier_cp = self.force_closed(bezier_cp)
        
        nurbs_params = self.nurbs_decoder(latent_noise)
        nurbs_cp = nurbs_params[:, :self.num_control_points*2].view(B, self.num_control_points, 2)
        nurbs_cp = self.force_closed(nurbs_cp)
        nurbs_weights = nurbs_params[:, self.num_control_points*2:]
        nurbs_weights = torch.nn.functional.softplus(nurbs_weights)

        bspline_curve, nurbs_curve, bezier_curve = self.reconstruction(
            bspline_cp, nurbs_cp, nurbs_weights, bezier_cp, num_points=self.sample_points)
        return bspline_curve, nurbs_curve, bezier_curve

    def extract_params(self, x, noise):
        B = x.size(0)
        latent = self.encoder(x)
        latent_noise = torch.cat([latent, noise], dim=1)
        bspline_params = self.bspline_decoder(latent_noise)
        bspline_cp = bspline_params.view(B, self.num_control_points, 2)
        bspline_cp = self.force_closed(bspline_cp)
        
        bezier_params = self.bezier_decoder(latent_noise)
        bezier_cp = bezier_params.view(B, self.num_control_points, 2)
        bezier_cp = self.force_closed(bezier_cp)
        
        nurbs_params = self.nurbs_decoder(latent_noise)
        nurbs_cp = nurbs_params[:, :self.num_control_points*2].view(B, self.num_control_points, 2)
        nurbs_cp = self.force_closed(nurbs_cp)
        nurbs_weights = nurbs_params[:, self.num_control_points*2:]
        nurbs_weights = torch.nn.functional.softplus(nurbs_weights)
        return bspline_cp, nurbs_cp, nurbs_weights, bezier_cp

# -----------------------------
# Airfoil Dataset
# -----------------------------
class AirfoilDataset(Dataset):
    def __init__(self, data_tensor, augment=False):
        self.data = data_tensor
        self.augment = augment

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = sample.permute(1, 0)
        if self.augment:
            angle = np.random.uniform(-10, 10)
            rad = np.deg2rad(angle)
            rotation_matrix = torch.tensor(
                [[np.cos(rad), -np.sin(rad)],
                 [np.sin(rad),  np.cos(rad)]],
                dtype=sample.dtype,
                device=sample.device
            )
            sample = torch.matmul(rotation_matrix, sample)
            scale = np.random.uniform(0.9, 1.1)
            sample = sample * scale
        return sample

# -----------------------------
# Geometric Loss (Smoothness Loss)
# -----------------------------
def compute_geometric_loss(curve):
    """
    Computes a smoothness loss based on second finite differences computed with wrap-around.
    Args:
        curve: tensor of shape [B, 2, num_points]
    Returns:
        A scalar smoothness loss.
    """
    # Use roll to wrap-around for closed curves.
    d_curve = torch.roll(curve, shifts=-1, dims=2) - curve
    dd_curve = torch.roll(d_curve, shifts=-1, dims=2) - d_curve
    return torch.mean(dd_curve ** 2)

def compute_custom_smoothing_loss(curve, omega):
    """
    Computes the custom smoothing loss L_S for the Y coordinates of a curve.
    For each sample in the batch, the moving average is computed using a window of 3 points
    with cyclic wrap-around. The loss penalizes the deviation of each Y coordinate from its moving average.
    
    Args:
        curve: Tensor of shape [B, 2, num_points].
        omega: Coefficient that determines the strength of the smoothing penalty.
        
    Returns:
        A scalar smoothing loss L_S.
    """
    # Extract Y coordinates from the curve.
    Y = curve[:, 1, :]  # shape: [B, num_points]
    # Compute the moving average using cyclic wrap-around.
    Y_left  = torch.roll(Y, shifts=1, dims=1)
    Y_right = torch.roll(Y, shifts=-1, dims=1)
    Y_avg = (Y_left + Y + Y_right) / 3.0
    # Compute deviation.
    delta = Y - Y_avg
    # Loss: mean squared deviation, scaled by omega.
    loss = omega * torch.mean(delta ** 2)
    return loss


# -----------------------------
# Training, Evaluation, and Visualization
# -----------------------------
def evaluate(model, dataloader, criterion, noise_dim, device, geo_loss_weight=0.1, smoothing_loss_weight=0.1):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            B = batch.size(0)
            noise = torch.randn(B, noise_dim, device=device)
            bspline_curve, nurbs_curve, bezier_curve = model(batch, noise)
            
            recon_loss = (criterion(bspline_curve, batch) +
                          criterion(nurbs_curve, batch) +
                          criterion(bezier_curve, batch))
            geo_loss = (compute_geometric_loss(bspline_curve) +
                        compute_geometric_loss(nurbs_curve) +
                        compute_geometric_loss(bezier_curve))
            smooth_loss = (compute_custom_smoothing_loss(bspline_curve, smoothing_loss_weight) +
                           compute_custom_smoothing_loss(nurbs_curve, smoothing_loss_weight) +
                           compute_custom_smoothing_loss(bezier_curve, smoothing_loss_weight))
            loss = recon_loss + geo_loss_weight * geo_loss + smooth_loss
            total_loss += loss.item()
    return total_loss / len(dataloader)

# -----------------------------
# Custom Smoothing Loss L_S
# -----------------------------
# -----------------------------
# Custom Smoothing Loss L_S (from before)
# -----------------------------
def compute_custom_smoothing_loss(curve, omega):
    """
    Computes the custom smoothing loss L_S for the Y coordinates of a curve.
    For each sample in the batch, the moving average is computed using a window of 3 points
    with cyclic wrap-around. The loss penalizes the deviation of each Y coordinate from its moving average.
    
    Args:
        curve: Tensor of shape [B, 2, num_points].
        omega: Coefficient that determines the strength of the smoothing penalty.
        
    Returns:
        A scalar smoothing loss L_S.
    """
    # Extract Y coordinates.
    Y = curve[:, 1, :]  # shape: [B, num_points]
    # Compute moving average with cyclic wrap-around.
    Y_left  = torch.roll(Y, shifts=1, dims=1)
    Y_right = torch.roll(Y, shifts=-1, dims=1)
    Y_avg = (Y_left + Y + Y_right) / 3.0
    delta = Y - Y_avg
    loss = omega * torch.mean(delta ** 2)
    return loss

# -----------------------------
# C1 Continuity Loss
# -----------------------------
def compute_C1_continuity_loss(curve):
    """
    Enforces first-derivative (C1) continuity along the curve.
    We approximate the derivative at each point using forward and backward differences
    and penalize their discrepancy.
    
    Args:
        curve: Tensor of shape [B, 2, num_points]
        
    Returns:
        A scalar loss penalizing discontinuities in the first derivative.
    """
    # Compute forward differences.
    d_forward = torch.roll(curve, shifts=-1, dims=2) - curve
    # Compute backward differences.
    d_backward = curve - torch.roll(curve, shifts=1, dims=2)
    # The discrepancy between these two approximations indicates a lack of C1 continuity.
    d_diff = d_forward - d_backward
    return torch.mean(d_diff ** 2)

# -----------------------------
# C2 Continuity Loss
# -----------------------------
def compute_C2_continuity_loss(curve):
    """
    Enforces second-derivative (C2) continuity along the curve.
    We compute the second finite differences (which approximate curvature changes)
    and penalize large differences.
    
    Args:
        curve: Tensor of shape [B, 2, num_points]
        
    Returns:
        A scalar loss penalizing discontinuities in the second derivative.
    """
    d_curve = torch.roll(curve, shifts=-1, dims=2) - curve
    dd_curve = torch.roll(d_curve, shifts=-1, dims=2) - d_curve
    return torch.mean(dd_curve ** 2)

# -----------------------------
# Geometric Loss (Existing)
# -----------------------------
def compute_geometric_loss(curve):
    """
    Computes a smoothness loss based on second finite differences computed with wrap-around.
    Args:
        curve: tensor of shape [B, 2, num_points]
    Returns:
        A scalar smoothness loss.
    """
    d_curve = torch.roll(curve, shifts=-1, dims=2) - curve
    dd_curve = torch.roll(d_curve, shifts=-1, dims=2) - d_curve
    return torch.mean(dd_curve ** 2)

# -----------------------------
# Modified Evaluation Function
# -----------------------------
def evaluate(model, dataloader, criterion, noise_dim, device,
             geo_loss_weight=0.0, smoothing_loss_weight=0.0,
             c1_loss_weight=0.1, c2_loss_weight=0.1):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            B = batch.size(0)
            noise = torch.randn(B, noise_dim, device=device)
            bspline_curve, nurbs_curve, bezier_curve = model(batch, noise)
            
            recon_loss = (criterion(bspline_curve, batch) +
                          criterion(nurbs_curve, batch) +
                          criterion(bezier_curve, batch))
            
            geo_loss = (compute_geometric_loss(bspline_curve) +
                        compute_geometric_loss(nurbs_curve) +
                        compute_geometric_loss(bezier_curve))
            
            smooth_loss = (compute_custom_smoothing_loss(bspline_curve, smoothing_loss_weight) +
                           compute_custom_smoothing_loss(nurbs_curve, smoothing_loss_weight) +
                           compute_custom_smoothing_loss(bezier_curve, smoothing_loss_weight))
            
            c1_loss = (compute_C1_continuity_loss(bspline_curve) +
                       compute_C1_continuity_loss(nurbs_curve) +
                       compute_C1_continuity_loss(bezier_curve))
            
            c2_loss = (compute_C2_continuity_loss(bspline_curve) +
                       compute_C2_continuity_loss(nurbs_curve) +
                       compute_C2_continuity_loss(bezier_curve))
            
            loss = recon_loss + geo_loss_weight * geo_loss + smooth_loss \
                   + c1_loss_weight * c1_loss + c2_loss_weight * c2_loss
            total_loss += loss.item()
    return total_loss / len(dataloader)

# -----------------------------
# Modified Training Function
# -----------------------------
# -----------------------------
# Modified Training Function with Epoch Checkpointing and Multiple Visualizations
# -----------------------------
def train(model, train_loader, test_loader, num_epochs=20, lr=1e-3, noise_dim=32, 
          patience=5, save_path="best_model.pth", device='cpu', test_dataset=None,
          geo_loss_weight=0.0, smoothing_loss_weight=0,
          c1_loss_weight=0, c2_loss_weight=0):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    patience_counter = 0

    # Lists for tracking losses over epochs.
    train_total_losses = []
    train_recon_losses = []
    train_c1_losses = []
    train_c2_losses = []
    test_total_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_recon_loss = 0.0
        running_c1_loss = 0.0
        running_c2_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            B = batch.size(0)
            noise = torch.randn(B, noise_dim, device=device)
            optimizer.zero_grad()
            bspline_curve, nurbs_curve, bezier_curve = model(batch, noise)
            
            # Compute losses.
            recon_loss = (criterion(bspline_curve, batch) +
                          criterion(nurbs_curve, batch) +
                          criterion(bezier_curve, batch))
            
            geo_loss = (compute_geometric_loss(bspline_curve) +
                        compute_geometric_loss(nurbs_curve) +
                        compute_geometric_loss(bezier_curve))
            
            smooth_loss = (compute_custom_smoothing_loss(bspline_curve, smoothing_loss_weight) +
                           compute_custom_smoothing_loss(nurbs_curve, smoothing_loss_weight) +
                           compute_custom_smoothing_loss(bezier_curve, smoothing_loss_weight))
            
            c1_loss = (compute_C1_continuity_loss(bspline_curve) +
                       compute_C1_continuity_loss(nurbs_curve) +
                       compute_C1_continuity_loss(bezier_curve))
            
            c2_loss = (compute_C2_continuity_loss(bspline_curve) +
                       compute_C2_continuity_loss(nurbs_curve) +
                       compute_C2_continuity_loss(bezier_curve))
            
            total_loss = recon_loss + geo_loss_weight * geo_loss + smooth_loss \
                         + c1_loss_weight * c1_loss + c2_loss_weight * c2_loss
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            running_recon_loss += recon_loss.item()
            running_c1_loss += c1_loss.item()
            running_c2_loss += c2_loss.item()
        
        # Average losses per epoch.
        train_loss = running_loss / len(train_loader)
        train_recon = running_recon_loss / len(train_loader)
        train_c1 = running_c1_loss / len(train_loader)
        train_c2 = running_c2_loss / len(train_loader)
        train_total_losses.append(train_loss)
        train_recon_losses.append(train_recon)
        train_c1_losses.append(train_c1)
        train_c2_losses.append(train_c2)
        
        test_loss = evaluate(model, test_loader, criterion, noise_dim, device,
                             geo_loss_weight, smoothing_loss_weight, c1_loss_weight, c2_loss_weight)
        test_total_losses.append(test_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
        
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model with test loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Optional: Save checkpoints and visualizations every 50 epochs.
        if (epoch + 1) % 50 == 0 and test_dataset is not None:
            checkpoint_path = f"checkpoint_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            subfolder = os.path.join("outputs", f"epoch{epoch+1}")
            os.makedirs(subfolder, exist_ok=True)
            for sample_idx in range(5):
                sample = test_dataset[sample_idx]
                noise = torch.randn(noise_dim)
                params = visualize_and_save(sample, noise, model, noise_dim, device, 
                                            sample_idx=f"sample{sample_idx}_epoch{epoch+1}", 
                                            out_dir=subfolder)
                print(f"Visualization saved for sample {sample_idx} at epoch {epoch+1}.")
    
    # Plot and save the loss curves.
# Plot individual loss curves with logarithmic y-axis.
    loss_dict = {
        "Total_Train_Loss": train_total_losses,
        "Reconstruction_Loss": train_recon_losses,
        "C1_Continuity_Loss": train_c1_losses,
        "C2_Continuity_Loss": train_c2_losses
    }

    epochs = range(1, len(train_total_losses) + 1)
    for loss_name, loss_values in loss_dict.items():
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, loss_values, label=loss_name)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")  # Set y-axis to logarithmic scale.
        plt.title(f"{loss_name.replace('_', ' ')} Over Epochs")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join("outputs", f"{loss_name}_log.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved {loss_name} plot to {plot_path}")

    
    # Plot test loss curve.
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, test_total_losses, label="Test Total Loss", color='magenta')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Test Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    test_loss_plot_path = os.path.join("outputs", "test_loss_plot.png")
    plt.savefig(test_loss_plot_path)
    plt.close()
    print(f"Saved test loss plot to {test_loss_plot_path}")

    return model



def visualize_and_save(sample, noise, model, noise_dim, device, sample_idx, out_dir="outputs"):
    model.eval()
    sample = sample.unsqueeze(0).to(device)
    noise = noise.unsqueeze(0).to(device)
    with torch.no_grad():
        bspline_curve, nurbs_curve, bezier_curve = model(sample, noise)
        bspline_cp, nurbs_cp, nurbs_weights, bezier_cp = model.extract_params(sample, noise)
    
    gt_curve = sample.cpu().numpy().squeeze()        
    bspline_curve_np = bspline_curve.cpu().numpy().squeeze()
    nurbs_curve_np = nurbs_curve.cpu().numpy().squeeze()
    bezier_curve_np = bezier_curve.cpu().numpy().squeeze()
    
    plt.figure(figsize=(24, 6))
    plt.plot(gt_curve[0], gt_curve[1], 'k-', label='Ground Truth')
    plt.plot(bspline_curve_np[0], bspline_curve_np[1], 'r--', label='B-Spline')
    plt.plot(nurbs_curve_np[0], nurbs_curve_np[1], 'b--', label='NURBS')
    plt.plot(bezier_curve_np[0], bezier_curve_np[1], 'g--', label='Bezier')
    plt.legend()
    plt.title(f"Sample {sample_idx} Curve Reconstruction")
    plt.xlabel("x")
    plt.ylabel("y")
    
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"sample_{sample_idx}_curve.png")
    plt.savefig(plot_path)
    plt.close()
    
    params = {
        "B-Spline": bspline_cp.cpu().numpy().squeeze(),
        "NURBS_CP": nurbs_cp.cpu().numpy().squeeze(),
        "NURBS_Weights": nurbs_weights.cpu().numpy().squeeze(),
        "Bezier": bezier_cp.cpu().numpy().squeeze()
    }
    return params

def save_params_to_csv(params_list, curve_type, out_dir="outputs"):
    rows = []
    for sample_idx, arr in params_list:
        if curve_type == "NURBS":
            cp_arr, weight_arr = arr
            num_cp = cp_arr.shape[0]
            for cp_idx in range(num_cp):
                row = {"sample_idx": sample_idx,
                       "control_point": cp_idx,
                       "x": cp_arr[cp_idx, 0],
                       "y": cp_arr[cp_idx, 1],
                       "weight": weight_arr[cp_idx]}
                rows.append(row)
        else:
            num_cp = arr.shape[0]
            for cp_idx in range(num_cp):
                row = {"sample_idx": sample_idx,
                       "control_point": cp_idx,
                       "x": arr[cp_idx, 0],
                       "y": arr[cp_idx, 1]}
                rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{curve_type}_params.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {curve_type} parameters to {csv_path}")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your train and test data; shape: [N, 192, 2]
    train_data = torch.tensor(np.load("train.npy"), dtype=torch.float32)
    test_data = torch.tensor(np.load("test.npy"), dtype=torch.float32)
    
    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    
    sample_points = train_data.shape[1]
    train_dataset = AirfoilDataset(train_data, augment=True)
    test_dataset = AirfoilDataset(test_data, augment=False)

    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Hyperparameters.
    num_control_points = 64
    degree = 3
    noise_dim = 0
    latent_dim = 128
    weights_enabled = True
    num_epochs = 4000
    lr = 1e-3
    patience = 100
    geo_loss_weight = 0
    
    model = AirfoilAutoencoder(num_control_points=num_control_points,
                               degree=degree,
                               noise_dim=noise_dim,
                               latent_dim=latent_dim,
                               weights_enabled=weights_enabled,
                               sample_points=sample_points).to(device)

    train(model, train_loader, test_loader, num_epochs=num_epochs, lr=lr,
          noise_dim=noise_dim, patience=patience, save_path="best_model_Air_NOC1C2.pth",
          device=device, test_dataset=test_dataset, geo_loss_weight=geo_loss_weight)

    # Process first 10 test samples: save plots and key parameters.
    bspline_params_list = []  
    nurbs_params_list = []    
    bezier_params_list = []   
    
    for sample_idx in range(10):
        sample = test_dataset[sample_idx]
        noise = torch.randn(noise_dim)
        params = visualize_and_save(sample, noise, model, noise_dim, device, sample_idx, out_dir="outputs")
        bspline_params_list.append((sample_idx, params["B-Spline"]))
        nurbs_params_list.append((sample_idx, (params["NURBS_CP"], params["NURBS_Weights"])))
        bezier_params_list.append((sample_idx, params["Bezier"]))
    
    save_params_to_csv(bspline_params_list, "B-Spline", out_dir="outputs")
    save_params_to_csv(nurbs_params_list, "NURBS", out_dir="outputs")
    save_params_to_csv(bezier_params_list, "Bezier", out_dir="outputs")
