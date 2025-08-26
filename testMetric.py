import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Import the necessary components from train.py
sys.path.append('C:/3IN1')

from train import (
    AirfoilAutoencoder,
    AirfoilDataset,
    evaluate,
    visualize_and_save,
    save_params_to_csv,
    compute_geometric_loss,
    compute_C1_continuity_loss,
    compute_C2_continuity_loss
)

def create_global_plots(model, dataset, noise_dim, device, out_dir, grid_shape=(12, 3), 
                        sample_range=(5, 200), step=5):
    """
    Creates global plots for Ground Truth, B-Spline, NURBS, and Bezier curves.
    """
    model.eval()
    sample_indices = list(range(sample_range[0], sample_range[1], step))
    num_plots = len(sample_indices)
    
    n_rows, n_cols = grid_shape
    total_subplots = n_rows * n_cols
    if num_plots > total_subplots:
        sample_indices = sample_indices[:total_subplots]
        num_plots = total_subplots
    elif num_plots < total_subplots:
        sample_indices += [sample_indices[-1]] * (total_subplots - num_plots)
        num_plots = total_subplots
    
    samples = [dataset[i] for i in sample_indices]
    samples_tensor = torch.stack(samples).to(device)
    
    noise = torch.randn(num_plots, noise_dim, device=device) if noise_dim > 0 else torch.empty(num_plots, 0, device=device)
    
    with torch.no_grad():
        bspline_curves, nurbs_curves, bezier_curves = model(samples_tensor, noise)
    
    gt_curves      = samples_tensor.cpu().numpy()
    bspline_curves = bspline_curves.cpu().numpy()
    nurbs_curves   = nurbs_curves.cpu().numpy()
    bezier_curves  = bezier_curves.cpu().numpy()
    
    def plot_grid(curves, title, filename):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 10))
        axs = axs.flatten()
        for idx in range(num_plots):
            ax = axs[idx]
            # Data is consistently [2, Points], so transpose for plotting.
            curve = curves[idx].T
            ax.plot(curve[:, 0], curve[:, 1], 'k-')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
        save_path = os.path.join(out_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved global plot: {save_path}")
    
    # The plot_grid function handles transposition, so no need to reshape here.
    plot_grid(gt_curves, "Ground Truth Curves", "global_GT.png")
    plot_grid(bspline_curves, "B-Spline Reconstructions", "global_BSpline.png")
    plot_grid(nurbs_curves, "NURBS Reconstructions", "global_NURBS.png")
    plot_grid(bezier_curves, "Bezier Reconstructions", "global_Bezier.png")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data; expected shape: [N, 192, 2]
    test_data = torch.tensor(np.load("test.npy"), dtype=torch.float32)
    print("Test data shape:", test_data.shape)
    
    # Create the test dataset and dataloader
    test_dataset = AirfoilDataset(test_data)
    batch_size = 512
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Hyperparameters
    num_control_points = 64
    degree = 3
    noise_dim = 0
    latent_dim = 128
    weights_enabled = True
    sample_points = test_data.shape[1]
    
    # Instantiate the model
    model = AirfoilAutoencoder(num_control_points=num_control_points,
                               degree=degree,
                               noise_dim=noise_dim,
                               latent_dim=latent_dim,
                               weights_enabled=weights_enabled,
                               sample_points=sample_points).to(device)
    
    # Load the best saved model state.
    model_state = torch.load("best_model_Ori.pth", map_location=device)
    model.load_state_dict(model_state)
    model.eval()
    
    # Evaluate the overall test loss.
    criterion = torch.nn.MSELoss()
    test_loss = evaluate(model, test_loader, criterion, noise_dim, device)
    print(f"Overall Test Loss: {test_loss:.6f}")
    
    # --- Independent Loss and Timing Calculation ---
    bspline_mse_total, bspline_geo_total, bspline_c1_total, bspline_c2_total = 0, 0, 0, 0
    nurbs_mse_total,   nurbs_geo_total,   nurbs_c1_total,   nurbs_c2_total   = 0, 0, 0, 0
    bezier_mse_total,  bezier_geo_total,  bezier_c1_total,  bezier_c2_total  = 0, 0, 0, 0
    
    num_batches = 0
    total_inference_time = 0.0

    for batch in test_loader:
        batch = batch.to(device)
        B = batch.size(0)
        noise = torch.randn(B, noise_dim, device=device) if noise_dim > 0 else torch.empty(B, 0, device=device)
        
        if device.type == 'cuda': torch.cuda.synchronize()
        start_time = time.time()
        
        bspline_curve, nurbs_curve, bezier_curve = model(batch, noise)
        
        if device.type == 'cuda': torch.cuda.synchronize()
        end_time = time.time()
        total_inference_time += (end_time - start_time)
        
        # Calculate losses
        bspline_mse = criterion(bspline_curve, batch); nurbs_mse = criterion(nurbs_curve, batch); bezier_mse = criterion(bezier_curve, batch)
        bspline_geo = compute_geometric_loss(bspline_curve); nurbs_geo = compute_geometric_loss(nurbs_curve); bezier_geo = compute_geometric_loss(bezier_curve)
        bspline_c1 = compute_C1_continuity_loss(bspline_curve); nurbs_c1 = compute_C1_continuity_loss(nurbs_curve); bezier_c1 = compute_C1_continuity_loss(bezier_curve)
        bspline_c2 = compute_C2_continuity_loss(bspline_curve); nurbs_c2 = compute_C2_continuity_loss(nurbs_curve); bezier_c2 = compute_C2_continuity_loss(bezier_curve)
        
        bspline_mse_total += bspline_mse.item(); bspline_geo_total += bspline_geo.item(); bspline_c1_total += bspline_c1.item(); bspline_c2_total += bspline_c2.item()
        nurbs_mse_total   += nurbs_mse.item();   nurbs_geo_total   += nurbs_geo.item();   nurbs_c1_total   += nurbs_c1.item();   nurbs_c2_total   += nurbs_c2.item()
        bezier_mse_total  += bezier_mse.item();  bezier_geo_total  += bezier_geo.item();  bezier_c1_total  += bezier_c1.item();  bezier_c2_total  += bezier_c2.item()
        
        num_batches += 1

    # --- Print Summaries ---
    print("\n" + "="*50)
    print("Independent Losses per Curve (Averaged)")
    print("="*50)
    print("B-Spline -> MSE: {:.6f}, Geo: {:.6f}, C1: {:.6f}, C2: {:.6f}".format(bspline_mse_total/num_batches, bspline_geo_total/num_batches, bspline_c1_total/num_batches, bspline_c2_total/num_batches))
    print("NURBS    -> MSE: {:.6f}, Geo: {:.6f}, C1: {:.6f}, C2: {:.6f}".format(nurbs_mse_total/num_batches, nurbs_geo_total/num_batches, nurbs_c1_total/num_batches, nurbs_c2_total/num_batches))
    print("Bezier   -> MSE: {:.6f}, Geo: {:.6f}, C1: {:.6f}, C2: {:.6f}".format(bezier_mse_total/num_batches, bezier_geo_total/num_batches, bezier_c1_total/num_batches, bezier_c2_total/num_batches))
    
    avg_time_per_batch = total_inference_time / num_batches
    avg_time_per_sample = total_inference_time / len(test_dataset)

    print("\n" + "="*50)
    print("Inference Performance Summary")
    print("="*50)
    print(f"  Total Inference Time for {len(test_dataset)} samples: {total_inference_time:.4f} seconds")
    print(f"  Average Time per Batch ({batch_size} samples): {avg_time_per_batch:.6f} seconds")
    print(f"  Average Time per Sample: {avg_time_per_sample:.6f} seconds")
    print("="*50)
    
    # --- Visualization and Parameter Extraction ---
    out_dir = "test_outputs_Ori_Metric"
    os.makedirs(out_dir, exist_ok=True)
    
    start_idx, end_idx = 100, 110
    
    # **FIXED**: Initialize lists to store parameters for CSV saving
    bspline_params_list = []
    nurbs_params_list = []
    bezier_params_list = []
    
    for sample_idx in range(start_idx, end_idx):
        sample = test_dataset[sample_idx]
        noise = torch.randn(noise_dim) if noise_dim > 0 else torch.empty(0)
        params = visualize_and_save(sample, noise, model, noise_dim, device, sample_idx, out_dir=out_dir)
        print(f"Processed sample {sample_idx} for visualization.")
        
        # **FIXED**: Collect parameters from each sample
        bspline_params_list.append((sample_idx, params["B-Spline"]))
        nurbs_params_list.append((sample_idx, (params["NURBS_CP"], params["NURBS_Weights"])))
        bezier_params_list.append((sample_idx, params["Bezier"]))
    
    # **FIXED**: Save the collected parameters to CSV files
    save_params_to_csv(bspline_params_list, "B-Spline", out_dir=out_dir)
    save_params_to_csv(nurbs_params_list, "NURBS", out_dir=out_dir)
    save_params_to_csv(bezier_params_list, "Bezier", out_dir=out_dir)
    
    print("Visualization and parameter CSV saving complete.")
    
    create_global_plots(model, test_dataset, noise_dim, device, out_dir)

if __name__ == '__main__':
    main()
