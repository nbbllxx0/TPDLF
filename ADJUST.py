import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import simpledialog
###########################################
# 1) Helper Functions: B-Spline / NURBS
###########################################
def get_knot_vector(n_cp, degree):
    p = degree
    if n_cp - p - 1 > 0:
        internal = np.linspace(0, 1, n_cp - p + 1)[1:-1]
        knots = np.concatenate([
            np.zeros(p + 1),
            internal,
            np.ones(p + 1)
        ])
    else:
        knots = np.concatenate([
            np.zeros(p + 1),
            np.ones(p + 1)
        ])
    return knots

def bspline_basis(t, knots, degree):
    t = np.asarray(t)
    n_cp = len(knots) - degree - 1
    
    # Degree 0
    N = np.zeros((n_cp, len(t)), dtype=float)
    for i in range(n_cp):
        left = knots[i]
        right = knots[i + 1]
        if i == (n_cp - 1):
            N[i] = ((t >= left) & (t <= right)).astype(float)
        else:
            N[i] = ((t >= left) & (t < right)).astype(float)
    
    # Higher degrees
    for d in range(1, degree + 1):
        N_new = np.zeros_like(N)
        for i in range(n_cp):
            denom1 = knots[i + d] - knots[i]
            term1 = 0.0
            if denom1 != 0:
                term1 = ((t - knots[i]) / denom1) * N[i]
            
            term2 = 0.0
            if (i + 1) < n_cp:
                denom2 = knots[i + d + 1] - knots[i + 1]
                if denom2 != 0:
                    term2 = ((knots[i + d + 1] - t) / denom2) * N[i + 1]
            N_new[i] = term1 + term2
        N = N_new
    return N

def compute_bspline_curve(control_points, degree=3, num_points=192):
    n_cp = control_points.shape[0]
    knots = get_knot_vector(n_cp, degree)
    t_min, t_max = knots[degree], knots[-degree-1]
    t_vals = np.linspace(t_min, t_max, num_points)
    basis = bspline_basis(t_vals, knots, degree)
    
    curve_x = np.sum(basis * control_points[:, 0:1], axis=0)
    curve_y = np.sum(basis * control_points[:, 1:2], axis=0)
    return np.vstack([curve_x, curve_y])

def compute_nurbs_curve(control_points, weights, degree=3, num_points=192):
    n_cp = control_points.shape[0]
    knots = get_knot_vector(n_cp, degree)
    t_min, t_max = knots[degree], knots[-degree-1]
    t_vals = np.linspace(t_min, t_max, num_points)
    basis = bspline_basis(t_vals, knots, degree)
    
    W = basis * weights[:, None]  # (n_cp, num_points)
    x_num = np.sum(W * control_points[:, 0:1], axis=0)
    y_num = np.sum(W * control_points[:, 1:2], axis=0)
    denom = np.sum(W, axis=0) + 1e-12
    curve_x = x_num / denom
    curve_y = y_num / denom
    return np.vstack([curve_x, curve_y])

###############################################
# 2) Draggable + On-Click Weight Editing
###############################################
class DraggableCurvePlot:
    """
    - Left-click (button=1) + drag to move a control point.
    - Right-click (button=3) on a single control point => edit weight in console (if NURBS).
    - If multiple or no points found under right-click, ignore to prevent glitch.
    """
    def __init__(self, ax, control_points, 
                 gt_curve=None, use_nurbs=False, weights=None,
                 degree=3, num_points=192):
        """
        Args:
            ax: Matplotlib Axes for the curve
            control_points: (n_cp, 2)
            gt_curve: (2, M) or None, ground truth for reference
            use_nurbs: bool => NURBS or B-spline
            weights: (n_cp,) or None
            degree, num_points: spline config
        """
        self.ax = ax
        self.degree = degree
        self.num_points = num_points
        self.use_nurbs = use_nurbs
        self.control_points = control_points
        
        n_cp = control_points.shape[0]
        self.weights = weights if (weights is not None) else np.ones(n_cp)

        self.gt_curve = gt_curve
        
        # Plot GT curve (optional)
        if self.gt_curve is not None and self.gt_curve.size > 0:
            ax.plot(self.gt_curve[0, :], self.gt_curve[1, :], 'k--', lw=2, label='Ground Truth')
        
        # Scatter control points
        self.scat = ax.scatter(
            self.control_points[:, 0],
            self.control_points[:, 1],
            c='red', s=40, zorder=5, picker=True, label='Control Points'
        )
        
        # Plot the initial curve
        self.curve_line, = ax.plot([], [], 'b-', lw=2, label='Spline/NURBS')
        self.update_curve()
        
        # For dragging
        self._dragging_idx = None
        
        # Connect events
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def update_curve(self):
        """
        Recompute B-spline or NURBS curve and update the plot.
        """
        if self.use_nurbs:
            curve = compute_nurbs_curve(
                self.control_points, self.weights,
                degree=self.degree, num_points=self.num_points
            )
        else:
            curve = compute_bspline_curve(
                self.control_points,
                degree=self.degree, num_points=self.num_points
            )
        self.curve_line.set_xdata(curve[0, :])
        self.curve_line.set_ydata(curve[1, :])
        self.ax.figure.canvas.draw_idle()

    def on_press(self, event):
        """
        On mouse press:
          - If left-click near exactly one point => start drag
          - If right-click near exactly one point => console-based weight editing (if NURBS).
            If multiple or no points found, skip to avoid glitch.
        """
        if event.inaxes != self.ax:
            return
        
        # Check if we are over any control points
        cont, ind = self.scat.contains(event)
        if not cont:
            return
        
        indices = ind["ind"]
        if len(indices) == 0:
            # No CP found under the cursor
            return
        
        # If the click covers multiple CPs, pick the first or skip?
        # We'll skip if multiple, to avoid confusion:
        distances = []
        for i in indices:
            cp = self.control_points[i]
            distances.append(np.hypot(cp[0] - event.xdata, cp[1] - event.ydata))
        nearest_idx = indices[np.argmin(distances)]
        idx = nearest_idx
        
        idx = indices[0]
        
        # Distinguish left-click vs right-click
        # NOTE: On some systems, right-click might be event.button == 2
        # Adjust if you see the wrong behavior
        if event.button == 1:
            # Left-click => drag
            self._dragging_idx = idx
        if event.button == 3 and self.use_nurbs:
            current_w = self.weights[idx]
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            new_w = simpledialog.askstring("Edit Weight", f"CP {idx}, current weight={current_w:.3f}\nEnter new weight:")
            root.destroy()
            if new_w is not None and new_w != "":
                try:
                    self.weights[idx] = float(new_w)
                    print(f"Updated weight of CP {idx} to {new_w}")
                    self.update_curve()
                except ValueError:
                    print("Invalid input. No change made.")

    def on_release(self, event):
        """
        Stop dragging on mouse release.
        """
        self._dragging_idx = None

    def on_motion(self, event):
        """
        If dragging is active, move the selected control point.
        """
        if self._dragging_idx is None:
            return
        if event.inaxes != self.ax:
            return
        
        x_new = event.xdata
        y_new = event.ydata
        if x_new is not None and y_new is not None:
            self.control_points[self._dragging_idx, 0] = x_new
            self.control_points[self._dragging_idx, 1] = y_new
            self.scat.set_offsets(self.control_points)
            self.update_curve()

###############################################
# 3) Main: Example usage
###############################################
def main():
    """
    - Left-click + drag to move CP.
    - Right-click CP => console-based weight editing (only if use_nurbs = True).
    - If multiple/no CP under right-click, skip => prevents glitching.
    - After closing figure, prompt to save changes.
    """
    # User config
    csv_path = "test_outputsC1C20.50.5_Air/NURBS_params.csv"
    test_npy_path = "test.npy"
    save_output_csv = "test_outputsC1C20.50.5_Air/NURBS_params_edited.csv"
    
    sample_idx_to_edit = 3
    gt_idx = 3
    
    use_nurbs = True
    bspline_degree = 3
    curve_num_points = 192
    
    # 1) Load ground truth from test.npy
    if not os.path.isfile(test_npy_path):
        print(f"File not found: {test_npy_path}")
        return
    test_data = np.load(test_npy_path)  # shape [N, 192, 2]
    if not (0 <= gt_idx < test_data.shape[0]):
        print(f"Invalid gt_idx {gt_idx} for test data of shape {test_data.shape}")
        return
    gt_curve_2d = test_data[gt_idx]  # (192, 2)
    gt_curve = gt_curve_2d.T         # (2, 192)
    
    # 2) Load param CSV
    if not os.path.isfile(csv_path):
        print(f"CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    if use_nurbs:
        df_sample = df[df["sample_idx"] == sample_idx_to_edit].copy()
        df_sample.sort_values("control_point", inplace=True)
        if len(df_sample) == 0:
            print(f"No data for sample_idx={sample_idx_to_edit}")
            return
        control_points = df_sample[["x", "y"]].to_numpy()
        weights = df_sample["weight"].to_numpy()
    else:
        df_sample = df[df["sample_idx"] == sample_idx_to_edit].copy()
        df_sample.sort_values("control_point", inplace=True)
        if len(df_sample) == 0:
            print(f"No data for sample_idx={sample_idx_to_edit}")
            return
        control_points = df_sample[["x", "y"]].to_numpy()
        weights = None
    
    # 3) Create the figure and DraggableCurvePlot
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_title(f"Sample {sample_idx_to_edit} (GT idx={gt_idx})")
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    draggable = DraggableCurvePlot(
        ax=ax,
        control_points=control_points,
        gt_curve=gt_curve,
        use_nurbs=use_nurbs,
        weights=weights,
        degree=bspline_degree,
        num_points=curve_num_points
    )
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # 4) Prompt user to save changes after closing the figure
    user_choice = input("Save updated control points (and weights if NURBS)? (y/n): ").strip().lower()
    if user_choice == 'y':
        df_sample.loc[:, 'x'] = draggable.control_points[:, 0]
        df_sample.loc[:, 'y'] = draggable.control_points[:, 1]
        if use_nurbs:
            df_sample.loc[:, 'weight'] = draggable.weights
        
        if os.path.exists(save_output_csv):
            # Remove old rows for this sample_idx
            df_existing = pd.read_csv(save_output_csv)
            df_existing = df_existing[df_existing["sample_idx"] != sample_idx_to_edit]
            df_updated = pd.concat([df_existing, df_sample], ignore_index=True)
            df_updated.to_csv(save_output_csv, index=False)
        else:
            df_sample.to_csv(save_output_csv, index=False)
        
        print(f"Updated params saved to {save_output_csv}")
    else:
        print("No changes saved.")

if __name__ == '__main__':
    main()
