# Triple‑Parametric Autoencoder (TPDLF): 2D Curve Reparameterization via Bézier, B‑spline, and NURBS

## About
This repository provides a reference implementation of a **triple‑parametric deep learning framework (TPDLF)** that takes a discretized 2D curve and **reparameterizes** it into three classical spline forms—**Bézier**, **B‑spline**, and **NURBS**—in a single model. A shared encoder compresses the input curve; three light decoders infer:
- Bézier control points
- B‑spline control points
- NURBS control points + **positive** weights (via softplus)

Closed‑form geometric layers reconstruct smooth curves directly from these parameters, allowing **interpretable** design variables that transfer cleanly to CAD/CAE and downstream editing.

---

## Publication
**Computer‑Aided Design (CAD)**  
**Triple‑parametric autoencoder for 2D reparameterization via Bézier, B‑spline, and NURBS representations**  
Shaoliang Yang, Jun Wang (2025), Article 103936.  
**Publisher page:** https://www.sciencedirect.com/science/article/pii/S0010448525000971  
**DOI:** https://doi.org/10.1016/j.cad.2025.103936

> If you use this project in academic work, please cite the paper (BibTeX below).

---

## Abstract (concise)
We present a **single autoencoder** that maps a sampled 2D curve to three spline parameterizations. A shared encoder learns a latent embedding; three decoders output Bézier/B‑spline/NURBS parameters (control points and, for NURBS, non‑negative weights). Analytical curve layers reconstruct smooth curves directly from these parameters, enabling **interpretable** design variables and easy post‑editing. Experiments on airfoils and Superformula‑like shapes show accurate reconstruction and complementary strengths of the three representations.

---

## Quick Start

### 1) Environment
```bash
# Python >= 3.9
pip install torch numpy matplotlib pandas
```

### 2) Minimal inference (import from `train.py`)
`train.py` already contains the model definitions and helpers. The model expects a curve of shape **[B, 2, N]** with `N = sample_points` (default 192).

```python
import torch, numpy as np, matplotlib.pyplot as plt
from train import AirfoilAutoencoder  # models live in train.py

device = "cuda" if torch.cuda.is_available() else "cpu"

# Build a toy closed curve (unit circle) with 192 samples
N = 192
theta = np.linspace(0, 2*np.pi, N, endpoint=False)
X = np.stack([np.cos(theta), np.sin(theta)], axis=0)          # [2, N]
curve = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 2, N]

# Initialize model (defaults shown)
model = AirfoilAutoencoder(
    num_control_points=10,   # try 10~64 for fidelity/runtime trade‑off
    degree=3,
    noise_dim=32,
    latent_dim=128,
    weights_enabled=True,
    sample_points=N
).to(device).eval()

# Forward pass
noise = torch.zeros(1, model.noise_dim, device=device)
bspline, nurbs, bezier = model(curve, noise)   # each: [1, 2, N]

# Visualize
def show(name, arr):
    xy = arr.squeeze(0).detach().cpu().numpy()
    plt.figure(); plt.plot(xy[0], xy[1]); plt.axis("equal"); plt.title(name)

show("Input", curve)
show("B‑spline", bspline)
show("NURBS", nurbs)
show("Bézier", bezier)
plt.show()
```

### 3) Train with your data
`train.py` also includes a `AirfoilDataset` class and a `train(...)` function. The dataset expects arrays shaped **[N_samples, 192, 2]** (note the last two axes are `(samples_along_curve, xy)`).

```python
import numpy as np, torch
from torch.utils.data import DataLoader
from train import AirfoilAutoencoder, AirfoilDataset, train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your .npy files (shape: [N, 192, 2])
train_data = torch.tensor(np.load("data/train.npy"), dtype=torch.float32)
test_data  = torch.tensor(np.load("data/test.npy"),  dtype=torch.float32)

# Datasets & loaders
train_dataset = AirfoilDataset(train_data, augment=True)
test_dataset  = AirfoilDataset(test_data,  augment=False)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# Model
model = AirfoilAutoencoder(num_control_points=10, degree=3, noise_dim=32, latent_dim=128,
                           weights_enabled=True, sample_points=192).to(device)

# Train
train(model, train_loader, test_loader,
      num_epochs=20, lr=1e-3, noise_dim=32, patience=5,
      save_path="best_model.pth", device=device, test_dataset=test_dataset,
      geo_loss_weight=0.0)  # set smoothing/C1/C2 weights >0.0 to enable those losses
```



## Suggested repository layout
```
.
├── README.md
├── train.py        
└── testMetric.py
└── ADJUST.py  
├── data/
│   ├── train.npy   # shape [N, 192, 2]
│   └── test.npy    # shape [M, 192, 2]
```

> **Tips**
> - Normalize/center your curves consistently before training/evaluation.
> - To enforce closed curves, the code ties the last control point to the first.
> - Increase `num_control_points` for higher fidelity; cubic (`degree=3`) works well in most cases.

---

## License (MIT)
This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2025 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Cite (Google Scholar BibTeX)
```bibtex
@article{yang2025triple,
  title   = {Triple-parametric autoencoder for 2D reparameterization via B{\'e}zier, B-spline, and NURBS representations},
  author  = {Yang, Shaoliang and Wang, Jun},
  journal = {Computer-Aided Design},
  volume  = {189},
  pages   = {103936},
  year    = {2025},
  publisher = {Elsevier},
  doi     = {10.1016/j.cad.2025.103936}
}
```

---

## Links
- **CAD article page:** https://www.sciencedirect.com/science/article/pii/S0010448525000971
- **DOI:** https://doi.org/10.1016/j.cad.2025.103936
