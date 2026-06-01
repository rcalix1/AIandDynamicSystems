## Cassimir


* Points

```python

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)

# ------------------------------------
# Casimir Force
# F = K / a^4
# ------------------------------------

K = 1.0

def casimir_force(a):
    return K / (a**4)

# ------------------------------------
# NIO
# ------------------------------------

z = nn.Parameter(torch.tensor([0.0], device=device))

optimizer = torch.optim.Adam([z], lr=0.05)

history = []

for step in range(500):

    optimizer.zero_grad()

    a = 0.1 + 2.9*torch.sigmoid(z)

    force = casimir_force(a)

    loss = -force

    loss.backward()

    optimizer.step()

    history.append(force.item())

# ------------------------------------
# Result
# ------------------------------------

best_a = (0.1 + 2.9*torch.sigmoid(z)).item()

print("Best spacing:", best_a)
print("Force:", history[-1])

# ------------------------------------
# Plot convergence
# ------------------------------------

plt.figure(figsize=(6,4))
plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("Casimir Force")
plt.title("Paper A: NIO discovers important spacing")
plt.show()

# ------------------------------------
# Compare to random search
# ------------------------------------

random_scores = []

for _ in range(10000):

    a = np.random.uniform(0.1,3.0)

    random_scores.append(
        K/(a**4)
    )

print()
print("Random Best:", max(random_scores))
print("NIO Best   :", history[-1])
print("Improvement:", history[-1]/max(random_scores))

```

* structure


```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)

# ------------------------------------
# Ground Truth Modes
# ------------------------------------

c = 1.0
a = 1.0

N_MODES = 20

true_modes = torch.tensor(
    [n*c/(2*a)
     for n in range(1,N_MODES+1)],
    dtype=torch.float32
)

# ------------------------------------
# Discover Modes
# ------------------------------------

discovered = []

for n in range(1,N_MODES+1):

    z = nn.Parameter(
        torch.randn(1)
    )

    optimizer = torch.optim.Adam(
        [z],
        lr=0.05
    )

    target = true_modes[n-1]

    for step in range(500):

        optimizer.zero_grad()

        mode = (
            0.1
            +
            20*torch.sigmoid(z)
        )

        loss = (mode-target)**2

        loss.backward()

        optimizer.step()

    discovered.append(
        mode.item()
    )

discovered = np.array(discovered)

# ------------------------------------
# Metrics
# ------------------------------------

mse = np.mean(
    (discovered -
     true_modes.numpy())**2
)

print("Mode MSE:", mse)

# ------------------------------------
# Plot
# ------------------------------------

plt.figure(figsize=(8,4))

plt.plot(
    true_modes.numpy(),
    label="True"
)

plt.plot(
    discovered,
    '--',
    label="NIO"
)

plt.xlabel("Mode Index")
plt.ylabel("Frequency")

plt.title(
    "Paper B: Recovery of Casimir Mode Structure"
)

plt.legend()

plt.show()

# ------------------------------------
# Spectrum Overlay
# ------------------------------------

plt.figure(figsize=(8,4))

plt.vlines(
    true_modes.numpy(),
    0,
    1,
    label="True"
)

plt.vlines(
    discovered,
    0,
    0.8,
    linestyles='dashed',
    label="Discovered"
)

plt.legend()

plt.title(
    "Vacuum Mode Spectrum"
)

plt.show()
```



better structure 

```python
# ==========================================================
# PAPER B
# Casimir Structure Discovery via NIO
#
# Goal:
#
# Generate vacuum-mode spectra from plate spacings.
#
# Hide the spacing.
#
# Train a density model on spectra.
#
# Use NIO to discover spectra that belong
# to the hidden vacuum-mode manifold.
#
# Compare discovered manifold against truth.
#
# ==========================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
np.random.seed(0)

# ==========================================================
# STEP 1
# Generate synthetic Casimir observations
# ==========================================================

C = 1.0
N_MODES = 10
N_SAMPLES = 5000

def generate_modes(a):

    modes = []

    for n in range(1, N_MODES+1):

        f = n*C/(2*a)

        modes.append(f)

    return np.array(modes)

spectra = []

spacings = np.random.uniform(
    0.2,
    3.0,
    N_SAMPLES
)

for a in spacings:

    s = generate_modes(a)

    s += np.random.normal(
        0,
        0.01,
        N_MODES
    )

    spectra.append(s)

spectra = np.array(spectra)

print("Spectra shape:", spectra.shape)

# ==========================================================
# STEP 2
# Generate negatives
# ==========================================================

mins = spectra.min(axis=0)
maxs = spectra.max(axis=0)

negative = np.random.uniform(
    mins,
    maxs,
    size=spectra.shape
)

X = np.vstack([
    spectra,
    negative
])

y = np.concatenate([
    np.ones(len(spectra)),
    np.zeros(len(negative))
])

# ==========================================================
# STEP 3
# Torch tensors
# ==========================================================

X = torch.tensor(
    X,
    dtype=torch.float32,
    device=device
)

y = torch.tensor(
    y.reshape(-1,1),
    dtype=torch.float32,
    device=device
)

# ==========================================================
# STEP 4
# Density model
# ==========================================================

model = nn.Sequential(

    nn.Linear(N_MODES,128),
    nn.ReLU(),

    nn.Linear(128,128),
    nn.ReLU(),

    nn.Linear(128,64),
    nn.ReLU(),

    nn.Linear(64,1),
    nn.Sigmoid()

).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.BCELoss()

loss_history = []

for epoch in range(300):

    optimizer.zero_grad()

    pred = model(X)

    loss = criterion(pred,y)

    loss.backward()

    optimizer.step()

    loss_history.append(
        loss.item()
    )

    if epoch % 50 == 0:
        print(epoch, loss.item())

# ==========================================================
# Plot training
# ==========================================================

plt.figure(figsize=(6,4))
plt.plot(loss_history)
plt.title("Density Model Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# ==========================================================
# STEP 5
# NIO structure discovery
#
# Find spectra that maximize
# probability of belonging
# to hidden manifold.
# ==========================================================

discovered = []

for trial in range(2000):

    z = nn.Parameter(

        torch.tensor(
            np.random.uniform(
                mins,
                maxs
            ),
            dtype=torch.float32,
            device=device
        )

    )

    opt = torch.optim.Adam(
        [z],
        lr=0.05
    )

    for step in range(150):

        opt.zero_grad()

        score = model(
            z.unsqueeze(0)
        )

        loss = -score.mean()

        loss.backward()

        opt.step()

    discovered.append(
        z.detach().cpu().numpy()
    )

discovered = np.array(discovered)

# ==========================================================
# STEP 6
# Compare discovered structure
# ==========================================================

plt.figure(figsize=(8,6))

for i in range(100):

    plt.plot(
        spectra[i],
        alpha=0.05,
        color='blue'
    )

for i in range(100):

    plt.plot(
        discovered[i],
        alpha=0.05,
        color='red'
    )

plt.title(
    "Blue=True Vacuum Mode Manifold\nRed=NIO Discovered Structure"
)

plt.xlabel("Mode Index")
plt.ylabel("Frequency")

plt.show()

# ==========================================================
# STEP 7
# Reconstruction Metric
# ==========================================================

true_mean = spectra.mean(axis=0)

disc_mean = discovered.mean(axis=0)

mse = np.mean(
    (true_mean-disc_mean)**2
)

print()
print("Mean Spectrum MSE:", mse)

# ==========================================================
# STEP 8
# PCA Visualization
# ==========================================================

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

combined = np.vstack([
    spectra,
    discovered
])

proj = pca.fit_transform(combined)

n_true = len(spectra)

plt.figure(figsize=(7,6))

plt.scatter(
    proj[:n_true,0],
    proj[:n_true,1],
    s=5,
    alpha=0.3,
    label="True Structure"
)

plt.scatter(
    proj[n_true:,0],
    proj[n_true:,1],
    s=5,
    alpha=0.3,
    label="NIO Structure"
)

plt.legend()

plt.title(
    "Vacuum Mode Manifold Discovery"
)

plt.show()

print()
print("Finished.")
```



even better structure casimir 

```python id="rk0j3n"
# ==========================================================
# PAPER B
# CASIMIR VACUUM MODE MANIFOLD DISCOVERY
#
# Hidden Structure:
# Vacuum Mode Manifold
#
# Geometry
#     ↓
# Allowed Modes
#     ↓
# Observations
#
# AI only sees spectra.
#
# ==========================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
np.random.seed(0)

# ==========================================================
# PARAMETERS
# ==========================================================

N_MODES   = 12
N_SAMPLES = 5000

# ==========================================================
# VACUUM MODE GENERATOR
#
# Parallel Plates
#
# f_n = n*c/(2a)
#
# AI never sees:
#
# a
#
# ==========================================================

C = 1.0

def vacuum_modes(a):

    modes = []

    for n in range(1,N_MODES+1):

        f = n*C/(2*a)

        modes.append(f)

    return np.array(modes)

# ==========================================================
# GENERATE OBSERVATIONS
# ==========================================================

spectra = []

plate_spacings = np.random.uniform(
    0.2,
    5.0,
    N_SAMPLES
)

for a in plate_spacings:

    s = vacuum_modes(a)

    # measurement noise

    s += np.random.normal(
        0,
        0.01*np.mean(s),
        N_MODES
    )

    spectra.append(s)

spectra = np.array(
    spectra
)

print(
    "Spectra shape:",
    spectra.shape
)

# ==========================================================
# NEGATIVE SAMPLES
#
# Random spectra
# ==========================================================

mins = spectra.min(axis=0)
maxs = spectra.max(axis=0)

negative = np.random.uniform(
    mins,
    maxs,
    size=spectra.shape
)

X = np.vstack([
    spectra,
    negative
])

y = np.concatenate([
    np.ones(len(spectra)),
    np.zeros(len(negative))
])

# ==========================================================
# TORCH
# ==========================================================

X = torch.tensor(
    X,
    dtype=torch.float32,
    device=device
)

y = torch.tensor(
    y.reshape(-1,1),
    dtype=torch.float32,
    device=device
)

# ==========================================================
# DENSITY MODEL
# ==========================================================

model = nn.Sequential(

    nn.Linear(
        N_MODES,
        128
    ),
    nn.ReLU(),

    nn.Linear(
        128,
        128
    ),
    nn.ReLU(),

    nn.Linear(
        128,
        64
    ),
    nn.ReLU(),

    nn.Linear(
        64,
        1
    ),
    nn.Sigmoid()

).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.BCELoss()

history = []

# ==========================================================
# TRAIN
# ==========================================================

for epoch in range(300):

    optimizer.zero_grad()

    pred = model(X)

    loss = criterion(
        pred,
        y
    )

    loss.backward()

    optimizer.step()

    history.append(
        loss.item()
    )

    if epoch % 50 == 0:

        print(
            epoch,
            loss.item()
        )

# ==========================================================
# TRAINING CURVE
# ==========================================================

plt.figure(figsize=(6,4))

plt.plot(history)

plt.title(
    "Density Model Training"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()

# ==========================================================
# NIO STRUCTURE DISCOVERY
#
# Discover spectra
# belonging to hidden
# vacuum manifold
# ==========================================================

discovered = []

for trial in range(2000):

    z = nn.Parameter(

        torch.tensor(
            np.random.uniform(
                mins,
                maxs
            ),
            dtype=torch.float32,
            device=device
        )

    )

    opt = torch.optim.Adam(
        [z],
        lr=0.05
    )

    for step in range(150):

        opt.zero_grad()

        score = model(
            z.unsqueeze(0)
        )

        loss = -score.mean()

        loss.backward()

        opt.step()

    discovered.append(
        z.detach().cpu().numpy()
    )

discovered = np.array(
    discovered
)

# ==========================================================
# STRUCTURE VISUALIZATION
# ==========================================================

plt.figure(figsize=(8,6))

for i in range(100):

    plt.plot(
        spectra[i],
        alpha=0.05,
        color="blue"
    )

for i in range(100):

    plt.plot(
        discovered[i],
        alpha=0.05,
        color="red"
    )

plt.title(
    "Blue=True Vacuum Mode Manifold\nRed=NIO Structure"
)

plt.xlabel(
    "Mode Index"
)

plt.ylabel(
    "Frequency"
)

plt.show()

# ==========================================================
# RECONSTRUCTION ERROR
# ==========================================================

true_mean = spectra.mean(
    axis=0
)

disc_mean = discovered.mean(
    axis=0
)

mse = np.mean(
    (true_mean-disc_mean)**2
)

print()
print(
    "Mean Spectrum MSE:",
    mse
)

# ==========================================================
# PCA VISUALIZATION
# ==========================================================

combined = np.vstack([
    spectra,
    discovered
])

pca = PCA(
    n_components=2
)

proj = pca.fit_transform(
    combined
)

n_true = len(spectra)

plt.figure(
    figsize=(7,6)
)

plt.scatter(
    proj[:n_true,0],
    proj[:n_true,1],
    s=5,
    alpha=0.3,
    label="True Structure"
)

plt.scatter(
    proj[n_true:,0],
    proj[n_true:,1],
    s=5,
    alpha=0.3,
    label="NIO Structure"
)

plt.legend()

plt.title(
    "Vacuum Mode Manifold Discovery"
)

plt.show()

# ==========================================================
# OPTIONAL
# RECOVER IMPLIED SPACING
# ==========================================================

implied_spacing = []

for s in discovered:

    f1 = s[0]

    a_est = C/(2*f1)

    implied_spacing.append(
        a_est
    )

implied_spacing = np.array(
    implied_spacing
)

print()
print(
    "Recovered spacing range:"
)

print(
    implied_spacing.min(),
    implied_spacing.max()
)

print()
print("Finished.")
```



## Discovering the Lorenz Attractor from Data using Neural Input Optimization


```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)

# ==========================================================
# STEP 1
# Generate Lorenz Data
# (After this step we pretend the equations are unknown)
# ==========================================================

sigma = 10.0
rho   = 28.0
beta  = 8.0/3.0

def lorenz_step(x):

    dx = sigma*(x[:,1]-x[:,0])

    dy = x[:,0]*(rho-x[:,2]) - x[:,1]

    dz = x[:,0]*x[:,1] - beta*x[:,2]

    return torch.stack([dx,dy,dz],dim=1)

def simulate_lorenz(
        x0,
        n_steps=5000,
        dt=0.01):

    x = x0.clone()

    traj = []

    for _ in range(n_steps):

        x = x + dt*lorenz_step(x)

        traj.append(x.clone())

    return torch.cat(traj,dim=0)

x0 = torch.tensor(
    [[1.0,1.0,1.0]],
    device=device
)

traj = simulate_lorenz(x0)

lorenz_data = traj.detach()

print("Data shape:",lorenz_data.shape)

# ==========================================================
# STEP 2
# Train Density Network
#
# Positive samples:
#     Lorenz trajectory
#
# Negative samples:
#     Random points in space
#
# Network learns:
#
#     f(x,y,z)
#
# probability point belongs
# to attractor
# ==========================================================

mins = lorenz_data.min(dim=0)[0]
maxs = lorenz_data.max(dim=0)[0]

n_real = len(lorenz_data)

random_points = (
    mins
    +
    (maxs-mins)
    *
    torch.rand(
        n_real,
        3,
        device=device
    )
)

X = torch.cat([
    lorenz_data,
    random_points
])

y = torch.cat([
    torch.ones(n_real,1,device=device),
    torch.zeros(n_real,1,device=device)
])

perm = torch.randperm(len(X))

X = X[perm]
y = y[perm]

# ==========================================================
# Density Model
# ==========================================================

model = nn.Sequential(

    nn.Linear(3,64),
    nn.ReLU(),

    nn.Linear(64,64),
    nn.ReLU(),

    nn.Linear(64,1),
    nn.Sigmoid()

).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.BCELoss()

loss_history = []

for epoch in range(300):

    optimizer.zero_grad()

    pred = model(X)

    loss = criterion(pred,y)

    loss.backward()

    optimizer.step()

    loss_history.append(
        loss.item()
    )

    if epoch % 50 == 0:
        print(
            epoch,
            loss.item()
        )

# ==========================================================
# Plot Training Loss
# ==========================================================

plt.figure(figsize=(6,4))

plt.plot(loss_history)

plt.title(
    "Density Model Training"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()

# ==========================================================
# STEP 3
# NIO Structure Discovery
#
# Optimize point x
#
# maximize density score
#
# score = model(x)
#
# Repeated many times
#
# produces attractor cloud
# ==========================================================

discovered_points = []

for trial in range(2000):

    z = nn.Parameter(

        mins
        +
        (maxs-mins)
        *
        torch.rand(
            3,
            device=device
        )

    )

    opt = torch.optim.Adam(
        [z],
        lr=0.05
    )

    for step in range(150):

        opt.zero_grad()

        score = model(
            z.unsqueeze(0)
        )

        loss = -score.mean()

        loss.backward()

        opt.step()

    discovered_points.append(
        z.detach().cpu().numpy()
    )

discovered_points = np.array(
    discovered_points
)

# ==========================================================
# STEP 4
# Visual Comparison
# ==========================================================

sample_idx = np.random.choice(
    len(lorenz_data),
    5000,
    replace=False
)

true_points = (
    lorenz_data[
        sample_idx
    ]
    .cpu()
    .numpy()
)

fig = plt.figure(
    figsize=(12,5)
)

ax1 = fig.add_subplot(
    121,
    projection='3d'
)

ax1.scatter(
    true_points[:,0],
    true_points[:,1],
    true_points[:,2],
    s=1
)

ax1.set_title(
    "Ground Truth Attractor"
)

ax2 = fig.add_subplot(
    122,
    projection='3d'
)

ax2.scatter(
    discovered_points[:,0],
    discovered_points[:,1],
    discovered_points[:,2],
    s=3
)

ax2.set_title(
    "NIO Discovered Structure"
)

plt.show()

# ==========================================================
# STEP 5
# Simple Coverage Metric
# ==========================================================

true_center = np.mean(
    true_points,
    axis=0
)

discovered_center = np.mean(
    discovered_points,
    axis=0
)

center_distance = np.linalg.norm(
    true_center -
    discovered_center
)

print()
print(
    "Center Distance:",
    center_distance
)

# ==========================================================
# Optional:
# Density Heatmap
# ==========================================================

scores = []

for p in discovered_points:

    p_t = torch.tensor(
        p,
        dtype=torch.float32,
        device=device
    )

    score = model(
        p_t.unsqueeze(0)
    )

    scores.append(
        score.item()
    )

scores = np.array(scores)

print(
    "Average Density Score:",
    scores.mean()
)
```


## String



structure


```python
# ==========================================================
# PAPER B
# STRING MODE STRUCTURE DISCOVERY VIA NIO
#
# Hidden Structure:
# Harmonic Manifold
#
# Analogous to:
#
# Lorenz -> Butterfly Attractor
#
# String -> Harmonic Mode Manifold
#
# ==========================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
np.random.seed(0)

# ==========================================================
# STEP 1
# Generate Observations
#
# Physics:
#
# f_n = n/(2L)*sqrt(T/mu)
#
# AI never sees equation.
# ==========================================================

N_MODES = 10
N_SAMPLES = 5000

spectra = []

lengths = np.random.uniform(
    0.5,
    5.0,
    N_SAMPLES
)

tensions = np.random.uniform(
    20.0,
    200.0,
    N_SAMPLES
)

densities = np.random.uniform(
    0.01,
    0.10,
    N_SAMPLES
)

for L,T,mu in zip(
        lengths,
        tensions,
        densities):

    base = (1.0/(2.0*L))*np.sqrt(T/mu)

    spectrum = []

    for n in range(1,N_MODES+1):

        spectrum.append(
            n*base
        )

    spectrum = np.array(spectrum)

    spectrum += np.random.normal(
        0,
        0.01*np.mean(spectrum),
        N_MODES
    )

    spectra.append(
        spectrum
    )

spectra = np.array(spectra)

print(
    "Spectra shape:",
    spectra.shape
)

# ==========================================================
# STEP 2
# Negatives
#
# Random spectra
# ==========================================================

mins = spectra.min(axis=0)
maxs = spectra.max(axis=0)

negative = np.random.uniform(
    mins,
    maxs,
    size=spectra.shape
)

X = np.vstack([
    spectra,
    negative
])

y = np.concatenate([
    np.ones(len(spectra)),
    np.zeros(len(negative))
])

# ==========================================================
# STEP 3
# Torch
# ==========================================================

X = torch.tensor(
    X,
    dtype=torch.float32,
    device=device
)

y = torch.tensor(
    y.reshape(-1,1),
    dtype=torch.float32,
    device=device
)

# ==========================================================
# STEP 4
# Density Network
# ==========================================================

model = nn.Sequential(

    nn.Linear(
        N_MODES,
        128
    ),
    nn.ReLU(),

    nn.Linear(
        128,
        128
    ),
    nn.ReLU(),

    nn.Linear(
        128,
        64
    ),
    nn.ReLU(),

    nn.Linear(
        64,
        1
    ),
    nn.Sigmoid()

).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.BCELoss()

loss_history = []

for epoch in range(300):

    optimizer.zero_grad()

    pred = model(X)

    loss = criterion(
        pred,
        y
    )

    loss.backward()

    optimizer.step()

    loss_history.append(
        loss.item()
    )

    if epoch % 50 == 0:

        print(
            epoch,
            loss.item()
        )

# ==========================================================
# Training Curve
# ==========================================================

plt.figure(figsize=(6,4))

plt.plot(
    loss_history
)

plt.title(
    "Density Model Training"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.show()

# ==========================================================
# STEP 5
# NIO Structure Discovery
#
# Discover spectra
# belonging to hidden manifold
# ==========================================================

discovered = []

for trial in range(2000):

    z = nn.Parameter(

        torch.tensor(
            np.random.uniform(
                mins,
                maxs
            ),
            dtype=torch.float32,
            device=device
        )

    )

    opt = torch.optim.Adam(
        [z],
        lr=0.05
    )

    for step in range(150):

        opt.zero_grad()

        score = model(
            z.unsqueeze(0)
        )

        loss = -score.mean()

        loss.backward()

        opt.step()

    discovered.append(
        z.detach().cpu().numpy()
    )

discovered = np.array(
    discovered
)

# ==========================================================
# STEP 6
# Compare Manifolds
# ==========================================================

plt.figure(
    figsize=(8,6)
)

for i in range(100):

    plt.plot(
        spectra[i],
        alpha=0.05,
        color="blue"
    )

for i in range(100):

    plt.plot(
        discovered[i],
        alpha=0.05,
        color="red"
    )

plt.title(
    "Blue=True Harmonic Manifold\nRed=NIO Structure"
)

plt.xlabel(
    "Mode Index"
)

plt.ylabel(
    "Frequency"
)

plt.show()

# ==========================================================
# STEP 7
# Mean Spectrum Error
# ==========================================================

true_mean = spectra.mean(
    axis=0
)

disc_mean = discovered.mean(
    axis=0
)

mse = np.mean(
    (true_mean-disc_mean)**2
)

print()
print(
    "Mean Spectrum MSE:",
    mse
)

# ==========================================================
# STEP 8
# PCA Visualization
# ==========================================================

from sklearn.decomposition import PCA

combined = np.vstack([
    spectra,
    discovered
])

pca = PCA(
    n_components=2
)

proj = pca.fit_transform(
    combined
)

n_true = len(spectra)

plt.figure(
    figsize=(7,6)
)

plt.scatter(
    proj[:n_true,0],
    proj[:n_true,1],
    s=5,
    alpha=0.3,
    label="True Structure"
)

plt.scatter(
    proj[n_true:,0],
    proj[n_true:,1],
    s=5,
    alpha=0.3,
    label="NIO Structure"
)

plt.legend()

plt.title(
    "Harmonic Manifold Discovery"
)

plt.show()

print()
print("Finished.")
```




## EM Cavity



structure 


```python
# ==========================================================
# PAPER B
# EM CAVITY MODE STRUCTURE DISCOVERY
#
# Hidden Structure:
# Electromagnetic Mode Manifold
#
# Physics:
#
# f_mnp =
# c/2 * sqrt(
# (m/a)^2 +
# (n/b)^2 +
# (p/c)^2
# )
#
# AI only sees spectra.
#
# ==========================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
np.random.seed(0)

# ==========================================================
# PARAMETERS
# ==========================================================

N_MODES = 12
N_SAMPLES = 5000

# ==========================================================
# MODE GENERATOR
# ==========================================================

def cavity_modes(ax, ay, az):

    modes = []

    for m in range(1,4):
        for n in range(1,4):
            for p in range(1,4):

                f = np.sqrt(
                    (m/ax)**2 +
                    (n/ay)**2 +
                    (p/az)**2
                )

                modes.append(f)

    modes = np.sort(modes)

    return modes[:N_MODES]

# ==========================================================
# GENERATE OBSERVATIONS
# ==========================================================

spectra = []

for _ in range(N_SAMPLES):

    ax = np.random.uniform(
        0.5,
        5.0
    )

    ay = np.random.uniform(
        0.5,
        5.0
    )

    az = np.random.uniform(
        0.5,
        5.0
    )

    s = cavity_modes(
        ax,
        ay,
        az
    )

    s += np.random.normal(
        0,
        0.01*np.mean(s),
        len(s)
    )

    spectra.append(s)

spectra = np.array(
    spectra
)

print(
    "Spectra shape:",
    spectra.shape
)

# ==========================================================
# NEGATIVES
# ==========================================================

mins = spectra.min(axis=0)
maxs = spectra.max(axis=0)

negative = np.random.uniform(
    mins,
    maxs,
    size=spectra.shape
)

X = np.vstack([
    spectra,
    negative
])

y = np.concatenate([
    np.ones(len(spectra)),
    np.zeros(len(negative))
])

# ==========================================================
# TORCH
# ==========================================================

X = torch.tensor(
    X,
    dtype=torch.float32,
    device=device
)

y = torch.tensor(
    y.reshape(-1,1),
    dtype=torch.float32,
    device=device
)

# ==========================================================
# DENSITY MODEL
# ==========================================================

model = nn.Sequential(

    nn.Linear(
        N_MODES,
        128
    ),
    nn.ReLU(),

    nn.Linear(
        128,
        128
    ),
    nn.ReLU(),

    nn.Linear(
        128,
        64
    ),
    nn.ReLU(),

    nn.Linear(
        64,
        1
    ),
    nn.Sigmoid()

).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.BCELoss()

history = []

# ==========================================================
# TRAIN
# ==========================================================

for epoch in range(300):

    optimizer.zero_grad()

    pred = model(X)

    loss = criterion(
        pred,
        y
    )

    loss.backward()

    optimizer.step()

    history.append(
        loss.item()
    )

    if epoch % 50 == 0:

        print(
            epoch,
            loss.item()
        )

# ==========================================================
# LOSS CURVE
# ==========================================================

plt.figure(figsize=(6,4))

plt.plot(history)

plt.title(
    "Density Model Training"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()

# ==========================================================
# NIO DISCOVERY
# ==========================================================

discovered = []

for trial in range(2000):

    z = nn.Parameter(

        torch.tensor(
            np.random.uniform(
                mins,
                maxs
            ),
            dtype=torch.float32,
            device=device
        )

    )

    opt = torch.optim.Adam(
        [z],
        lr=0.05
    )

    for step in range(150):

        opt.zero_grad()

        score = model(
            z.unsqueeze(0)
        )

        loss = -score.mean()

        loss.backward()

        opt.step()

    discovered.append(
        z.detach().cpu().numpy()
    )

discovered = np.array(
    discovered
)

# ==========================================================
# STRUCTURE VISUALIZATION
# ==========================================================

plt.figure(figsize=(8,6))

for i in range(100):

    plt.plot(
        spectra[i],
        alpha=0.05,
        color="blue"
    )

for i in range(100):

    plt.plot(
        discovered[i],
        alpha=0.05,
        color="red"
    )

plt.title(
    "Blue=True EM Mode Manifold\nRed=NIO Structure"
)

plt.xlabel(
    "Mode Index"
)

plt.ylabel(
    "Frequency"
)

plt.show()

# ==========================================================
# MSE METRIC
# ==========================================================

true_mean = spectra.mean(
    axis=0
)

disc_mean = discovered.mean(
    axis=0
)

mse = np.mean(
    (true_mean-disc_mean)**2
)

print()
print(
    "Mean Spectrum MSE:",
    mse
)

# ==========================================================
# PCA VIEW
# ==========================================================

combined = np.vstack([
    spectra,
    discovered
])

pca = PCA(
    n_components=2
)

proj = pca.fit_transform(
    combined
)

n_true = len(spectra)

plt.figure(
    figsize=(7,6)
)

plt.scatter(
    proj[:n_true,0],
    proj[:n_true,1],
    s=5,
    alpha=0.3,
    label="True Structure"
)

plt.scatter(
    proj[n_true:,0],
    proj[n_true:,1],
    s=5,
    alpha=0.3,
    label="NIO Structure"
)

plt.legend()

plt.title(
    "EM Mode Manifold Discovery"
)

plt.show()

print()
print("Finished.")
```














