## Readings

* Chaos by James Gleick - https://principus.si/2026/04/06/james-gleick-chaos/
* The Dripping Faucet as a Chaotic System by Rober Shaw
* Ergodic theory of chaos and strange attractors, J.-P. Eckmann, D. Ruelle
* Strange Attractors, Chaotic Behavior, and Information Flow, Robert Shaw

## plots

NIO identified non-uniform regions of the Lorenz state space associated with enhanced trajectory sensitivity. The discovered regions emerge solely from optimization of trajectory divergence and exhibit clear geometric structure in multiple state-space projections.

* An interactive discovery framework for exploring sensitivity structure in complex systems.

## Lyapunov


Mean FTLE

NIO-Max      1.25

Random       0.82

NIO-Min      0.18


```python



import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# LORENZ SYSTEM
# =====================================================

sigma = 10.0
beta  = 8.0 / 3.0
rho   = 28.0

dt = 0.01


def lorenz_step(x):

    dx = sigma * (x[1] - x[0])

    dy = x[0] * (rho - x[2]) - x[1]

    dz = x[0] * x[1] - beta * x[2]

    return x + dt * np.array([dx, dy, dz])


def simulate_lorenz(x0, steps=1000):

    x = np.array(x0, dtype=np.float64)

    traj = np.zeros((steps + 1, 3))

    traj[0] = x

    for i in range(steps):

        x = lorenz_step(x)

        traj[i + 1] = x

    return traj


# =====================================================
# NIO SCORE
# SAME IDEA AS PAPER A LOSS
# =====================================================

def nio_divergence_score(
    x0,
    delta=1e-6,
    steps=1000
):

    x1 = np.array(x0)

    x2 = np.array(x0)

    x2[0] += delta

    traj1 = simulate_lorenz(x1, steps)

    traj2 = simulate_lorenz(x2, steps)

    separation = np.linalg.norm(
        traj2 - traj1,
        axis=1
    )

    score = np.sum(separation**2)

    return score


# =====================================================
# FINITE TIME LYAPUNOV
# =====================================================

def finite_time_lyapunov(
    x0,
    delta=1e-6,
    steps=1000
):

    x1 = np.array(x0)

    x2 = np.array(x0)

    x2[0] += delta

    traj1 = simulate_lorenz(x1, steps)

    traj2 = simulate_lorenz(x2, steps)

    d0 = np.linalg.norm(
        traj2[0] - traj1[0]
    )

    dT = np.linalg.norm(
        traj2[-1] - traj1[-1]
    )

    dT = max(dT, 1e-12)

    lam = np.log(dT / d0) / (steps * dt)

    return lam


# =====================================================
# RANDOM BASELINE
# =====================================================

def generate_random_points(n=200):

    pts = np.zeros((n,3))

    pts[:,0] = np.random.uniform(-20,20,n)

    pts[:,1] = np.random.uniform(-30,30,n)

    pts[:,2] = np.random.uniform(0,50,n)

    return pts


random_points = generate_random_points(200)

# =====================================================
# COMPUTE FTLE
# =====================================================

print("Computing FTLE values...")

high_ftle = []
low_ftle = []
rand_ftle = []

for p in high_points:

    high_ftle.append(
        finite_time_lyapunov(p)
    )

for p in low_points:

    low_ftle.append(
        finite_time_lyapunov(p)
    )

for p in random_points:

    rand_ftle.append(
        finite_time_lyapunov(p)
    )

high_ftle = np.array(high_ftle)
low_ftle = np.array(low_ftle)
rand_ftle = np.array(rand_ftle)

# =====================================================
# SUMMARY TABLE
# =====================================================

print("\n==============================")
print("FINITE TIME LYAPUNOV RESULTS")
print("==============================")

print(
    "NIO MAX  Mean FTLE:",
    np.mean(high_ftle)
)

print(
    "NIO MIN  Mean FTLE:",
    np.mean(low_ftle)
)

print(
    "Random   Mean FTLE:",
    np.mean(rand_ftle)
)

print()

print(
    "NIO MAX  Std:",
    np.std(high_ftle)
)

print(
    "NIO MIN  Std:",
    np.std(low_ftle)
)

print(
    "Random   Std:",
    np.std(rand_ftle)
)

# =====================================================
# HISTOGRAM
# =====================================================

plt.figure(figsize=(8,5))

plt.hist(
    high_ftle,
    bins=20,
    alpha=0.6,
    label="NIO Max"
)

plt.hist(
    low_ftle,
    bins=20,
    alpha=0.6,
    label="NIO Min"
)

plt.hist(
    rand_ftle,
    bins=20,
    alpha=0.6,
    label="Random"
)

plt.xlabel("Finite-Time Lyapunov Exponent")
plt.ylabel("Count")

plt.title("FTLE Distribution")

plt.legend()

plt.show()

# =====================================================
# BOXPLOT
# =====================================================

plt.figure(figsize=(6,5))

plt.boxplot(
    [
        high_ftle,
        low_ftle,
        rand_ftle
    ],
    labels=[
        "NIO Max",
        "NIO Min",
        "Random"
    ]
)

plt.ylabel("FTLE")

plt.title("FTLE Comparison")

plt.show()

# =====================================================
# CORRELATION WITH NIO SCORE
# =====================================================

print("\nComputing NIO scores...")

high_scores = []

for p in high_points:

    high_scores.append(
        nio_divergence_score(p)
    )

high_scores = np.array(high_scores)

corr = np.corrcoef(
    high_scores,
    high_ftle
)[0,1]

print()

print(
    "Correlation (NIO Score vs FTLE):",
    corr
)

plt.figure(figsize=(7,5))

plt.scatter(
    high_scores,
    high_ftle,
    alpha=0.7
)

plt.xlabel("NIO Divergence Score")

plt.ylabel("FTLE")

plt.title(
    f"NIO Score vs FTLE\nCorrelation={corr:.3f}"
)

plt.show()

# =====================================================
# MOST CHAOTIC POINT
# =====================================================

idx = np.argmax(high_ftle)

print("\nMost Chaotic NIO Point")

print(
    "Initial Condition:",
    high_points[idx]
)

print(
    "FTLE:",
    high_ftle[idx]
)

print(
    "NIO Score:",
    high_scores[idx]
)



```


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


more points


```python
# ==========================================================
# PAPER A
# Casimir Point Discovery via NIO
#
# Goal:
# Discover important plate spacings.
#
# NIO directly optimizes spacing.
#
# No training.
# No density model.
#
# ==========================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)

# ==========================================================
# Casimir Force
#
# F = K / a^4
#
# Simplified normalized model.
# ==========================================================

K = 1.0

def casimir_force(a):
    return K/(a**4)

# ==========================================================
# Search Range
# ==========================================================

A_MIN = 0.2
A_MAX = 5.0

# ==========================================================
# NIO Variable
#
# z unconstrained
#
# a = mapped spacing
# ==========================================================

z = nn.Parameter(
    torch.tensor([0.0], device=device)
)

optimizer = torch.optim.Adam(
    [z],
    lr=0.05
)

history = []

# ==========================================================
# Objective
#
# Discover spacing producing
# maximum Casimir force.
# ==========================================================

for step in range(500):

    optimizer.zero_grad()

    a = (
        A_MIN
        +
        (A_MAX-A_MIN)
        *
        torch.sigmoid(z)
    )

    force = casimir_force(a)

    loss = -force

    loss.backward()

    optimizer.step()

    history.append(
        force.item()
    )

# ==========================================================
# Result
# ==========================================================

best_a = (
    A_MIN
    +
    (A_MAX-A_MIN)
    *
    torch.sigmoid(z)
).item()

print()
print("Best spacing:",best_a)
print("Force:",history[-1])

# ==========================================================
# Convergence Plot
# ==========================================================

plt.figure(figsize=(6,4))

plt.plot(history)

plt.title(
    "NIO Convergence"
)

plt.xlabel("Iteration")
plt.ylabel("Casimir Force")

plt.show()

# ==========================================================
# Compare Against Full Curve
# ==========================================================

a_grid = np.linspace(
    A_MIN,
    A_MAX,
    1000
)

f_grid = K/(a_grid**4)

plt.figure(figsize=(7,4))

plt.plot(
    a_grid,
    f_grid,
    label="Casimir Force"
)

plt.scatter(
    [best_a],
    [history[-1]],
    s=80,
    color="red",
    label="NIO Solution"
)

plt.xlabel(
    "Plate Spacing"
)

plt.ylabel(
    "Force"
)

plt.legend()

plt.title(
    "Paper A: Important Point Discovery"
)

plt.show()

# ==========================================================
# Random Search Baseline
# ==========================================================

best_random = -1

for _ in range(10000):

    a = np.random.uniform(
        A_MIN,
        A_MAX
    )

    score = K/(a**4)

    best_random = max(
        best_random,
        score
    )

print()
print("Random Best:",best_random)
print("NIO Best   :",history[-1])

print(
    "Improvement:",
    history[-1]/best_random
)

# ==========================================================
# Sensitivity Curve
#
# Useful paper figure
# ==========================================================

gradient = np.abs(
    np.gradient(
        f_grid,
        a_grid
    )
)

plt.figure(figsize=(7,4))

plt.plot(
    a_grid,
    gradient
)

plt.xlabel(
    "Plate Spacing"
)

plt.ylabel(
    "|dF/da|"
)

plt.title(
    "Regions of Rapid Change"
)

plt.show()
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

points



```python
# ==========================================================
# PAPER A
# STRING POINT DISCOVERY VIA NIO
#
# Goal:
# Find pluck locations that produce
# rich harmonic content.
#
# ==========================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)

# ==========================================================
# PARAMETERS
# ==========================================================

N_MODES = 20

# ==========================================================
# NIO VARIABLE
# ==========================================================

z = nn.Parameter(
    torch.tensor([0.0], device=device)
)

optimizer = torch.optim.Adam(
    [z],
    lr=0.05
)

history = []

# ==========================================================
# OBJECTIVE
#
# Harmonic richness
# ==========================================================

for step in range(500):

    optimizer.zero_grad()

    x = torch.sigmoid(z)

    amplitudes = []

    for n in range(1,N_MODES+1):

        amp = torch.sin(
            np.pi*n*x
        )

        amplitudes.append(
            torch.abs(amp)
        )

    amplitudes = torch.stack(
        amplitudes
    )

    richness = amplitudes.sum()

    loss = -richness

    loss.backward()

    optimizer.step()

    history.append(
        richness.item()
    )

# ==========================================================
# RESULT
# ==========================================================

best_x = torch.sigmoid(z).item()

print()
print("Best pluck position:",best_x)
print("Richness:",history[-1])

# ==========================================================
# CONVERGENCE
# ==========================================================

plt.figure(figsize=(6,4))

plt.plot(history)

plt.title(
    "NIO Convergence"
)

plt.xlabel("Iteration")
plt.ylabel("Richness")

plt.show()

# ==========================================================
# HARMONIC SPECTRUM
# ==========================================================

amps = []

for n in range(1,N_MODES+1):

    amps.append(
        np.abs(
            np.sin(
                np.pi*n*best_x
            )
        )
    )

plt.figure(figsize=(8,4))

plt.bar(
    np.arange(1,N_MODES+1),
    amps
)

plt.xlabel(
    "Mode Number"
)

plt.ylabel(
    "Amplitude"
)

plt.title(
    "Spectrum at NIO Location"
)

plt.show()

# ==========================================================
# FULL SEARCH CURVE
# ==========================================================

xs = np.linspace(
    0.01,
    0.99,
    500
)

scores = []

for x in xs:

    s = 0

    for n in range(1,N_MODES+1):

        s += abs(
            np.sin(
                np.pi*n*x
            )
        )

    scores.append(s)

scores = np.array(scores)

plt.figure(figsize=(8,4))

plt.plot(
    xs,
    scores
)

plt.scatter(
    [best_x],
    [history[-1]],
    color="red",
    s=80
)

plt.xlabel(
    "Pluck Position"
)

plt.ylabel(
    "Richness"
)

plt.title(
    "Important Point Discovery"
)

plt.show()

# ==========================================================
# RANDOM SEARCH BASELINE
# ==========================================================

best_random = -1

for _ in range(10000):

    x = np.random.uniform(
        0.01,
        0.99
    )

    score = 0

    for n in range(1,N_MODES+1):

        score += abs(
            np.sin(
                np.pi*n*x
            )
        )

    best_random = max(
        best_random,
        score
    )

print()
print("Random Best:",best_random)
print("NIO Best   :",history[-1])
print("Improvement:",
      history[-1]/best_random)
```




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


```python
# ==========================================================
# PAPER A
# EM CAVITY IMPORTANT GEOMETRY DISCOVERY
#
# Goal:
# Find cavity dimensions producing
# rich resonant structure.
#
# ==========================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)

# ==========================================================
# PARAMETERS
# ==========================================================

MAX_INDEX = 4

# ==========================================================
# MODE GENERATOR
# ==========================================================

def cavity_modes(ax,ay,az):

    modes = []

    for m in range(1,MAX_INDEX):
        for n in range(1,MAX_INDEX):
            for p in range(1,MAX_INDEX):

                f = np.sqrt(
                    (m/ax)**2 +
                    (n/ay)**2 +
                    (p/az)**2
                )

                modes.append(f)

    return np.array(modes)

# ==========================================================
# NIO VARIABLES
# ==========================================================

z = nn.Parameter(
    torch.zeros(3,device=device)
)

optimizer = torch.optim.Adam(
    [z],
    lr=0.05
)

history = []

# ==========================================================
# OBJECTIVE
#
# maximize number of modes
# below cutoff frequency
# ==========================================================

F_CUTOFF = 4.0

for step in range(500):

    optimizer.zero_grad()

    dims = (
        0.5
        +
        4.5*torch.sigmoid(z)
    )

    ax,ay,az = dims

    score = 0

    for m in range(1,MAX_INDEX):
        for n in range(1,MAX_INDEX):
            for p in range(1,MAX_INDEX):

                f = torch.sqrt(
                    (m/ax)**2 +
                    (n/ay)**2 +
                    (p/az)**2
                )

                score += torch.sigmoid(
                    10*(F_CUTOFF-f)
                )

    loss = -score

    loss.backward()

    optimizer.step()

    history.append(
        score.item()
    )

# ==========================================================
# RESULT
# ==========================================================

dims = (
    0.5
    +
    4.5*torch.sigmoid(z)
).detach().cpu().numpy()

print()
print("Best Geometry")
print("a =",dims[0])
print("b =",dims[1])
print("c =",dims[2])

# ==========================================================
# CONVERGENCE
# ==========================================================

plt.figure(figsize=(6,4))

plt.plot(history)

plt.title(
    "NIO Convergence"
)

plt.xlabel("Iteration")
plt.ylabel("Mode Density")

plt.show()

# ==========================================================
# SPECTRUM
# ==========================================================

modes = cavity_modes(
    dims[0],
    dims[1],
    dims[2]
)

modes = np.sort(modes)

plt.figure(figsize=(8,4))

plt.vlines(
    modes,
    0,
    1
)

plt.axvline(
    F_CUTOFF,
    color='red'
)

plt.title(
    "Discovered EM Spectrum"
)

plt.xlabel(
    "Frequency"
)

plt.show()

# ==========================================================
# RANDOM BASELINE
# ==========================================================

best_random = -1

for _ in range(10000):

    ax = np.random.uniform(0.5,5)
    ay = np.random.uniform(0.5,5)
    az = np.random.uniform(0.5,5)

    score = 0

    for m in range(1,MAX_INDEX):
        for n in range(1,MAX_INDEX):
            for p in range(1,MAX_INDEX):

                f = np.sqrt(
                    (m/ax)**2 +
                    (n/ay)**2 +
                    (p/az)**2
                )

                score += (
                    f < F_CUTOFF
                )

    best_random = max(
        best_random,
        score
    )

print()
print("Random Best:",best_random)
print("NIO Best   :",history[-1])
```



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




## Projections and density map plots


```python

# ============================================================
# Lorenz NIO Paper
#
# Step 1 : Load NIO Results
# Step 2 : Summary Statistics
# Step 3 : 3D State Space Plot
# Step 4 : Projection Plots
# Step 5 : Sensitivity Density Maps
#
# Assumes:
#   nio_optimized_200_max.csv
#   nio_optimized_200_minimize.csv
#
# Uses:
#   900-step horizon only
#
# Ricardo Calix
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import gaussian_kde

# ============================================================
# CONFIGURATION
# ============================================================

MAX_FILE = "nio_optimized_200_max.csv"
MIN_FILE = "nio_optimized_200_minimize.csv"

HORIZON = 900

# ============================================================
# LOAD DATA
# ============================================================

max_df = pd.read_csv(MAX_FILE)
min_df = pd.read_csv(MIN_FILE)

time_col = [c for c in max_df.columns
            if "time" in c.lower()][0]

max_data = max_df[max_df[time_col] == HORIZON].copy()
min_data = min_df[min_df[time_col] == HORIZON].copy()

print("\nLoaded Data")
print("----------------------------")
print("Max-sensitive points :", len(max_data))
print("Min-sensitive points :", len(min_data))

# ============================================================
# STEP 2
# SUMMARY STATISTICS
# ============================================================

print("\n")
print("="*70)
print("SUMMARY STATISTICS")
print("="*70)

def summarize(df, label):

    stats = pd.DataFrame({
        "Mean": df[["x0","y0","z0"]].mean(),
        "Std": df[["x0","y0","z0"]].std(),
        "Min": df[["x0","y0","z0"]].min(),
        "Max": df[["x0","y0","z0"]].max()
    })

    print("\n")
    print(label)
    print("-"*50)
    print(stats)

summarize(max_data, "MAX-SENSITIVE")
summarize(min_data, "MIN-SENSITIVE")

# ============================================================
# STEP 3
# FIGURE 1
# 3D STATE SPACE
# ============================================================

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(
    111,
    projection='3d'
)

ax.scatter(
    min_data["x0"],
    min_data["y0"],
    min_data["z0"],
    s=10,
    alpha=0.15,
    label="Min Sensitive"
)

ax.scatter(
    max_data["x0"],
    max_data["y0"],
    max_data["z0"],
    s=40,
    alpha=0.75,
    label="Max Sensitive"
)

ax.set_xlabel("x0")
ax.set_ylabel("y0")
ax.set_zlabel("z0")

ax.set_title(
    f"Lorenz State Space\nMax vs Min Sensitive Initial Conditions ({HORIZON} Steps)"
)

ax.legend()

plt.tight_layout()
plt.savefig(
    "Figure1_3D_StateSpace.png",
    dpi=300
)

plt.show()

# ============================================================
# STEP 4
# FIGURE 2
# PROJECTION PLOTS
# ============================================================

fig, axs = plt.subplots(
    1,
    3,
    figsize=(16,5)
)

# ------------------------------------------------------------
# XY
# ------------------------------------------------------------

axs[0].scatter(
    min_data["x0"],
    min_data["y0"],
    s=10,
    alpha=0.15,
    label="Min"
)

axs[0].scatter(
    max_data["x0"],
    max_data["y0"],
    s=40,
    alpha=0.7,
    label="Max"
)

axs[0].set_title("XY Projection")
axs[0].set_xlabel("x0")
axs[0].set_ylabel("y0")

# ------------------------------------------------------------
# XZ
# ------------------------------------------------------------

axs[1].scatter(
    min_data["x0"],
    min_data["z0"],
    s=10,
    alpha=0.15
)

axs[1].scatter(
    max_data["x0"],
    max_data["z0"],
    s=40,
    alpha=0.7
)

axs[1].set_title("XZ Projection")
axs[1].set_xlabel("x0")
axs[1].set_ylabel("z0")

# ------------------------------------------------------------
# YZ
# ------------------------------------------------------------

axs[2].scatter(
    min_data["y0"],
    min_data["z0"],
    s=10,
    alpha=0.15
)

axs[2].scatter(
    max_data["y0"],
    max_data["z0"],
    s=40,
    alpha=0.7
)

axs[2].set_title("YZ Projection")
axs[2].set_xlabel("y0")
axs[2].set_ylabel("z0")

axs[0].legend()

plt.suptitle(
    f"Lorenz NIO Initial Conditions ({HORIZON} Steps)"
)

plt.tight_layout()

plt.savefig(
    "Figure2_ProjectionPlots.png",
    dpi=300
)

plt.show()

# ============================================================
# STEP 5
# FIGURE 3
# SENSITIVITY DENSITY MAPS
#
# rho_max - rho_min
# ============================================================

fig, axs = plt.subplots(
    1,
    3,
    figsize=(16,5)
)

pairs = [
    ("x0","y0","XY"),
    ("x0","z0","XZ"),
    ("y0","z0","YZ")
]

for ax, (a,b,title) in zip(axs,pairs):

    all_a = np.concatenate([
        max_data[a],
        min_data[a]
    ])

    all_b = np.concatenate([
        max_data[b],
        min_data[b]
    ])

    grid_a = np.linspace(
        all_a.min(),
        all_a.max(),
        120
    )

    grid_b = np.linspace(
        all_b.min(),
        all_b.max(),
        120
    )

    A,B = np.meshgrid(
        grid_a,
        grid_b
    )

    kde_max = gaussian_kde(
        np.vstack([
            max_data[a],
            max_data[b]
        ])
    )

    kde_min = gaussian_kde(
        np.vstack([
            min_data[a],
            min_data[b]
        ])
    )

    rho_max = kde_max(
        np.vstack([
            A.ravel(),
            B.ravel()
        ])
    ).reshape(A.shape)

    rho_min = kde_min(
        np.vstack([
            A.ravel(),
            B.ravel()
        ])
    ).reshape(A.shape)

    density_difference = (
        rho_max - rho_min
    )

    vmax = np.abs(
        density_difference
    ).max()

    image = ax.imshow(
        density_difference,
        origin="lower",
        extent=[
            grid_a.min(),
            grid_a.max(),
            grid_b.min(),
            grid_b.max()
        ],
        aspect="auto",
        cmap="bwr",
        vmin=-vmax,
        vmax=vmax
    )

    ax.set_title(
        f"{title}: ρmax - ρmin"
    )

    ax.set_xlabel(a)
    ax.set_ylabel(b)

    plt.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

plt.suptitle(
    f"Sensitivity Density Maps\nLorenz NIO ({HORIZON} Steps)"
)

plt.tight_layout()

plt.savefig(
    "Figure3_DensityMaps.png",
    dpi=300
)

plt.show()

# ============================================================
# SAVE SUMMARY TABLES
# ============================================================

summary_max = pd.DataFrame({
    "Mean": max_data[["x0","y0","z0"]].mean(),
    "Std": max_data[["x0","y0","z0"]].std(),
    "Min": max_data[["x0","y0","z0"]].min(),
    "Max": max_data[["x0","y0","z0"]].max()
})

summary_min = pd.DataFrame({
    "Mean": min_data[["x0","y0","z0"]].mean(),
    "Std": min_data[["x0","y0","z0"]].std(),
    "Min": min_data[["x0","y0","z0"]].min(),
    "Max": min_data[["x0","y0","z0"]].max()
})

summary_max.to_csv(
    "Table_MaxSensitive.csv"
)

summary_min.to_csv(
    "Table_MinSensitive.csv"
)

print("\n")
print("="*70)
print("DONE")
print("="*70)

print("Generated:")
print("  Figure1_3D_StateSpace.png")
print("  Figure2_ProjectionPlots.png")
print("  Figure3_DensityMaps.png")
print("  Table_MaxSensitive.csv")
print("  Table_MinSensitive.csv")



```



## Math notes



## Projection Plots

Each NIO solution corresponds to a point in the three-dimensional Lorenz state space:

$$
p_i = (x_i, y_i, z_i)
$$

To visualize the distribution of sensitive and insensitive states, we project the three-dimensional state space onto two-dimensional planes.

### XY Projection

$$
\Pi_{xy}(x,y,z) = (x,y)
$$

### XZ Projection

$$
\Pi_{xz}(x,y,z) = (x,z)
$$

### YZ Projection

$$
\Pi_{yz}(x,y,z) = (y,z)
$$

In practice, the projection plots are obtained by simply discarding one coordinate:

* XY projection removes $z$
* XZ projection removes $y$
* YZ projection removes $x$

Although mathematically simple, these projections revealed structures that were difficult to observe in the full three-dimensional state space.

---

## Sensitivity Density Maps

The projection plots provide a qualitative view of the discovered states. To quantify where NIO prefers to place sensitive initial conditions, we estimate probability densities for both the max-sensitive and min-sensitive sets using Kernel Density Estimation (KDE).

### Kernel Density Estimation

Suppose the max-sensitive set contains:

$$
(x_1,y_1), (x_2,y_2), \ldots, (x_N,y_N)
$$

The density estimate is

$$
\rho_{\max}(x,y)
================

\frac{1}{N}
\sum_{i=1}^{N}
K_h
\Big(
(x,y)-(x_i,y_i)
\Big)
$$

where $K_h$ is a Gaussian kernel.

The Gaussian kernel is

$$
K_h(\mathbf r)
==============

\frac{1}{2\pi h^2}
\exp
\left(
-\frac{|\mathbf r|^2}
{2h^2}
\right)
$$

Each point contributes a small Gaussian bump, and the density estimate is obtained by summing all bumps.

Similarly, for the min-sensitive points:

$$
\rho_{\min}(x,y)
================

\frac{1}{M}
\sum_{j=1}^{M}
K_h
\Big(
(x,y)-(u_j,v_j)
\Big)
$$

---

## Density Difference Map

The key visualization used in this work is the density difference map:

$$

D(x,y) =  \rho_{\max}(x,y) -  \rho_{\min}(x,y)

$$

This quantity directly measures where NIO preferentially identifies sensitive states.

### Interpretation

#### Red Regions

$$
D(x,y) > 0
$$

These locations contain a higher density of max-sensitive states than min-sensitive states.

NIO prefers these regions.

#### Blue Regions

$$
D(x,y) < 0
$$

These locations contain a higher density of min-sensitive states.

NIO avoids these regions when searching for highly sensitive initial conditions.

#### White Regions

$$
D(x,y) \approx 0
$$

No significant difference exists between the two populations.

---

## Scientific Interpretation

The density difference maps transform a collection of optimized initial conditions into an interpretable sensitivity map of the Lorenz state space.

Rather than simply reporting a list of coordinates, the workflow becomes:

$$
\text{NIO}
\rightarrow
\text{Sensitive States}
\rightarrow
\text{Projection Plots}
\rightarrow
\text{Density Maps}
\rightarrow
\text{Interpretation}
$$

This process functions as a form of "state-space oscilloscope," allowing researchers to visualize where sensitivity is concentrated and to iteratively explore the structure of complex nonlinear systems.






