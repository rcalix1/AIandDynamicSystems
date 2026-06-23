## Medical

To get the low-tumor group, change only one line:

* loss = torch.mean(final_tumor)



```python




#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------
# settings
# -----------------------------------

dt = 0.01
iters = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------
# Tumor-Immune-Cytokine Model
# -----------------------------------

class TumorImmuneStep(nn.Module):

    def __init__(self, dt=0.01):

        super().__init__()

        # tumor growth
        self.a = 0.50
        self.b = 0.01

        # immune suppression
        self.c = 0.02

        # immune dynamics
        self.d = 0.01
        self.e = 0.10
        self.f = 0.05

        # cytokine dynamics
        self.g = 0.05
        self.h = 0.10

        self.dt = dt

    def forward(self, x):

        T = x[:,0]
        I = x[:,1]
        C = x[:,2]

        dT = self.a*T*(1.0 - self.b*T) - self.c*T*I

        dI = self.d*T*I - self.e*I + self.f*C

        dC = self.g*T - self.h*C

        T = T + self.dt*dT
        I = I + self.dt*dI
        C = C + self.dt*dC

        T = torch.clamp(T, min=0.0)
        I = torch.clamp(I, min=0.0)
        C = torch.clamp(C, min=0.0)

        return torch.stack([T,I,C], dim=1)

model = TumorImmuneStep(dt=dt).to(device)

# -----------------------------------
# search space
# -----------------------------------

low = torch.tensor(
    [1.0, 1.0, 1.0],
    device=device
)

high = torch.tensor(
    [100.0, 100.0, 100.0],
    device=device
)

# -----------------------------------
# settings
# -----------------------------------

N = 1000

T_list = [300,500,900,1300]

avg_tumor_list = []

optimized_rows = []

# -----------------------------------
# MAIN LOOP
# -----------------------------------

for Time_steps in T_list:

    print("\nRunning NIO for T =", Time_steps)

    z_init = torch.randn(
        N,
        3,
        device=device,
        requires_grad=True
    )

    optimizer = optim.Adam(
        [z_init],
        lr=0.05
    )

    # -----------------------------------
    # NIO
    # -----------------------------------

    for i in range(iters):

        x0 = low + (high-low)*torch.sigmoid(z_init)

        x = x0

        for t in range(Time_steps):

            x = model(x)

        final_tumor = x[:,0]

        # -----------------------------------
        # MAXIMUM TUMOR
        # -----------------------------------

        loss = -torch.mean(final_tumor)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if i % 100 == 0:

            print(
                "iter",
                i,
                "loss",
                loss.item()
            )

    # -----------------------------------
    # SAVE RESULTS
    # -----------------------------------

    with torch.no_grad():

        x0 = low + (high-low)*torch.sigmoid(z_init)

        x0_cpu = x0.cpu().numpy()

        for point_id in range(N):

            optimized_rows.append({

                "Time_steps": Time_steps,

                "point_id": point_id,

                "T0": x0_cpu[point_id,0],

                "I0": x0_cpu[point_id,1],

                "C0": x0_cpu[point_id,2]

            })

        x = x0

        for t in range(Time_steps):

            x = model(x)

        final_tumor = x[:,0]

        avg_tumor = final_tumor.mean().item()

        avg_tumor_list.append(
            avg_tumor
        )

        print(
            "T =",
            Time_steps,
            "avg final tumor =",
            avg_tumor
        )

# -----------------------------------
# SAVE CSV
# -----------------------------------

df = pd.DataFrame(
    optimized_rows
)

df.to_csv(
    "nio_tumor_max.csv",
    index=False
)

print(
    "saved nio_tumor_max.csv"
)

# -----------------------------------
# PLOT
# -----------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    T_list,
    avg_tumor_list,
    marker="o"
)

plt.xlabel("Time Steps")

plt.ylabel("Average Final Tumor")

plt.title(
    "NIO Tumor Optimization"
)

plt.grid(True)

plt.show()




```










## math


# Tumor–Immune–Cytokine Dynamical System for Neural Input Optimization

## Overview

This biomedical case study extends Neural Input Optimization (NIO) from the Lorenz system to a simplified tumor–immune–cytokine dynamical system. The goal is to investigate whether NIO can discover biological initial conditions associated with favorable and unfavorable disease progression.

The system contains three state variables:

* **T**: Tumor cell population
* **I**: Immune effector cell population
* **C**: Cytokine concentration

Unlike the Lorenz system, where the variables represent fluid dynamics quantities, the variables in this model have direct biological interpretations.

---

# Related Work

The present model is inspired by mathematical oncology and tumor–immune interaction models.

### Classical Tumor–Immune Model

Kuznetsov, V. A.; Makalkin, I. A.; Taylor, M. A.; Perelson, A. S.; Nonlinear Dynamics of Immunogenic Tumors: Parameter Estimation and Global Bifurcation Analysis. *Bulletin of Mathematical Biology* **1994**, *56*(2), 295–321.

This is one of the most influential tumor–immune dynamical systems papers. The model describes the interaction between tumor cells and immune effector cells using coupled nonlinear differential equations.

---

### Mathematical Oncology

Adam, J. A.; Bellomo, N.; *A Survey of Models for Tumor–Immune System Dynamics*. Birkhäuser, Boston, 1997.

This work provides a broad overview of mathematical models describing tumor growth, immune response, and cancer progression.

---

### Tumor–Immune System Modeling

Kirschner, D.; Panetta, J. C.; Modeling Immunotherapy of the Tumor–Immune Interaction. *Journal of Mathematical Biology* **1998**, *37*(3), 235–252.

This paper introduced tumor–immune interaction models that incorporate immune stimulation and treatment effects and became highly influential in cancer systems biology.

---

### Mathematical Oncology Reference

de Pillis, L. G.; Radunskaya, A. E.; Wiseman, C. L.; A Validated Mathematical Model of Cell-Mediated Immune Response to Tumor Growth. *Cancer Research* **2005**, *65*(17), 7950–7958.

This work presents a biologically motivated mathematical framework for studying immune-mediated suppression of tumor growth.

---

# Dynamical System

The tumor–immune–cytokine model is defined by the following ordinary differential equations.

## Tumor Dynamics

```math
\frac{dT}{dt}
=
aT(1-bT)-cTI
```

where:

* (a) controls tumor growth
* (b) controls carrying capacity
* (c) controls immune-mediated tumor suppression

The first term models logistic tumor growth while the second term represents immune destruction of tumor cells.

---

## Immune Dynamics

```math
\frac{dI}{dt}
=
dTI-eI+fC
```

where:

* (d) controls immune activation
* (e) controls immune decay
* (f) controls cytokine stimulation

Immune cells increase in response to tumor presence and cytokine signaling and decrease through natural decay.

---

## Cytokine Dynamics

```math
\frac{dC}{dt}
=
gT-hC
```

where:

* (g) controls cytokine production
* (h) controls cytokine decay

This equation represents production of signaling molecules by the tumor and their subsequent degradation.

---

# Numerical Integration

The system is evolved using Euler integration.

For the tumor population:

```math
T_{t+1}
=
T_t
+
\Delta t
\left[
aT_t(1-bT_t)-cT_tI_t
\right]
```

For the immune population:

```math
I_{t+1}
=
I_t
+
\Delta t
\left[
dT_tI_t-eI_t+fC_t
\right]
```

For the cytokine population:

```math
C_{t+1}
=
C_t
+
\Delta t
\left[
gT_t-hC_t
\right]
```

This numerical approach is identical to the Euler integration used in the Lorenz experiments.

---

# Neural Input Optimization

The optimization variables are the initial conditions:

```math
(T_0,I_0,C_0)
```

NIO searches the biological state space for conditions associated with favorable and unfavorable outcomes.

---

## Unfavorable Biological States

To discover states associated with aggressive disease progression, NIO maximizes the final tumor burden:

```math
\max T(t_{final})
```

The optimization objective becomes:

```math
L
=
-\frac{1}{N}
\sum_{i=1}^{N}
T_i(t_{final})
```

---

## Favorable Biological States

To discover states associated with reduced tumor burden, NIO minimizes the final tumor burden:

```math
\min T(t_{final})
```

The optimization objective becomes:

```math
L
=
\frac{1}{N}
\sum_{i=1}^{N}
T_i(t_{final})
```

---

# Scientific Interpretation

The goal of this study is not diagnosis or treatment.

Instead, the goal is to determine whether Neural Input Optimization can automatically discover biologically meaningful regions of state space.

Specifically, NIO searches for initial biological conditions that lead to:

* High tumor burden
* Low tumor burden

The resulting states can be analyzed using:

* 3D state-space visualizations
* 2D projections
* Kernel Density Estimation (KDE)
* Statistical comparisons

Potential observations include:

* High tumor burden associated with weak immune activity
* Low tumor burden associated with strong immune activity
* Distinct cytokine patterns associated with favorable outcomes

---

# Contribution

The primary contribution is not the development of a new cancer model.

The contribution is the application of Neural Input Optimization as a state-space discovery framework for biomedical dynamical systems.

The long-term vision is to apply the same framework to:

* Tumor–immune systems
* Glucose–insulin systems
* Cardiac dynamics
* Biological regulatory networks
* Epidemic systems
* Other nonlinear biomedical dynamical systems

In this perspective, NIO functions as a computational search mechanism for discovering important regions of biological state space that may be difficult to identify through manual exploration alone.


---



# Finite-Time Lyapunov Exponents (FTLE) in the Tumor–Immune–Cytokine Model

The sensitivity analysis used in this study is based on the **Finite-Time Lyapunov Exponent (FTLE)**, a standard metric from dynamical systems theory that measures how small perturbations to the initial conditions evolve over a finite time horizon.

Given two nearby initial states separated by a small perturbation:

```text
x₁(0)
x₂(0) = x₁(0) + δ
```

the FTLE is computed as:

```math
λ = \frac{\ln(d_T/d_0)}{T}
```

where:

```math
d_0 = \|x_2(0) - x_1(0)\|
```

is the initial separation,

```math
d_T = \|x_2(T) - x_1(T)\|
```

is the separation after evolving the system for time (T), and

```math
λ
```

is the finite-time Lyapunov exponent.

In the implementation used for this work:

```python
d0 = np.linalg.norm(traj2[0] - traj1[0])

dT = np.linalg.norm(traj2[-1] - traj1[-1])

lam = np.log(dT / d0) / (steps * dt)
```

which is the standard FTLE formulation.

## Interpretation

The FTLE is commonly associated with chaos analysis because positive values indicate exponential divergence of nearby trajectories. However, the tumor–immune–cytokine model studied here is not presented as a chaotic system. Instead, FTLE is used as an independent measure of sensitivity to perturbations in the initial conditions.

Interpretation is straightforward:

```text
λ > 0   → perturbations grow (sensitive dynamics)

λ = 0   → perturbations remain unchanged

λ < 0   → perturbations shrink (stable dynamics)
```

The results obtained in this study were predominantly negative, indicating that nearby trajectories tend to converge rather than diverge. Therefore, the FTLE analysis should be interpreted as a measure of finite-time sensitivity and stability rather than evidence of chaotic behavior.

## Role in the BIBM Study

The primary objective of the paper is not to detect chaos, but to evaluate whether Neural Input Optimization (NIO) discovers distinct regions of the tumor–immune–cytokine state space.

FTLE is therefore used as an independent validation metric. If the NIO-generated high-tumor and low-tumor populations exhibit systematically different FTLE distributions, this suggests that NIO is identifying dynamically distinct regions of the system rather than simply sampling random initial conditions.

Consequently, throughout the manuscript the FTLE results can be described as:

> FTLE-based sensitivity analysis

or

> finite-time sensitivity analysis of tumor trajectories

which is mathematically equivalent to the Lyapunov formulation while emphasizing the biomedical interpretation of the results.






