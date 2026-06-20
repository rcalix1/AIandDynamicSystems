## Medical



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
