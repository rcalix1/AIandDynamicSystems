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







