## Cassimir




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








```
