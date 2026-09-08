## Paper Ideas

* link

## NIO + cybersecurity/malware dynamics

```
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# NIO EXPERIMENT 1:
# Malware Spread with Interpretable Cybersecurity Parameters
#
# State:
#   S = susceptible fraction of hosts
#   I = infected / compromised fraction
#   R = protected / recovered / isolated fraction
#   C = cumulative infection exposure
#
# NIO optimizes:
#   I0    = initial compromised fraction
#   beta  = malware transmission rate
#   q     = detection / isolation rate
#   u     = proactive protection / patching rate
#
# Goal:
#   Find conditions that MAXIMIZE or MINIMIZE malware outbreak.
#
# This is intentionally structured like the Lorenz NIO code.
# ============================================================


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
dt = 0.02
Time_steps = 800
iters = 1000
N = 200

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# ============================================================
# Malware dynamical system
# ============================================================

class MalwareStep(nn.Module):

    def __init__(self, dt=0.02):
        super().__init__()
        self.dt = dt

    def forward(self, state, beta, q, u):

        S = state[:, 0]
        I = state[:, 1]
        R = state[:, 2]
        C = state[:, 3]

        # ----------------------------------------------------
        # Malware dynamics
        #
        # Infection:
        # susceptible machines become compromised
        #
        # Isolation:
        # compromised machines are detected / isolated
        #
        # Protection:
        # susceptible machines are patched / hardened
        # ----------------------------------------------------

        infection = beta * S * I
        isolation = q * I
        protection = u * S

        dS = -infection - protection
        dI = infection - isolation
        dR = isolation + protection

        # Integral of infection burden
        dC = I

        # Euler integration
        S_new = S + self.dt * dS
        I_new = I + self.dt * dI
        R_new = R + self.dt * dR
        C_new = C + self.dt * dC

        return torch.stack(
            [S_new, I_new, R_new, C_new],
            dim=1
        )


model = MalwareStep(dt=dt).to(device)


# ============================================================
# Parameter bounds
# ============================================================
#
# These are intentionally broad experimental bounds.
#
# I0:
#   0.1% - 10% of network initially compromised
#
# beta:
#   malware transmission strength
#
# q:
#   detection / isolation strength
#
# u:
#   proactive patching / protection strength
#
# NIO can NEVER leave these ranges because we use sigmoid.
# ============================================================

low = torch.tensor(
    [0.001, 0.10, 0.00, 0.00],
    device=device
)

high = torch.tensor(
    [0.10, 2.00, 1.00, 1.00],
    device=device
)


# ============================================================
# Convert unconstrained NIO variables into real parameters
# ============================================================

def decode_parameters(z):

    p = low + (high - low) * torch.sigmoid(z)

    I0   = p[:, 0]
    beta = p[:, 1]
    q    = p[:, 2]
    u    = p[:, 3]

    return I0, beta, q, u


# ============================================================
# Simulate complete malware trajectory
# ============================================================

def simulate(I0, beta, q, u, Time_steps):

    # Initially:
    # infected fraction = I0
    # everybody else is susceptible
    # nobody has yet been protected/recovered

    S0 = 1.0 - I0
    R0 = torch.zeros_like(I0)
    C0 = torch.zeros_like(I0)

    state = torch.stack(
        [S0, I0, R0, C0],
        dim=1
    )

    trajectory = [state]

    for t in range(Time_steps):

        state = model(
            state,
            beta,
            q,
            u
        )

        trajectory.append(state)

    return torch.stack(trajectory)


# ============================================================
# Outbreak objective
# ============================================================
#
# We DO NOT optimize only final infection.
#
# Why?
#
# A devastating outbreak could infect most machines,
# then disappear before the final time.
#
# Instead we measure:
#
#   1. cumulative infection exposure
#   2. peak infection
#
# This gives NIO an objective representing the whole outbreak.
# ============================================================

def outbreak_score(trajectory):

    I = trajectory[:, :, 1]
    C = trajectory[-1, :, 3]

    peak_I = torch.max(I, dim=0).values

    # Cumulative burden is primary.
    # Peak infection provides additional pressure toward
    # severe outbreaks.

    score = C + 2.0 * peak_I

    return score.mean()


# ============================================================
# NIO FUNCTION
# ============================================================
#
# maximize=True:
#     search for dangerous / worst-case conditions
#
# maximize=False:
#     search for containment conditions
# ============================================================

def run_nio(maximize=True):

    # Random latent starting positions
    z_init = torch.randn(
        N,
        4,
        device=device,
        requires_grad=True
    )

    optimizer = optim.Adam(
        [z_init],
        lr=0.01
    )

    for i in range(iters):

        I0, beta, q, u = decode_parameters(z_init)

        trajectory = simulate(
            I0,
            beta,
            q,
            u,
            Time_steps
        )

        score = outbreak_score(trajectory)

        if maximize:
            loss = -score
        else:
            loss = score

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:

            with torch.no_grad():

                print(
                    "iter", i,
                    "score", score.item(),
                    "| I0", I0.mean().item(),
                    "| beta", beta.mean().item(),
                    "| q", q.mean().item(),
                    "| u", u.mean().item()
                )

    return z_init


# ============================================================
# RUN NIO: WORST-CASE MALWARE CONDITIONS
# ============================================================

print("\n")
print("==========================================")
print("NIO: MAXIMIZE MALWARE OUTBREAK")
print("==========================================")

z_max = run_nio(maximize=True)


# ============================================================
# RUN NIO: CONTAINMENT CONDITIONS
# ============================================================

print("\n")
print("==========================================")
print("NIO: MINIMIZE MALWARE OUTBREAK")
print("==========================================")

z_min = run_nio(maximize=False)


# ============================================================
# Extract final optimized parameters
# ============================================================

with torch.no_grad():

    I0_max, beta_max, q_max, u_max = decode_parameters(z_max)
    I0_min, beta_min, q_min, u_min = decode_parameters(z_min)

    traj_max = simulate(
        I0_max,
        beta_max,
        q_max,
        u_max,
        Time_steps
    )

    traj_min = simulate(
        I0_min,
        beta_min,
        q_min,
        u_min,
        Time_steps
    )


# ============================================================
# Print average optimized parameters
# ============================================================

print("\n")
print("==========================================")
print("AVERAGE NIO RESULTS")
print("==========================================")

print("\nWORST CASE:")
print("Initial infected I0 :", I0_max.mean().item())
print("Transmission beta   :", beta_max.mean().item())
print("Isolation q         :", q_max.mean().item())
print("Protection u        :", u_max.mean().item())

print("\nCONTAINMENT:")
print("Initial infected I0 :", I0_min.mean().item())
print("Transmission beta   :", beta_min.mean().item())
print("Isolation q         :", q_min.mean().item())
print("Protection u        :", u_min.mean().item())


# ============================================================
# Calculate outbreak statistics
# ============================================================

with torch.no_grad():

    infected_max = traj_max[:, :, 1]
    infected_min = traj_min[:, :, 1]

    mean_curve_max = infected_max.mean(dim=1)
    mean_curve_min = infected_min.mean(dim=1)

    peak_max = infected_max.max(dim=0).values.mean()
    peak_min = infected_min.max(dim=0).values.mean()

    cumulative_max = traj_max[-1, :, 3].mean()
    cumulative_min = traj_min[-1, :, 3].mean()

    final_max = infected_max[-1].mean()
    final_min = infected_min[-1].mean()


print("\n")
print("==========================================")
print("OUTBREAK STATISTICS")
print("==========================================")

print("\nWORST CASE")
print("Peak infected       :", peak_max.item())
print("Final infected      :", final_max.item())
print("Cumulative exposure :", cumulative_max.item())

print("\nCONTAINMENT")
print("Peak infected       :", peak_min.item())
print("Final infected      :", final_min.item())
print("Cumulative exposure :", cumulative_min.item())


# ============================================================
# Plot infection trajectories
# ============================================================

time = (
    torch.arange(Time_steps + 1)
    * dt
)

time = time.cpu().numpy()

plt.figure(figsize=(9, 5))

plt.plot(
    time,
    mean_curve_max.cpu().numpy(),
    label="NIO Maximum Outbreak",
    linewidth=2
)

plt.plot(
    time,
    mean_curve_min.cpu().numpy(),
    label="NIO Containment",
    linewidth=2
)

plt.xlabel("Time")
plt.ylabel("Fraction of Network Infected")
plt.title("NIO Malware Dynamics")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# Save all optimized parameter sets
# ============================================================

rows = []

with torch.no_grad():

    for j in range(N):

        rows.append({

            "experiment": "MAX",

            "point_id": j,

            "I0":
                I0_max[j].item(),

            "beta":
                beta_max[j].item(),

            "isolation_q":
                q_max[j].item(),

            "protection_u":
                u_max[j].item(),

            "peak_infected":
                traj_max[:, j, 1].max().item(),

            "final_infected":
                traj_max[-1, j, 1].item(),

            "cumulative_exposure":
                traj_max[-1, j, 3].item()
        })

    for j in range(N):

        rows.append({

            "experiment": "MIN",

            "point_id": j,

            "I0":
                I0_min[j].item(),

            "beta":
                beta_min[j].item(),

            "isolation_q":
                q_min[j].item(),

            "protection_u":
                u_min[j].item(),

            "peak_infected":
                traj_min[:, j, 1].max().item(),

            "final_infected":
                traj_min[-1, j, 1].item(),

            "cumulative_exposure":
                traj_min[-1, j, 3].item()
        })


df = pd.DataFrame(rows)

df.to_csv(
    "nio_malware_results.csv",
    index=False
)

print("\nSaved: nio_malware_results.csv")


# ============================================================
# EXTRA PLOT:
# Distribution of optimized physical/security parameters
# ============================================================

labels = [
    "Initial Infection",
    "Transmission",
    "Isolation",
    "Protection"
]

max_values = [
    I0_max.mean().item(),
    beta_max.mean().item(),
    q_max.mean().item(),
    u_max.mean().item()
]

min_values = [
    I0_min.mean().item(),
    beta_min.mean().item(),
    q_min.mean().item(),
    u_min.mean().item()
]

x = range(len(labels))

plt.figure(figsize=(9, 5))

plt.plot(
    x,
    max_values,
    marker="o",
    label="Maximum Outbreak"
)

plt.plot(
    x,
    min_values,
    marker="o",
    label="Containment"
)

plt.xticks(x, labels)

plt.ylabel("Optimized Parameter Value")
plt.title("Physical / Security Parameters Found by NIO")

plt.legend()
plt.grid(True)

plt.show()


```


## NIO + mathematical image/video dynamics


```



```


## NIO + geometry / vibrating string


```



```

## Information theory + dynamical systems




```




```








