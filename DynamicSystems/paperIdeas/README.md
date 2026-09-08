## Paper Ideas

* link

## NIO + cybersecurity/malware dynamics


Expectation is that MAX will drive roughly toward

$$ I_0\uparrow,\qquad \beta\uparrow,\qquad q\downarrow,\qquad u\downarrow $$

and MIN will do approximately the reverse.



The last graph is actually the one I care about most. If NIO finds that the optimal allocation changes with \(\beta\)—for example, slow malware favors proactive protection while aggressive malware favors rapid isolation, or there is some transition region—that is a result we can start thinking scientifically about.
And importantly, every optimized variable has a cybersecurity interpretation



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


and this one


```


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# NIO CYBERSECURITY EXPERIMENT
#
# Malware propagation with a LIMITED security budget.
#
# NIO discovers:
#
#   I0    = initial fraction of compromised hosts
#   beta  = malware transmission rate
#   alpha = fraction of security budget allocated to isolation
#
# Security budget:
#
#   q = alpha * B
#   u = (1-alpha) * B
#
# where:
#
#   q = detection / isolation rate
#   u = proactive protection / patching rate
#
# Therefore NIO CANNOT maximize both defenses.
# It must decide how the limited security budget should be used.
#
# State:
#
#   S = susceptible hosts
#   I = infected hosts
#   R = removed/protected/isolated hosts
#   C = cumulative infection burden
#
# ============================================================


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

dt = 0.02
Time_steps = 1000
iters = 1200

N = 200

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device =", device)


# ------------------------------------------------------------
# Fixed security budget
# ------------------------------------------------------------
#
# q + u = B
#
# NIO decides how B is divided.
#
# ------------------------------------------------------------

B = 0.50


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

        # ---------------------------------------------
        # New infections
        # ---------------------------------------------

        infection = beta * S * I

        # ---------------------------------------------
        # Infected machines detected and isolated
        # ---------------------------------------------

        isolation = q * I

        # ---------------------------------------------
        # Susceptible machines proactively protected
        # ---------------------------------------------

        protection = u * S


        # ---------------------------------------------
        # Differential equations
        # ---------------------------------------------

        dS = -infection - protection

        dI = infection - isolation

        dR = isolation + protection

        # cumulative infection exposure

        dC = I


        # ---------------------------------------------
        # Euler integration
        # ---------------------------------------------

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
# NIO parameter bounds
# ============================================================
#
# Physical / cybersecurity quantities:
#
# I0:
#     0.1% - 10% initially compromised
#
# beta:
#     0.10 - 2.00 malware transmission rate
#
# alpha:
#     0 - 1
#
# alpha = 0
#     ALL security resources go to proactive protection
#
# alpha = 1
#     ALL security resources go to detection/isolation
#
# ============================================================


low = torch.tensor(
    [
        0.001,      # I0
        0.10,       # beta
        0.00        # alpha
    ],
    device=device
)


high = torch.tensor(
    [
        0.10,       # I0
        2.00,       # beta
        1.00        # alpha
    ],
    device=device
)


# ============================================================
# Convert NIO latent variables into real parameters
# ============================================================

def decode_parameters(z):

    p = low + (high - low) * torch.sigmoid(z)

    I0 = p[:, 0]

    beta = p[:, 1]

    alpha = p[:, 2]


    # --------------------------------------------------------
    # Fixed budget allocation
    # --------------------------------------------------------

    q = alpha * B

    u = (1.0 - alpha) * B


    return I0, beta, alpha, q, u


# ============================================================
# Simulate malware outbreak
# ============================================================

def simulate(I0, beta, q, u):

    S0 = 1.0 - I0

    R0 = torch.zeros_like(I0)

    C0 = torch.zeros_like(I0)


    state = torch.stack(
        [
            S0,
            I0,
            R0,
            C0
        ],
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
# Malware damage function
# ============================================================
#
# We care about TWO things:
#
# 1. cumulative infection burden
#
#       integral I(t) dt
#
# 2. peak fraction of network infected
#
# This prevents the optimizer from caring only about
# the final state.
#
# ============================================================

def malware_damage(trajectory):

    I = trajectory[:, :, 1]

    cumulative = trajectory[-1, :, 3]

    peak = torch.max(
        I,
        dim=0
    ).values


    damage = cumulative + 2.0 * peak


    return damage


# ============================================================
# EXPERIMENT 1
#
# NIO discovers the WORST malware conditions.
#
# Security policy is fixed at 50/50 here.
#
# This finds dangerous initial conditions:
#
#       I0
#       beta
#
# ============================================================

print("\n")
print("==============================================")
print("EXPERIMENT 1")
print("NIO SEARCHING FOR WORST MALWARE CONDITIONS")
print("==============================================")


# alpha = 0.5 corresponds to q = u = B/2

# We optimize only I0 and beta here.

z_attack = torch.randn(
    N,
    2,
    device=device,
    requires_grad=True
)


optimizer = optim.Adam(
    [z_attack],
    lr=0.01
)


for i in range(iters):

    # bounded I0

    I0 = (
        low[0]
        +
        (high[0] - low[0])
        * torch.sigmoid(z_attack[:, 0])
    )

    # bounded beta

    beta = (
        low[1]
        +
        (high[1] - low[1])
        * torch.sigmoid(z_attack[:, 1])
    )


    # Equal security allocation

    q = torch.ones_like(I0) * B / 2.0

    u = torch.ones_like(I0) * B / 2.0


    trajectory = simulate(
        I0,
        beta,
        q,
        u
    )


    damage = malware_damage(
        trajectory
    ).mean()


    # Negative because Adam minimizes

    loss = -damage


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if i % 100 == 0:

        print(
            "iter", i,
            "damage", damage.item(),
            "I0", I0.mean().item(),
            "beta", beta.mean().item()
        )


# ============================================================
# Save discovered dangerous conditions
# ============================================================

with torch.no_grad():

    I0_attack = (
        low[0]
        +
        (high[0] - low[0])
        * torch.sigmoid(z_attack[:, 0])
    )


    beta_attack = (
        low[1]
        +
        (high[1] - low[1])
        * torch.sigmoid(z_attack[:, 1])
    )


print("\nWorst-case conditions found:")

print(
    "I0   =",
    I0_attack.mean().item()
)

print(
    "beta =",
    beta_attack.mean().item()
)


# ============================================================
# EXPERIMENT 2
#
# NOW THE IMPORTANT EXPERIMENT.
#
# Hold the malware conditions found above fixed.
#
# NIO determines how to allocate the LIMITED security budget.
#
# ============================================================

print("\n")
print("==============================================")
print("EXPERIMENT 2")
print("NIO OPTIMIZING LIMITED SECURITY BUDGET")
print("==============================================")


# Freeze dangerous malware conditions

I0_fixed = I0_attack.detach()

beta_fixed = beta_attack.detach()


# One NIO variable:
#
# alpha = security budget allocation

z_policy = torch.randn(
    N,
    1,
    device=device,
    requires_grad=True
)


optimizer = optim.Adam(
    [z_policy],
    lr=0.01
)


for i in range(iters):

    alpha = torch.sigmoid(
        z_policy[:, 0]
    )


    # -----------------------------------------
    # Fixed total security budget
    # -----------------------------------------

    q = alpha * B

    u = (1.0 - alpha) * B


    trajectory = simulate(
        I0_fixed,
        beta_fixed,
        q,
        u
    )


    damage = malware_damage(
        trajectory
    ).mean()


    # Here we MINIMIZE malware damage

    loss = damage


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if i % 100 == 0:

        print(
            "iter", i,
            "damage", damage.item(),
            "alpha", alpha.mean().item(),
            "q", q.mean().item(),
            "u", u.mean().item()
        )


# ============================================================
# Final optimized policy
# ============================================================

with torch.no_grad():

    alpha_opt = torch.sigmoid(
        z_policy[:, 0]
    )

    q_opt = alpha_opt * B

    u_opt = (1.0 - alpha_opt) * B


    trajectory_opt = simulate(
        I0_fixed,
        beta_fixed,
        q_opt,
        u_opt
    )


print("\n")
print("==============================================")
print("FINAL NIO SECURITY POLICY")
print("==============================================")


print(
    "Security budget B =",
    B
)

print(
    "Fraction allocated to isolation =",
    alpha_opt.mean().item()
)

print(
    "Isolation q =",
    q_opt.mean().item()
)

print(
    "Protection u =",
    u_opt.mean().item()
)


# ============================================================
# Compare against simple policies
# ============================================================
#
# This is important.
#
# We compare NIO against:
#
#   100% protection
#   50/50
#   100% isolation
#
# ============================================================

with torch.no_grad():


    # --------------------------------------------------------
    # Policy A: all proactive protection
    # --------------------------------------------------------

    q_A = torch.zeros_like(I0_fixed)

    u_A = torch.ones_like(I0_fixed) * B


    traj_A = simulate(
        I0_fixed,
        beta_fixed,
        q_A,
        u_A
    )


    damage_A = malware_damage(
        traj_A
    ).mean()


    # --------------------------------------------------------
    # Policy B: equal allocation
    # --------------------------------------------------------

    q_B = torch.ones_like(I0_fixed) * B / 2.0

    u_B = torch.ones_like(I0_fixed) * B / 2.0


    traj_B = simulate(
        I0_fixed,
        beta_fixed,
        q_B,
        u_B
    )


    damage_B = malware_damage(
        traj_B
    ).mean()


    # --------------------------------------------------------
    # Policy C: all detection/isolation
    # --------------------------------------------------------

    q_C = torch.ones_like(I0_fixed) * B

    u_C = torch.zeros_like(I0_fixed)


    traj_C = simulate(
        I0_fixed,
        beta_fixed,
        q_C,
        u_C
    )


    damage_C = malware_damage(
        traj_C
    ).mean()


    # --------------------------------------------------------
    # NIO
    # --------------------------------------------------------

    damage_NIO = malware_damage(
        trajectory_opt
    ).mean()


print("\n")
print("==============================================")
print("POLICY COMPARISON")
print("==============================================")


print(
    "100% protection damage:",
    damage_A.item()
)

print(
    "50/50 damage:",
    damage_B.item()
)

print(
    "100% isolation damage:",
    damage_C.item()
)

print(
    "NIO optimized damage:",
    damage_NIO.item()
)


# ============================================================
# Plot infection trajectories
# ============================================================

with torch.no_grad():

    curve_A = traj_A[:, :, 1].mean(dim=1)

    curve_B = traj_B[:, :, 1].mean(dim=1)

    curve_C = traj_C[:, :, 1].mean(dim=1)

    curve_NIO = trajectory_opt[:, :, 1].mean(dim=1)


time = (
    torch.arange(Time_steps + 1)
    * dt
).cpu().numpy()


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    time,
    curve_A.cpu().numpy(),
    label="100% Protection"
)


plt.plot(
    time,
    curve_B.cpu().numpy(),
    label="50/50"
)


plt.plot(
    time,
    curve_C.cpu().numpy(),
    label="100% Isolation"
)


plt.plot(
    time,
    curve_NIO.cpu().numpy(),
    label="NIO Policy",
    linewidth=3
)


plt.xlabel(
    "Time"
)

plt.ylabel(
    "Fraction of Network Infected"
)

plt.title(
    "Malware Spread Under Limited Security Budget"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# IMPORTANT EXPERIMENT 3
#
# Sweep malware transmission rate.
#
# For EACH beta:
#
# let NIO discover the best security policy.
#
# This tells us whether the optimal policy changes as the
# malware becomes more aggressive.
#
# ============================================================


print("\n")
print("==============================================")
print("EXPERIMENT 3")
print("POLICY VS MALWARE TRANSMISSION RATE")
print("==============================================")


beta_values = torch.linspace(
    0.20,
    2.00,
    20
)


alpha_results = []

damage_results = []


# Use 5% initial compromise for this controlled experiment

I0_sweep_value = 0.05


for beta_value in beta_values:


    I0_sweep = torch.ones(
        N,
        device=device
    ) * I0_sweep_value


    beta_sweep = torch.ones(
        N,
        device=device
    ) * beta_value.item()


    z = torch.zeros(
        N,
        1,
        device=device,
        requires_grad=True
    )


    optimizer = optim.Adam(
        [z],
        lr=0.02
    )


    for i in range(600):


        alpha = torch.sigmoid(
            z[:, 0]
        )


        q = alpha * B

        u = (1.0 - alpha) * B


        trajectory = simulate(
            I0_sweep,
            beta_sweep,
            q,
            u
        )


        loss = malware_damage(
            trajectory
        ).mean()


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


    with torch.no_grad():

        alpha = torch.sigmoid(
            z[:, 0]
        )


        q = alpha * B

        u = (1.0 - alpha) * B


        trajectory = simulate(
            I0_sweep,
            beta_sweep,
            q,
            u
        )


        damage = malware_damage(
            trajectory
        ).mean()


        alpha_results.append(
            alpha.mean().item()
        )

        damage_results.append(
            damage.item()
        )


        print(
            "beta",
            beta_value.item(),
            "alpha",
            alpha.mean().item(),
            "q",
            q.mean().item(),
            "u",
            u.mean().item(),
            "damage",
            damage.item()
        )


# ============================================================
# Plot optimal policy versus malware aggressiveness
# ============================================================

plt.figure(
    figsize=(9, 5)
)


plt.plot(
    beta_values.numpy(),
    alpha_results,
    marker="o"
)


plt.axhline(
    0.5,
    linestyle="--"
)


plt.xlabel(
    "Malware Transmission Rate beta"
)

plt.ylabel(
    "Fraction of Security Budget Allocated to Isolation"
)

plt.title(
    "NIO Optimal Security Policy vs Malware Transmission Rate"
)

plt.grid(True)

plt.show()


# ============================================================
# Save results
# ============================================================

results = pd.DataFrame({

    "beta":
        beta_values.numpy(),

    "optimal_isolation_fraction":
        alpha_results,

    "optimal_protection_fraction":
        [1.0 - x for x in alpha_results],

    "damage":
        damage_results
})


results.to_csv(
    "nio_cybersecurity_policy_sweep.csv",
    index=False
)


print(
    "\nSaved: nio_cybersecurity_policy_sweep.csv"
)


```


## NIO + mathematical image/video dynamics


```

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


# ============================================================
# NIO EXPERIMENT 4
#
# VIDEO -> MATHEMATICAL STATE -> LEARNED DYNAMICS
#
# PURPOSE
#
# Imagine that the frames below came from a camera looking
# at moving water.
#
# We DO NOT give the system the equations that generated
# the frames.
#
# STEP 1:
#
#   Each observed frame I_t is approximated by explicit
#   differentiable mathematical primitives:
#
#       I_t ~= F(theta_t)
#
#   NIO discovers theta_t.
#
# STEP 2:
#
#   The sequence
#
#       theta_0, theta_1, theta_2, ...
#
#   becomes a mathematical state trajectory.
#
# STEP 3:
#
#   Learn:
#
#       theta_(t+1) = G(theta_t)
#
# STEP 4:
#
#   Starting from ONE mathematical state, recursively predict
#   future states and render future frames.
#
# IMPORTANT:
#
# The dynamics used to create the observations are NEVER
# supplied to NIO or to G.
#
# ============================================================


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

torch.manual_seed(7)
np.random.seed(7)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device =", device)


# ============================================================
# SETTINGS
# ============================================================

H = 64
W = 64

# Number of mathematical primitives

K = 10

# Number of observed frames

T = 36

# Frames used to learn dynamics

TRAIN_T = 24

# NIO iterations for first frame

FIRST_FRAME_ITERS = 1200

# NIO iterations for following frames

FOLLOW_FRAME_ITERS = 350

# Learning rate for image fitting

fit_lr = 0.025


# ============================================================
# IMAGE COORDINATES
# ============================================================

x = torch.linspace(
    -1.0,
    1.0,
    W,
    device=device
)

y = torch.linspace(
    -1.0,
    1.0,
    H,
    device=device
)

Y, X = torch.meshgrid(
    y,
    x,
    indexing="ij"
)


# ============================================================
# CREATE OBSERVED VIDEO
#
# This is our stand-in for a river video.
#
# IMPORTANT:
#
# The representation model below does NOT know these equations.
#
# The field contains:
#
#   translation
#   waves
#   changing phase
#   local structures
#   deformation
#
# ============================================================

def make_observed_frame(t):

    tau = t / float(T - 1)

    # --------------------------------------------------------
    # Background flowing wave field
    # --------------------------------------------------------

    wave1 = torch.sin(
        8.0 * X
        - 5.0 * tau
        + 2.0 * torch.sin(2.5 * Y)
    )

    wave2 = torch.sin(
        11.0 * X
        + 4.0 * Y
        - 8.0 * tau
    )

    background = (
        0.10 * wave1
        +
        0.07 * wave2
    )


    # --------------------------------------------------------
    # Moving elongated structure 1
    # --------------------------------------------------------

    cx1 = (
        -0.75
        +
        1.45 * tau
    )

    cy1 = (
        0.30
        * math.sin(
            2.0 * math.pi * tau
        )
    )

    theta1 = (
        0.35
        * math.sin(
            2.0 * math.pi * tau
        )
    )


    dx1 = X - cx1
    dy1 = Y - cy1

    ct1 = math.cos(theta1)
    st1 = math.sin(theta1)

    xr1 = (
        ct1 * dx1
        +
        st1 * dy1
    )

    yr1 = (
        -st1 * dx1
        +
        ct1 * dy1
    )

    structure1 = torch.exp(

        -(

            (xr1 / 0.32)**2

            +

            (yr1 / 0.11)**2

        )

    )


    # --------------------------------------------------------
    # Moving structure 2
    #
    # Different trajectory and speed.
    # --------------------------------------------------------

    cx2 = (
        0.65
        -
        1.05 * tau
    )

    cy2 = (
        -0.40
        +
        0.55 * tau
    )


    dx2 = X - cx2
    dy2 = Y - cy2


    structure2 = torch.exp(

        -(

            (dx2 / 0.18)**2

            +

            (dy2 / 0.22)**2

        )

    )


    # --------------------------------------------------------
    # Small oscillating structure
    # --------------------------------------------------------

    cx3 = (
        0.10
        +
        0.25
        * math.sin(
            4.0 * math.pi * tau
        )
    )

    cy3 = (
        0.55
        -
        0.75 * tau
    )


    structure3 = torch.exp(

        -(

            ((X - cx3) / 0.12)**2

            +

            ((Y - cy3) / 0.12)**2

        )

    )


    # --------------------------------------------------------
    # Combine into observed frame
    # --------------------------------------------------------

    image = (

        0.20

        +

        background

        +

        0.65 * structure1

        +

        0.45 * structure2

        +

        0.30 * structure3

    )


    image = torch.clamp(
        image,
        0.0,
        1.0
    )


    return image


# ============================================================
# GENERATE VIDEO
# ============================================================

observed_frames = []

for t in range(T):

    observed_frames.append(
        make_observed_frame(t)
    )


observed_frames = torch.stack(
    observed_frames
)


print(
    "observed video shape =",
    observed_frames.shape
)


# ============================================================
# SHOW SELECTED OBSERVED FRAMES
# ============================================================

show_times = [
    0,
    8,
    16,
    24,
    35
]


for t in show_times:

    plt.figure(figsize=(5,5))

    plt.imshow(
        observed_frames[t]
        .detach()
        .cpu()
        .numpy(),

        cmap="gray",

        origin="lower",

        vmin=0,

        vmax=1
    )

    plt.title(
        "Observed Frame t = "
        + str(t)
    )

    plt.axis("off")

    plt.show()


# ============================================================
# MATHEMATICAL IMAGE REPRESENTATION
#
# Each frame will be represented using K explicit primitives:
#
#
# B_k(x,y) =
#
# A_k exp(
#
#   -[
#
#      (x'_k/a_k)^2
#
#      +
#
#      (y'_k/b_k)^2
#
#    ]^p_k
#
# )
#
#
# Parameters per primitive:
#
#   x0
#   y0
#   a
#   b
#   theta
#   p
#   amplitude
#
# ============================================================


# ------------------------------------------------------------
# Parameter ranges
# ------------------------------------------------------------

def decode(z):

    x0 = torch.tanh(
        z[:,0]
    )


    y0 = torch.tanh(
        z[:,1]
    )


    a = (
        0.04
        +
        0.55
        * torch.sigmoid(
            z[:,2]
        )
    )


    b = (
        0.04
        +
        0.55
        * torch.sigmoid(
            z[:,3]
        )
    )


    theta = (
        math.pi
        *
        torch.tanh(
            z[:,4]
        )
    )


    p = (
        0.7
        +
        2.8
        * torch.sigmoid(
            z[:,5]
        )
    )


    amplitude = torch.sigmoid(
        z[:,6]
    )


    return (
        x0,
        y0,
        a,
        b,
        theta,
        p,
        amplitude
    )


# ============================================================
# DIFFERENTIABLE MATHEMATICAL RENDERER
# ============================================================

def render(z):

    (
        x0,
        y0,
        a,
        b,
        theta,
        p,
        amplitude

    ) = decode(z)


    field = torch.zeros_like(
        X
    )


    for k in range(K):

        dx = X - x0[k]

        dy = Y - y0[k]


        ct = torch.cos(
            theta[k]
        )

        st = torch.sin(
            theta[k]
        )


        xr = (
            ct * dx
            +
            st * dy
        )


        yr = (
            -st * dx
            +
            ct * dy
        )


        r2 = (

            (xr / a[k])**2

            +

            (yr / b[k])**2

        )


        blob = (

            amplitude[k]

            *

            torch.exp(

                -torch.pow(
                    r2 + 1e-8,
                    p[k]
                )

            )

        )


        field = (
            field
            +
            blob
        )


    # --------------------------------------------------------
    # Background intensity
    #
    # We intentionally keep the representation simple.
    # --------------------------------------------------------

    image = (

        0.15

        +

        (
            1.0
            -
            torch.exp(
                -field
            )
        )

    )


    return torch.clamp(
        image,
        0.0,
        1.0
    )


# ============================================================
# FIT ONE FRAME WITH NIO
# ============================================================

def fit_frame(
    target,
    z_start,
    iterations
):


    z = (
        z_start
        .clone()
        .detach()
        .requires_grad_(True)
    )


    optimizer = optim.Adam(
        [z],
        lr=fit_lr
    )


    for i in range(iterations):

        prediction = render(
            z
        )


        reconstruction_loss = torch.mean(

            (
                prediction
                -
                target
            )**2

        )


        # ----------------------------------------------------
        # Small parameter-motion penalty
        #
        # For frames after the first, this encourages primitive
        # identity to remain stable through time.
        #
        # ----------------------------------------------------

        continuity_loss = torch.mean(

            (
                z
                -
                z_start
            )**2

        )


        loss = (

            reconstruction_loss

            +

            0.0002
            *
            continuity_loss

        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


    return (
        z.detach(),
        reconstruction_loss.detach()
    )


# ============================================================
# NIO FIT ALL VIDEO FRAMES
#
# IMPORTANT:
#
# Frame t begins from the optimized parameters of frame t-1.
#
# This is tracking.
#
# ============================================================

print("\n")
print("==============================================")
print("NIO CONVERTING VIDEO INTO MATHEMATICAL STATES")
print("==============================================")


z_current = torch.randn(
    K,
    7,
    device=device
)


states = []

fit_errors = []


for t in range(T):


    if t == 0:

        iterations = FIRST_FRAME_ITERS

    else:

        iterations = FOLLOW_FRAME_ITERS


    z_current, error = fit_frame(

        observed_frames[t],

        z_current,

        iterations

    )


    states.append(
        z_current.clone()
    )


    fit_errors.append(
        error.item()
    )


    print(
        "frame",
        t,
        "MSE",
        error.item()
    )


# ============================================================
# STACK MATHEMATICAL STATES
#
# Shape:
#
#       [T,K,7]
#
# ============================================================

states = torch.stack(
    states
)


print(
    "\nmathematical state shape =",
    states.shape
)


# ============================================================
# RENDER RECONSTRUCTED OBSERVED FRAMES
# ============================================================

reconstructed_frames = []


with torch.no_grad():

    for t in range(T):

        reconstructed_frames.append(

            render(
                states[t]
            )

        )


reconstructed_frames = torch.stack(
    reconstructed_frames
)


# ============================================================
# SHOW OBSERVATION VS MATHEMATICAL RECONSTRUCTION
# ============================================================

for t in show_times:


    plt.figure(figsize=(5,5))

    plt.imshow(

        observed_frames[t]
        .cpu()
        .numpy(),

        cmap="gray",

        origin="lower",

        vmin=0,

        vmax=1
    )

    plt.title(
        "Observed t = "
        + str(t)
    )

    plt.axis("off")

    plt.show()


    plt.figure(figsize=(5,5))

    plt.imshow(

        reconstructed_frames[t]
        .cpu()
        .numpy(),

        cmap="gray",

        origin="lower",

        vmin=0,

        vmax=1
    )

    plt.title(
        "Mathematical Reconstruction t = "
        + str(t)
    )

    plt.axis("off")

    plt.show()


# ============================================================
# PLOT FRAME FIT ERROR
# ============================================================

plt.figure(
    figsize=(9,5)
)


plt.plot(
    fit_errors,
    marker="o"
)


plt.xlabel(
    "Video Frame"
)

plt.ylabel(
    "NIO Reconstruction MSE"
)

plt.title(
    "Video -> Mathematical Representation Error"
)

plt.grid(True)

plt.show()


# ============================================================
# ============================================================
#
# LEARN DYNAMICS OF THE MATHEMATICAL STATE
#
# ============================================================
# ============================================================


# ============================================================
# Flatten explicit mathematical state
#
# theta_t has:
#
#       K x 7
#
# parameters.
#
# ============================================================

state_vectors = states.reshape(
    T,
    K * 7
)


state_dim = K * 7


print(
    "state dimension =",
    state_dim
)


# ============================================================
# IMPORTANT:
#
# We learn dynamics only from the FIRST TRAIN_T frames.
#
# Frames after TRAIN_T are held out.
#
# ============================================================

train_states = state_vectors[
    :TRAIN_T
]


# ============================================================
# NORMALIZE STATE VARIABLES
#
# Different mathematical parameters have different scales.
# ============================================================

state_mean = train_states.mean(
    dim=0
)


state_std = train_states.std(
    dim=0
) + 1e-6


normalized_states = (

    state_vectors
    -
    state_mean

) / state_std


# ============================================================
# TRAINING PAIRS
#
# theta_t -> theta_(t+1)
# ============================================================

X_train = normalized_states[
    :TRAIN_T-1
]


Y_train = normalized_states[
    1:TRAIN_T
]


# ============================================================
# LEARN A SIMPLE LINEAR DYNAMICAL SYSTEM
#
#       theta_(t+1) = A theta_t + b
#
# WHY LINEAR FIRST?
#
# Because if this works, the result is much more interesting
# than immediately using a large neural network.
#
# ============================================================

ones = torch.ones(

    X_train.shape[0],
    1,

    device=device
)


X_aug = torch.cat(

    [
        X_train,
        ones
    ],

    dim=1
)


# ------------------------------------------------------------
# Least-squares solution
#
# X_aug @ W ~= Y_train
#
# ------------------------------------------------------------

W_dyn = torch.linalg.lstsq(

    X_aug,

    Y_train

).solution


print(
    "dynamics matrix shape =",
    W_dyn.shape
)


# ============================================================
# ONE-STEP TRAINING ERROR
# ============================================================

train_prediction = (

    X_aug
    @
    W_dyn

)


train_dynamics_mse = torch.mean(

    (
        train_prediction
        -
        Y_train
    )**2

)


print(
    "\nOne-step normalized dynamics training MSE =",
    train_dynamics_mse.item()
)


# ============================================================
# RECURSIVE FUTURE PREDICTION
#
# This is the important test.
#
# We start at the LAST TRAINING state.
#
# After this point the model receives NO real future states.
#
# It recursively evolves its own prediction.
#
# ============================================================

current_state = normalized_states[
    TRAIN_T - 1
]


predicted_states = []


for t in range(
    TRAIN_T,
    T
):


    current_aug = torch.cat(

        [
            current_state,

            torch.ones(
                1,
                device=device
            )
        ]

    )


    next_state = (
        current_aug
        @
        W_dyn
    )


    predicted_states.append(
        next_state
    )


    # --------------------------------------------------------
    # Critical:
    #
    # prediction becomes next input
    # --------------------------------------------------------

    current_state = next_state


predicted_states = torch.stack(
    predicted_states
)


# ============================================================
# Convert back to original mathematical parameter scale
# ============================================================

predicted_state_vectors = (

    predicted_states
    *
    state_std

    +
    state_mean

)


predicted_z = predicted_state_vectors.reshape(

    T - TRAIN_T,
    K,
    7
)


# ============================================================
# RENDER PREDICTED FUTURE VIDEO
# ============================================================

predicted_frames = []


with torch.no_grad():

    for t in range(
        T - TRAIN_T
    ):

        predicted_frames.append(

            render(
                predicted_z[t]
            )

        )


predicted_frames = torch.stack(
    predicted_frames
)


# ============================================================
# FUTURE IMAGE ERROR
#
# Compare predictions with held-out observations.
# ============================================================

future_errors = []


for j in range(
    T - TRAIN_T
):


    true_t = TRAIN_T + j


    mse = torch.mean(

        (
            predicted_frames[j]

            -

            observed_frames[true_t]

        )**2

    )


    future_errors.append(
        mse.item()
    )


    print(
        "future frame",
        true_t,
        "prediction MSE",
        mse.item()
    )


# ============================================================
# SHOW HELD-OUT FUTURE PREDICTIONS
# ============================================================

future_show = [

    TRAIN_T,

    min(
        TRAIN_T + 3,
        T - 1
    ),

    min(
        TRAIN_T + 7,
        T - 1
    ),

    T - 1
]


for true_t in future_show:


    j = (
        true_t
        -
        TRAIN_T
    )


    # --------------------------------------------------------
    # Actual future observation
    # --------------------------------------------------------

    plt.figure(
        figsize=(5,5)
    )


    plt.imshow(

        observed_frames[
            true_t
        ]
        .cpu()
        .numpy(),

        cmap="gray",

        origin="lower",

        vmin=0,

        vmax=1
    )


    plt.title(
        "TRUE Future Frame t = "
        +
        str(true_t)
    )


    plt.axis("off")

    plt.show()


    # --------------------------------------------------------
    # Predicted future
    # --------------------------------------------------------

    plt.figure(
        figsize=(5,5)
    )


    plt.imshow(

        predicted_frames[
            j
        ]
        .cpu()
        .numpy(),

        cmap="gray",

        origin="lower",

        vmin=0,

        vmax=1
    )


    plt.title(
        "PREDICTED From Mathematical Dynamics t = "
        +
        str(true_t)
    )


    plt.axis("off")

    plt.show()


# ============================================================
# FUTURE ERROR VS PREDICTION HORIZON
# ============================================================

prediction_horizon = np.arange(
    1,
    T - TRAIN_T + 1
)


plt.figure(
    figsize=(9,5)
)


plt.plot(

    prediction_horizon,

    future_errors,

    marker="o"
)


plt.xlabel(
    "Recursive Prediction Horizon"
)

plt.ylabel(
    "Image MSE"
)

plt.title(
    "Future Prediction Error"
)

plt.grid(True)

plt.show()


# ============================================================
# VISUALIZE SOME DISCOVERED MATHEMATICAL PARAMETERS
#
# Track x and y positions of the first few primitives.
#
# ============================================================

decoded_x = []
decoded_y = []


with torch.no_grad():

    for t in range(T):

        (
            x0,
            y0,
            a,
            b,
            theta,
            p,
            amplitude

        ) = decode(
            states[t]
        )


        decoded_x.append(
            x0.cpu().numpy()
        )

        decoded_y.append(
            y0.cpu().numpy()
        )


decoded_x = np.array(
    decoded_x
)

decoded_y = np.array(
    decoded_y
)


plt.figure(
    figsize=(9,5)
)


for k in range(
    min(K,5)
):

    plt.plot(

        range(T),

        decoded_x[:,k],

        label=
        "Primitive "
        +
        str(k+1)
    )


plt.xlabel(
    "Frame"
)

plt.ylabel(
    "x Position"
)

plt.title(
    "Evolution of Mathematical Primitive Positions"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# SAVE MATHEMATICAL STATE TRAJECTORY
# ============================================================

rows = []


with torch.no_grad():

    for t in range(T):


        (
            x0,
            y0,
            a,
            b,
            theta,
            p,
            amplitude

        ) = decode(
            states[t]
        )


        for k in range(K):

            rows.append({

                "frame":
                    t,

                "primitive":
                    k + 1,

                "x0":
                    x0[k].item(),

                "y0":
                    y0[k].item(),

                "a":
                    a[k].item(),

                "b":
                    b[k].item(),

                "theta":
                    theta[k].item(),

                "p":
                    p[k].item(),

                "amplitude":
                    amplitude[k].item()

            })


df = pd.DataFrame(
    rows
)


df.to_csv(

    "nio_video_mathematical_states.csv",

    index=False
)


# ============================================================
# SAVE FUTURE PREDICTION RESULTS
# ============================================================

future_df = pd.DataFrame({

    "frame":
        list(
            range(
                TRAIN_T,
                T
            )
        ),

    "prediction_horizon":
        prediction_horizon,

    "image_mse":
        future_errors

})


future_df.to_csv(

    "nio_video_future_prediction.csv",

    index=False
)


print(
    "\nSaved: nio_video_mathematical_states.csv"
)

print(
    "Saved: nio_video_future_prediction.csv"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("==============================================")
print("FINAL SUMMARY")
print("==============================================")


print(
    "Mean frame reconstruction MSE =",
    np.mean(
        fit_errors
    )
)


print(
    "Mean held-out future prediction MSE =",
    np.mean(
        future_errors
    )
)


print(
    "Last-frame prediction MSE =",
    future_errors[-1]
)

```











## NIO + geometry / vibrating string


```

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import math


# ============================================================
# NIO EXPERIMENT 2
#
# OPTIMIZATION OF AN INITIAL GEOMETRY
# THROUGH THE 1-D WAVE EQUATION
#
# Physical system:
#
#       d2y/dt2 = c^2 d2y/dx2
#
# Fixed boundary conditions:
#
#       y(0,t) = 0
#       y(L,t) = 0
#
# Initial geometry:
#
#       y(x,0) =
#           sum A_n sin(n*pi*x/L)
#
# NIO optimizes the PHYSICAL NORMAL-MODE AMPLITUDES A_n.
#
# Important:
#
# We impose a fixed initial modal energy.
#
# Therefore NIO cannot simply increase all amplitudes.
#
# It must discover HOW to distribute a fixed amount
# of initial energy among the physical vibration modes.
#
# Research question:
#
# What initial geometry causes the string to produce
# the largest displacement at a specified location
# and future time?
#
# ============================================================


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device =", device)


L = 1.0                # string length

c = 1.0                # wave speed

Nx = 101               # spatial grid points

dx = L / (Nx - 1)

dt = 0.004             # stable because c*dt/dx < 1

Time_steps = 500

iters = 1500

N = 200                # independent NIO searches

K = 8                  # number of physical normal modes


# ------------------------------------------------------------
# Courant number
#
# For explicit wave equation:
#
#       c dt / dx <= 1
#
# ------------------------------------------------------------

courant = c * dt / dx

print("Courant number =", courant)

assert courant <= 1.0


# ============================================================
# Spatial grid
# ============================================================

x = torch.linspace(
    0.0,
    L,
    Nx,
    device=device
)


# ============================================================
# Physical normal modes
#
# phi_n(x) = sin(n*pi*x/L)
#
# Shape:
#
#       [K, Nx]
#
# ============================================================

modes = []

for n in range(1, K + 1):

    phi = torch.sin(
        n * math.pi * x / L
    )

    modes.append(phi)


modes = torch.stack(modes)


print(
    "mode matrix shape =",
    modes.shape
)


# ============================================================
# Wave-equation step
# ============================================================

class WaveStep(nn.Module):

    def __init__(self, c, dt, dx):

        super().__init__()

        self.r = (c * dt / dx) ** 2


    def forward(self, y_previous, y_current):

        # ----------------------------------------------------
        # y_previous:
        #       y at time t-dt
        #
        # y_current:
        #       y at time t
        #
        # ----------------------------------------------------

        y_next = torch.zeros_like(
            y_current
        )


        # ----------------------------------------------------
        # Finite difference wave equation
        #
        # y_i^(t+1) =
        #
        #   2 y_i^t
        # - y_i^(t-1)
        # + r (y_(i+1)^t - 2y_i^t + y_(i-1)^t)
        #
        # ----------------------------------------------------

        y_next[:, 1:-1] = (

            2.0 * y_current[:, 1:-1]

            - y_previous[:, 1:-1]

            + self.r * (

                y_current[:, 2:]

                - 2.0 * y_current[:, 1:-1]

                + y_current[:, :-2]

            )
        )


        # ----------------------------------------------------
        # Fixed endpoints
        # ----------------------------------------------------

        y_next[:, 0] = 0.0

        y_next[:, -1] = 0.0


        return y_next


model = WaveStep(
    c=c,
    dt=dt,
    dx=dx
).to(device)


# ============================================================
# FIXED INITIAL ENERGY
# ============================================================
#
# This is extremely important.
#
# Without a constraint, NIO could simply make the
# initial displacement arbitrarily large.
#
# We instead give every candidate exactly the same
# modal energy budget.
#
# For a stretched string, mode energy scales roughly as:
#
#       n^2 A_n^2
#
# when initial velocity is zero.
#
# Therefore:
#
#       sum n^2 A_n^2 = E
#
# is our normalized fixed-energy constraint.
#
# ============================================================

E = 1.0


mode_numbers = torch.arange(
    1,
    K + 1,
    dtype=torch.float32,
    device=device
)


# ============================================================
# Decode NIO variables
# ============================================================
#
# z contains unconstrained optimization variables.
#
# We normalize them so that:
#
#       sum n^2 A_n^2 = E
#
# exactly.
#
# ============================================================

def decode_amplitudes(z):

    # --------------------------------------------------------
    # Normalize in energy coordinates.
    #
    # Let:
    #
    #       b_n = n A_n
    #
    # Then:
    #
    #       sum b_n^2 = E
    #
    # --------------------------------------------------------

    norm = torch.sqrt(

        torch.sum(
            z**2,
            dim=1,
            keepdim=True
        )

        + 1e-12
    )


    b = math.sqrt(E) * z / norm


    A = b / mode_numbers


    return A


# ============================================================
# Convert physical modal amplitudes into string geometry
# ============================================================

def amplitudes_to_geometry(A):

    # A:
    #       [N,K]
    #
    # modes:
    #       [K,Nx]
    #
    # result:
    #       [N,Nx]

    y0 = A @ modes

    return y0


# ============================================================
# Simulate wave equation
# ============================================================

def simulate(y0):

    # --------------------------------------------------------
    # Initial velocity is zero.
    #
    # For zero initial velocity:
    #
    #       y(-dt) approximately y(0)
    #
    # --------------------------------------------------------

    y_previous = y0

    y_current = y0


    trajectory = [y_current]


    for t in range(Time_steps):

        y_next = model(
            y_previous,
            y_current
        )


        y_previous = y_current

        y_current = y_next


        trajectory.append(
            y_current
        )


    return torch.stack(
        trajectory
    )


# ============================================================
# Target location and target time
# ============================================================
#
# We deliberately choose a point that is NOT the center.
#
# x = 0.72 L
#
# This makes the spatial problem less symmetric.
#
# ============================================================

target_x_fraction = 0.72


target_index = int(
    target_x_fraction
    * (Nx - 1)
)


target_time_step = 375


print(
    "target x =",
    x[target_index].item()
)

print(
    "target time =",
    target_time_step * dt
)


# ============================================================
# NIO
#
# Find the initial STRING GEOMETRY that produces the
# largest positive displacement at the target location
# and target future time.
#
# ============================================================

print("\n")
print("==============================================")
print("NIO OPTIMIZING INITIAL STRING GEOMETRY")
print("==============================================")


z_init = torch.randn(
    N,
    K,
    device=device,
    requires_grad=True
)


optimizer = optim.Adam(
    [z_init],
    lr=0.01
)


for i in range(iters):

    # --------------------------------------------------------
    # Physical normal-mode amplitudes
    # --------------------------------------------------------

    A = decode_amplitudes(
        z_init
    )


    # --------------------------------------------------------
    # Initial physical geometry
    # --------------------------------------------------------

    y0 = amplitudes_to_geometry(
        A
    )


    # --------------------------------------------------------
    # Evolve through actual wave equation
    # --------------------------------------------------------

    trajectory = simulate(
        y0
    )


    # --------------------------------------------------------
    # Displacement at desired future location/time
    # --------------------------------------------------------

    target_displacement = trajectory[
        target_time_step,
        :,
        target_index
    ]


    # --------------------------------------------------------
    # MAXIMIZE positive displacement.
    #
    # Adam minimizes, therefore negative sign.
    # --------------------------------------------------------

    loss = -target_displacement.mean()


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if i % 100 == 0:

        print(
            "iter",
            i,
            "target displacement",
            target_displacement.mean().item()
        )


# ============================================================
# Final NIO result
# ============================================================

with torch.no_grad():

    A_opt = decode_amplitudes(
        z_init
    )


    y0_opt = amplitudes_to_geometry(
        A_opt
    )


    trajectory_opt = simulate(
        y0_opt
    )


# ============================================================
# Average optimized mode amplitudes
# ============================================================

A_mean = A_opt.mean(
    dim=0
)


print("\n")
print("==============================================")
print("OPTIMIZED PHYSICAL MODE AMPLITUDES")
print("==============================================")


for n in range(K):

    print(
        "Mode",
        n + 1,
        "A =",
        A_mean[n].item()
    )


# ============================================================
# Verify energy constraint
# ============================================================

energy = torch.sum(

    (
        mode_numbers[None, :]
        * A_opt
    ) ** 2,

    dim=1
)


print("\nMean normalized initial energy =",
      energy.mean().item())


# ============================================================
# Choose representative optimized solution
#
# Since all N searches should converge toward equivalent
# solutions, use candidate 0 for visualization.
# ============================================================

candidate = 0


initial_shape = (
    y0_opt[candidate]
    .cpu()
    .numpy()
)


target_shape = (
    trajectory_opt[
        target_time_step,
        candidate
    ]
    .cpu()
    .numpy()
)


# ============================================================
# Plot initial geometry
# ============================================================

plt.figure(
    figsize=(10, 5)
)


plt.plot(
    x.cpu().numpy(),
    initial_shape,
    linewidth=2
)


plt.axhline(
    0.0,
    linestyle="--"
)


plt.xlabel(
    "Position Along String"
)

plt.ylabel(
    "Displacement"
)

plt.title(
    "NIO-Discovered Initial String Geometry"
)

plt.grid(True)

plt.show()


# ============================================================
# Plot geometry at target future time
# ============================================================

plt.figure(
    figsize=(10, 5)
)


plt.plot(
    x.cpu().numpy(),
    target_shape,
    linewidth=2
)


plt.scatter(

    [x[target_index].cpu().item()],

    [
        trajectory_opt[
            target_time_step,
            candidate,
            target_index
        ].cpu().item()
    ],

    s=80
)


plt.axhline(
    0.0,
    linestyle="--"
)


plt.xlabel(
    "Position Along String"
)

plt.ylabel(
    "Displacement"
)

plt.title(
    "String Geometry at NIO Target Time"
)

plt.grid(True)

plt.show()


# ============================================================
# Plot target-point motion through time
# ============================================================

target_motion = (

    trajectory_opt[
        :,
        candidate,
        target_index
    ]

    .cpu()
    .numpy()
)


time = (

    torch.arange(
        Time_steps + 1
    )

    * dt

).numpy()


plt.figure(
    figsize=(10, 5)
)


plt.plot(
    time,
    target_motion,
    linewidth=2
)


plt.axvline(
    target_time_step * dt,
    linestyle="--",
    label="NIO Target Time"
)


plt.xlabel(
    "Time"
)

plt.ylabel(
    "Displacement at Target Position"
)

plt.title(
    "Motion of Target Point"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# Space-time image
#
# This is probably the most interesting visualization.
#
# Horizontal axis = position
# Vertical axis   = time
#
# Shows the entire geometry evolving.
# ============================================================

space_time = (

    trajectory_opt[
        :,
        candidate,
        :
    ]

    .cpu()
    .numpy()
)


plt.figure(
    figsize=(10, 7)
)


plt.imshow(

    space_time,

    aspect="auto",

    origin="lower",

    extent=[
        0,
        L,
        0,
        Time_steps * dt
    ]
)


plt.colorbar(
    label="String Displacement"
)


plt.scatter(

    [
        x[
            target_index
        ].cpu().item()
    ],

    [
        target_time_step
        * dt
    ],

    marker="x",

    s=100
)


plt.xlabel(
    "Position Along String"
)

plt.ylabel(
    "Time"
)

plt.title(
    "NIO-Optimized String Dynamics"
)

plt.show()


# ============================================================
# BASELINE COMPARISON
#
# Compare NIO against putting the SAME energy entirely
# into each individual physical normal mode.
#
# This is important:
#
# Maybe NIO merely discovers one obvious normal mode.
#
# We want to know whether combining modes gives a larger
# target displacement than any single-mode initial condition.
# ============================================================

print("\n")
print("==============================================")
print("SINGLE-MODE BASELINES")
print("==============================================")


baseline_results = []


with torch.no_grad():

    for n in range(K):

        A_single = torch.zeros(
            1,
            K,
            device=device
        )


        # ----------------------------------------------------
        # Fixed energy:
        #
        #       n^2 A_n^2 = E
        #
        # Therefore:
        #
        #       A_n = sqrt(E)/n
        #
        # ----------------------------------------------------

        A_single[
            0,
            n
        ] = (
            math.sqrt(E)
            /
            float(n + 1)
        )


        y0_single = amplitudes_to_geometry(
            A_single
        )


        traj_single = simulate(
            y0_single
        )


        displacement = traj_single[
            target_time_step,
            0,
            target_index
        ].item()


        # Also test opposite sign because positive target
        # displacement may require negative initial mode.

        A_single_negative = (
            -A_single
        )


        y0_single_negative = amplitudes_to_geometry(
            A_single_negative
        )


        traj_single_negative = simulate(
            y0_single_negative
        )


        displacement_negative = traj_single_negative[
            target_time_step,
            0,
            target_index
        ].item()


        best_single = max(
            displacement,
            displacement_negative
        )


        baseline_results.append(
            best_single
        )


        print(
            "Mode",
            n + 1,
            "best target displacement =",
            best_single
        )


# ============================================================
# NIO result
# ============================================================

nio_target = (

    trajectory_opt[
        target_time_step,
        :,
        target_index
    ]

    .mean()

    .item()
)


best_baseline = max(
    baseline_results
)


print("\n")
print("==============================================")
print("COMPARISON")
print("==============================================")


print(
    "Best single physical mode =",
    best_baseline
)


print(
    "NIO optimized geometry =",
    nio_target
)


print(
    "NIO improvement =",
    nio_target - best_baseline
)


# ============================================================
# Save optimized mode amplitudes
# ============================================================

rows = []


A_cpu = (
    A_opt
    .cpu()
    .numpy()
)


for j in range(N):

    row = {

        "candidate":
            j,

        "target_displacement":
            trajectory_opt[
                target_time_step,
                j,
                target_index
            ].cpu().item()
    }


    for n in range(K):

        row[
            "A_mode_" + str(n + 1)
        ] = A_cpu[j, n]


    rows.append(
        row
    )


df = pd.DataFrame(
    rows
)


df.to_csv(
    "nio_string_geometry.csv",
    index=False
)


print(
    "\nSaved: nio_string_geometry.csv"
)

```






## Information theory + dynamical systems




```




```


## image → mathematical equations using NIO


```

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import math


# ============================================================
# NIO EXPERIMENT 3
#
# IMAGE -> MATHEMATICAL EQUATIONS
#
# Question:
#
# Can NIO reconstruct an image by optimizing the parameters
# of explicit differentiable mathematical functions?
#
# Each primitive is:
#
# B_k(x,y) =
#
# exp( -( (x'/a)^2 + (y'/b)^2 )^p )
#
# where rotated coordinates are:
#
# x' = cos(theta)(x-x0) + sin(theta)(y-y0)
# y' =-sin(theta)(x-x0) + cos(theta)(y-y0)
#
# NIO learns:
#
# x0     position
# y0     position
# a      horizontal scale
# b      vertical scale
# theta  rotation
# p      shape exponent
# A      intensity/amplitude
#
# The final image is the differentiable combination of blobs.
#
# ============================================================


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device =", device)

H = 96
W = 96

K = 20

iters = 3000

lr = 0.02


# ============================================================
# Coordinate system
#
# Image coordinates are normalized to:
#
#       x in [-1,1]
#       y in [-1,1]
#
# ============================================================

x = torch.linspace(
    -1.0,
    1.0,
    W,
    device=device
)

y = torch.linspace(
    -1.0,
    1.0,
    H,
    device=device
)

Y, X = torch.meshgrid(
    y,
    x,
    indexing="ij"
)


# ============================================================
# TARGET IMAGE
#
# We construct a recognizable target ONLY to provide pixels.
#
# The optimizer does not receive these target parameters.
#
# Target:
#
#     crescent / curved body
#     +
#     small circular object
#
# This is intentionally not simply one ellipse.
#
# ============================================================


def soft_circle(X, Y, cx, cy, radius, sharpness=80.0):

    distance = torch.sqrt(
        (X - cx)**2
        +
        (Y - cy)**2
        +
        1e-12
    )

    return torch.sigmoid(
        sharpness
        *
        (radius - distance)
    )


# Large disk

disk1 = soft_circle(
    X,
    Y,
    -0.15,
    0.00,
    0.58
)


# Shifted disk removes part of first disk

disk2 = soft_circle(
    X,
    Y,
    0.08,
    -0.02,
    0.50
)


crescent = torch.relu(
    disk1 - disk2
)


# Small second object

circle2 = soft_circle(
    X,
    Y,
    0.48,
    -0.42,
    0.12
)


target = torch.clamp(
    crescent + circle2,
    0.0,
    1.0
)


# ============================================================
# Display target
# ============================================================

plt.figure(figsize=(5,5))

plt.imshow(
    target.detach().cpu().numpy(),
    cmap="gray",
    origin="lower",
    extent=[-1,1,-1,1]
)

plt.title("Target Image")

plt.xlabel("x")
plt.ylabel("y")

plt.show()


# ============================================================
# NIO PARAMETERS
#
# One unconstrained vector per mathematical primitive.
#
# Parameters:
#
# 0 = x0
# 1 = y0
# 2 = a
# 3 = b
# 4 = theta
# 5 = p
# 6 = amplitude
#
# ============================================================

z = torch.randn(
    K,
    7,
    device=device,
    requires_grad=True
)


# ============================================================
# Decode unconstrained NIO variables
#
# We use sigmoid/tanh to enforce sensible ranges.
# ============================================================

def decode(z):

    # position

    x0 = torch.tanh(
        z[:,0]
    )

    y0 = torch.tanh(
        z[:,1]
    )


    # scale

    a = (
        0.03
        +
        0.60
        * torch.sigmoid(z[:,2])
    )

    b = (
        0.03
        +
        0.60
        * torch.sigmoid(z[:,3])
    )


    # rotation

    theta = (
        math.pi
        *
        torch.tanh(z[:,4])
    )


    # shape exponent
    #
    # p near 1 -> Gaussian-like ellipse
    #
    # larger p -> increasingly flat / sharp shape

    p = (
        0.7
        +
        3.3
        * torch.sigmoid(z[:,5])
    )


    # intensity

    amplitude = torch.sigmoid(
        z[:,6]
    )


    return (
        x0,
        y0,
        a,
        b,
        theta,
        p,
        amplitude
    )


# ============================================================
# Differentiable mathematical renderer
# ============================================================

def render(z):

    (
        x0,
        y0,
        a,
        b,
        theta,
        p,
        amplitude

    ) = decode(z)


    image_sum = torch.zeros_like(
        X
    )


    for k in range(K):

        # --------------------------------------------
        # translate coordinates
        # --------------------------------------------

        dx = X - x0[k]

        dy = Y - y0[k]


        # --------------------------------------------
        # rotate coordinates
        # --------------------------------------------

        ct = torch.cos(
            theta[k]
        )

        st = torch.sin(
            theta[k]
        )


        xr = (
            ct * dx
            +
            st * dy
        )

        yr = (
            -st * dx
            +
            ct * dy
        )


        # --------------------------------------------
        # elliptical distance
        # --------------------------------------------

        r2 = (

            (xr / a[k])**2

            +

            (yr / b[k])**2

        )


        # --------------------------------------------
        # explicit mathematical primitive
        # --------------------------------------------

        blob = (

            amplitude[k]

            *

            torch.exp(

                -torch.pow(
                    r2 + 1e-8,
                    p[k]
                )

            )
        )


        image_sum = (
            image_sum
            +
            blob
        )


    # ------------------------------------------------
    # Differentiable saturation.
    #
    # Converts accumulated mathematical fields
    # into image intensity between 0 and 1.
    # ------------------------------------------------

    image = (
        1.0
        -
        torch.exp(
            -image_sum
        )
    )


    return image


# ============================================================
# LOSS
# ============================================================
#
# Main term:
#
#       pixel reconstruction error
#
# Complexity term:
#
#       discourage unnecessary blob intensity
#
# This is a small regularizer.
#
# ============================================================

def loss_function(prediction, target, z):

    reconstruction = torch.mean(
        (prediction - target)**2
    )


    (
        x0,
        y0,
        a,
        b,
        theta,
        p,
        amplitude

    ) = decode(z)


    complexity = torch.mean(
        amplitude
    )


    loss = (

        reconstruction

        +

        0.0005
        *
        complexity

    )


    return (
        loss,
        reconstruction,
        complexity
    )


# ============================================================
# NIO OPTIMIZATION
# ============================================================

optimizer = optim.Adam(
    [z],
    lr=lr
)


loss_history = []


print("\n")
print("==============================================")
print("NIO IMAGE -> EQUATIONS")
print("==============================================")


for i in range(iters):

    prediction = render(z)


    (
        loss,
        reconstruction,
        complexity

    ) = loss_function(
        prediction,
        target,
        z
    )


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    loss_history.append(
        reconstruction.item()
    )


    if i % 200 == 0:

        print(
            "iter",
            i,
            "loss",
            loss.item(),
            "reconstruction",
            reconstruction.item()
        )


# ============================================================
# Final reconstruction
# ============================================================

with torch.no_grad():

    reconstruction = render(z)


# ============================================================
# Plot loss
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(
    loss_history
)

plt.xlabel(
    "NIO Iteration"
)

plt.ylabel(
    "Image MSE"
)

plt.title(
    "NIO Reconstruction Error"
)

plt.grid(True)

plt.show()


# ============================================================
# TARGET
# ============================================================

plt.figure(figsize=(5,5))

plt.imshow(
    target.cpu().numpy(),
    cmap="gray",
    origin="lower",
    extent=[-1,1,-1,1],
    vmin=0,
    vmax=1
)

plt.title(
    "Target Image"
)

plt.xlabel("x")
plt.ylabel("y")

plt.show()


# ============================================================
# NIO RECONSTRUCTION
# ============================================================

plt.figure(figsize=(5,5))

plt.imshow(
    reconstruction.cpu().numpy(),
    cmap="gray",
    origin="lower",
    extent=[-1,1,-1,1],
    vmin=0,
    vmax=1
)

plt.title(
    "NIO Mathematical Reconstruction"
)

plt.xlabel("x")
plt.ylabel("y")

plt.show()


# ============================================================
# ERROR IMAGE
# ============================================================

error_image = torch.abs(
    target
    -
    reconstruction
)


plt.figure(figsize=(5,5))

plt.imshow(
    error_image.cpu().numpy(),
    origin="lower",
    extent=[-1,1,-1,1]
)

plt.colorbar(
    label="Absolute Error"
)

plt.title(
    "Reconstruction Error"
)

plt.xlabel("x")
plt.ylabel("y")

plt.show()


# ============================================================
# Print final mathematical parameters
# ============================================================

with torch.no_grad():

    (
        x0,
        y0,
        a,
        b,
        theta,
        p,
        amplitude

    ) = decode(z)


print("\n")
print("==============================================")
print("DISCOVERED MATHEMATICAL DESCRIPTION")
print("==============================================")


for k in range(K):

    print(
        "\nBlob",
        k + 1
    )

    print(
        "x0        =",
        x0[k].item()
    )

    print(
        "y0        =",
        y0[k].item()
    )

    print(
        "a         =",
        a[k].item()
    )

    print(
        "b         =",
        b[k].item()
    )

    print(
        "theta     =",
        theta[k].item()
    )

    print(
        "p         =",
        p[k].item()
    )

    print(
        "amplitude =",
        amplitude[k].item()
    )


# ============================================================
# Save equations / parameters
# ============================================================

rows = []


for k in range(K):

    rows.append({

        "blob":
            k + 1,

        "x0":
            x0[k].item(),

        "y0":
            y0[k].item(),

        "a":
            a[k].item(),

        "b":
            b[k].item(),

        "theta":
            theta[k].item(),

        "p":
            p[k].item(),

        "amplitude":
            amplitude[k].item()

    })


df = pd.DataFrame(
    rows
)


df.to_csv(
    "nio_image_equations.csv",
    index=False
)


print(
    "\nSaved: nio_image_equations.csv"
)


# ============================================================
# Quantitative metrics
# ============================================================

with torch.no_grad():

    mse = torch.mean(
        (target - reconstruction)**2
    )


    mae = torch.mean(
        torch.abs(
            target - reconstruction
        )
    )


    signal = torch.sum(
        target**2
    )


    error = torch.sum(
        (target - reconstruction)**2
    )


    relative_error = torch.sqrt(
        error / signal
    )


print("\n")
print("==============================================")
print("FINAL RECONSTRUCTION METRICS")
print("==============================================")


print(
    "MSE =",
    mse.item()
)

print(
    "MAE =",
    mae.item()
)

print(
    "Relative L2 error =",
    relative_error.item()
)


# ============================================================
# ACTIVE PRIMITIVES
#
# Count mathematical primitives carrying meaningful intensity.
# ============================================================

active = torch.sum(
    amplitude > 0.10
).item()


print(
    "Active mathematical primitives =",
    active,
    "out of",
    K
)


```



## Neural Input Optimization of the Riemann Zeta Landscape


```



import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


# ============================================================
# NIO EXPERIMENT 5
#
# NEURAL INPUT OPTIMIZATION OF THE
# RIEMANN ZETA LANDSCAPE
#
#
# Complex input:
#
#       s = sigma + i t
#
#
# Non-trivial zeros occur in:
#
#       0 < sigma < 1
#
#
# Riemann Hypothesis:
#
#       sigma = 1/2
#
# for every non-trivial zero.
#
#
# IMPORTANT:
#
# We DO NOT constrain sigma = 0.5.
#
# NIO is allowed to search the entire critical strip.
#
#
# Objective:
#
#       minimize |zeta(s)|^2
#
#
# Question:
#
# Where do points initialized throughout the critical
# strip move under gradient-based input optimization?
#
#
# This experiment DOES NOT attempt to prove the
# Riemann Hypothesis.
#
# It investigates the optimization geometry of a known
# mathematical landscape and tests whether NIO recovers
# known zero structure.
#
# ============================================================


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

torch.manual_seed(7)
np.random.seed(7)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device =", device)


# ------------------------------------------------------------
# Use double precision.
#
# This matters much more here than in ordinary ML.
# ------------------------------------------------------------

dtype = torch.float64


# ============================================================
# SETTINGS
# ============================================================

N = 500

iters = 2500

lr = 0.01


# ------------------------------------------------------------
# Search region
#
# First several non-trivial zeros have imaginary parts:
#
# 14.1347
# 21.0220
# 25.0109
# 30.4249
# 32.9351
# 37.5862
#
# Search through t = 40.
# ------------------------------------------------------------

sigma_min = 0.05
sigma_max = 0.95

t_min = 10.0
t_max = 40.0


# ------------------------------------------------------------
# Number of terms in eta approximation
# ------------------------------------------------------------

M = 1200


print("NIO points =", N)
print("eta terms =", M)
print("search sigma =", sigma_min, "to", sigma_max)
print("search t =", t_min, "to", t_max)


# ============================================================
# PRECOMPUTE n AND log(n)
# ============================================================

n = torch.arange(
    1,
    M + 1,
    dtype=dtype,
    device=device
)


log_n = torch.log(n)


# alternating signs:
#
# +1, -1, +1, -1, ...

signs = torch.where(

    (
        torch.arange(
            1,
            M + 1,
            device=device
        )
        % 2
    )
    == 1,

    torch.tensor(
        1.0,
        dtype=dtype,
        device=device
    ),

    torch.tensor(
        -1.0,
        dtype=dtype,
        device=device
    )

)


# ============================================================
# DIFFERENTIABLE RIEMANN ZETA APPROXIMATION
#
# Dirichlet eta function:
#
# eta(s) =
#
# sum (-1)^(n-1) / n^s
#
#
# and:
#
# zeta(s) =
#
# eta(s)
# ----------------
# 1 - 2^(1-s)
#
#
# This representation converges for Re(s) > 0,
# which includes our critical strip.
#
# ============================================================

def zeta_eta(sigma, t):

    # --------------------------------------------------------
    # sigma and t have shape:
    #
    #       [N]
    #
    # We construct:
    #
    #       [N,M]
    #
    # --------------------------------------------------------

    sigma_col = sigma[:, None]

    t_col = t[:, None]


    # --------------------------------------------------------
    # n^(-s)
    #
    # s = sigma + i t
    #
    #
    # n^(-s)
    #
    # = exp(-s log n)
    #
    # = n^(-sigma)
    #
    #   [cos(t log n) - i sin(t log n)]
    #
    # --------------------------------------------------------

    magnitude = torch.exp(

        -sigma_col
        *
        log_n[None, :]

    )


    angle = (

        t_col
        *
        log_n[None, :]

    )


    real_terms = (

        signs[None, :]

        *

        magnitude

        *

        torch.cos(angle)

    )


    imag_terms = (

        signs[None, :]

        *

        magnitude

        *

        (-torch.sin(angle))

    )


    eta_real = torch.sum(
        real_terms,
        dim=1
    )


    eta_imag = torch.sum(
        imag_terms,
        dim=1
    )


    # ========================================================
    # denominator:
    #
    #       1 - 2^(1-s)
    #
    #
    # 2^(1-s)
    #
    # =
    #
    # 2^(1-sigma)
    #
    # *
    #
    # exp(-i t ln 2)
    #
    # ========================================================

    ln2 = math.log(2.0)


    amp2 = torch.exp(

        (1.0 - sigma)
        *
        ln2

    )


    angle2 = (

        t
        *
        ln2

    )


    pow2_real = (

        amp2
        *
        torch.cos(angle2)

    )


    pow2_imag = (

        -amp2
        *
        torch.sin(angle2)

    )


    denom_real = (

        1.0
        -
        pow2_real

    )


    denom_imag = (

        -pow2_imag

    )


    # ========================================================
    # Complex division:
    #
    #       eta / denominator
    #
    # ========================================================

    denom_mag2 = (

        denom_real**2

        +

        denom_imag**2

        +

        1e-15

    )


    zeta_real = (

        eta_real * denom_real

        +

        eta_imag * denom_imag

    ) / denom_mag2


    zeta_imag = (

        eta_imag * denom_real

        -

        eta_real * denom_imag

    ) / denom_mag2


    return (
        zeta_real,
        zeta_imag
    )


# ============================================================
# NIO PARAMETERIZATION
#
# z[:,0] -> sigma
#
# z[:,1] -> t
#
#
# sigmoid keeps NIO inside the chosen region.
# ============================================================

def decode(z):

    sigma = (

        sigma_min

        +

        (sigma_max - sigma_min)

        *

        torch.sigmoid(
            z[:,0]
        )

    )


    t = (

        t_min

        +

        (t_max - t_min)

        *

        torch.sigmoid(
            z[:,1]
        )

    )


    return (
        sigma,
        t
    )


# ============================================================
# INITIALIZE POINTS THROUGHOUT CRITICAL STRIP
#
# Instead of ordinary random z, initialize approximately
# uniformly in physical sigma/t coordinates.
# ============================================================

sigma_initial = (

    sigma_min

    +

    (
        sigma_max
        -
        sigma_min
    )

    *

    torch.rand(
        N,
        dtype=dtype,
        device=device
    )

)


t_initial = (

    t_min

    +

    (
        t_max
        -
        t_min
    )

    *

    torch.rand(
        N,
        dtype=dtype,
        device=device
    )

)


# ------------------------------------------------------------
# Inverse sigmoid
#
# Convert physical initialization into latent z.
# ------------------------------------------------------------

sigma_fraction = (

    (
        sigma_initial
        -
        sigma_min
    )

    /

    (
        sigma_max
        -
        sigma_min
    )

)


t_fraction = (

    (
        t_initial
        -
        t_min
    )

    /

    (
        t_max
        -
        t_min
    )

)


def logit(p):

    p = torch.clamp(
        p,
        1e-6,
        1.0 - 1e-6
    )

    return torch.log(
        p / (1.0 - p)
    )


z = torch.stack(

    [

        logit(
            sigma_fraction
        ),

        logit(
            t_fraction
        )

    ],

    dim=1

)


z = (

    z
    .detach()
    .clone()
    .requires_grad_(True)

)


# ============================================================
# SAVE INITIAL LOCATIONS
# ============================================================

with torch.no_grad():

    sigma_start, t_start = decode(z)

    sigma_start = (
        sigma_start
        .cpu()
        .numpy()
    )

    t_start = (
        t_start
        .cpu()
        .numpy()
    )


# ============================================================
# TRACK SOME NIO TRAJECTORIES
#
# Save 30 points so we can visualize their movement.
# ============================================================

track_N = 30


track_sigma = []
track_t = []


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    [z],
    lr=lr
)


loss_history = []


print("\n")
print("==============================================")
print("NIO SEARCHING RIEMANN ZETA LANDSCAPE")
print("==============================================")


# ============================================================
# NIO OPTIMIZATION
# ============================================================

for i in range(iters):


    sigma, t = decode(z)


    zeta_real, zeta_imag = zeta_eta(
        sigma,
        t
    )


    # --------------------------------------------------------
    # |zeta(s)|^2
    # --------------------------------------------------------

    magnitude_squared = (

        zeta_real**2

        +

        zeta_imag**2

    )


    loss = torch.mean(
        magnitude_squared
    )


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    loss_history.append(
        loss.item()
    )


    # --------------------------------------------------------
    # Save trajectories occasionally
    # --------------------------------------------------------

    if i % 10 == 0:

        with torch.no_grad():

            sigma_track, t_track = decode(z)


            track_sigma.append(

                sigma_track[
                    :track_N
                ]
                .cpu()
                .numpy()

            )


            track_t.append(

                t_track[
                    :track_N
                ]
                .cpu()
                .numpy()

            )


    if i % 100 == 0:

        with torch.no_grad():

            mean_sigma = (
                sigma.mean().item()
            )

            mean_distance = (

                torch.abs(
                    sigma
                    -
                    0.5
                )

                .mean()
                .item()

            )


            minimum = (

                magnitude_squared
                .min()
                .item()

            )


        print(
            "iter",
            i,
            "loss",
            loss.item(),
            "mean sigma",
            mean_sigma,
            "mean |sigma-.5|",
            mean_distance,
            "best |zeta|^2",
            minimum
        )


# ============================================================
# FINAL NIO LOCATIONS
# ============================================================

with torch.no_grad():

    sigma_final, t_final = decode(z)


    zr_final, zi_final = zeta_eta(

        sigma_final,
        t_final

    )


    magnitude_final = torch.sqrt(

        zr_final**2

        +

        zi_final**2

    )


# ============================================================
# NUMPY COPIES
# ============================================================

sigma_final_np = (

    sigma_final
    .cpu()
    .numpy()
)


t_final_np = (

    t_final
    .cpu()
    .numpy()
)


magnitude_final_np = (

    magnitude_final
    .cpu()
    .numpy()
)


track_sigma = np.array(
    track_sigma
)


track_t = np.array(
    track_t
)


# ============================================================
# KNOWN FIRST NON-TRIVIAL ZEROS
#
# Positive imaginary components.
#
# All listed zeros have real component 0.5.
#
# These are used ONLY for validation after optimization.
#
# They are NOT used by NIO.
# ============================================================

known_zeros = np.array([

    14.134725141734693,

    21.022039638771555,

    25.010857580145688,

    30.424876125859513,

    32.935061587739189,

    37.586178158825671

])


# ============================================================
# ASSIGN EACH NIO RESULT TO NEAREST KNOWN ZERO
# ============================================================

nearest_zero_index = []

distance_to_zero = []


for j in range(N):


    distances = np.sqrt(

        (
            sigma_final_np[j]
            -
            0.5
        )**2

        +

        (
            t_final_np[j]
            -
            known_zeros
        )**2

    )


    index = np.argmin(
        distances
    )


    nearest_zero_index.append(
        index
    )


    distance_to_zero.append(
        distances[index]
    )


nearest_zero_index = np.array(
    nearest_zero_index
)


distance_to_zero = np.array(
    distance_to_zero
)


# ============================================================
# FINAL NUMERICAL SUMMARY
# ============================================================

print("\n")
print("==============================================")
print("FINAL NIO RESULTS")
print("==============================================")


print(
    "Mean sigma =",
    np.mean(
        sigma_final_np
    )
)


print(
    "Mean |sigma - 0.5| =",
    np.mean(
        np.abs(
            sigma_final_np
            -
            0.5
        )
    )
)


print(
    "Median |sigma - 0.5| =",
    np.median(
        np.abs(
            sigma_final_np
            -
            0.5
        )
    )
)


print(
    "Mean |zeta(s)| =",
    np.mean(
        magnitude_final_np
    )
)


print(
    "Median |zeta(s)| =",
    np.median(
        magnitude_final_np
    )
)


print(
    "Mean distance to nearest known zero =",
    np.mean(
        distance_to_zero
    )
)


print(
    "Median distance to nearest known zero =",
    np.median(
        distance_to_zero
    )
)


# ============================================================
# SUCCESS COUNTS
#
# Use several tolerances.
# ============================================================

for tolerance in [

    0.10,
    0.05,
    0.02,
    0.01

]:

    count = np.sum(

        distance_to_zero
        <
        tolerance

    )


    print(
        "Within",
        tolerance,
        "of known zero:",
        count,
        "/",
        N
    )


# ============================================================
# COUNT BASIN MEMBERSHIP
# ============================================================

print("\n")
print("==============================================")
print("BASIN MEMBERSHIP")
print("==============================================")


for k in range(
    len(
        known_zeros
    )
):

    count = np.sum(

        nearest_zero_index
        ==
        k

    )


    print(
        "zero",
        k + 1,
        "t =",
        known_zeros[k],
        "points =",
        count
    )


# ============================================================
# PLOT LOSS
# ============================================================

plt.figure(
    figsize=(9,5)
)


plt.semilogy(
    loss_history
)


plt.xlabel(
    "NIO Iteration"
)

plt.ylabel(
    "Mean |zeta(s)|^2"
)

plt.title(
    "NIO Optimization of the Riemann Zeta Landscape"
)

plt.grid(True)

plt.show()


# ============================================================
# PLOT INITIAL VS FINAL LOCATIONS
# ============================================================

plt.figure(
    figsize=(8,10)
)


plt.scatter(

    sigma_start,

    t_start,

    s=15,

    alpha=0.35,

    label="Initial NIO Points"

)


plt.scatter(

    sigma_final_np,

    t_final_np,

    s=18,

    label="Optimized NIO Points"

)


# critical line

plt.axvline(

    0.5,

    linestyle="--",

    label="Critical Line sigma = 0.5"

)


# known zeros

plt.scatter(

    np.ones_like(
        known_zeros
    )
    *
    0.5,

    known_zeros,

    marker="x",

    s=100,

    label="Known Zeros"

)


plt.xlim(
    0,
    1
)


plt.ylim(
    t_min,
    t_max
)


plt.xlabel(
    "Real Component sigma"
)

plt.ylabel(
    "Imaginary Component t"
)

plt.title(
    "NIO Search in the Critical Strip"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# PLOT NIO TRAJECTORIES
#
# These are optimization trajectories in the complex plane.
# ============================================================

plt.figure(
    figsize=(8,10)
)


for j in range(
    track_N
):

    plt.plot(

        track_sigma[:,j],

        track_t[:,j],

        alpha=0.7

    )


    # starting location

    plt.scatter(

        track_sigma[0,j],

        track_t[0,j],

        s=20

    )


    # ending location

    plt.scatter(

        track_sigma[-1,j],

        track_t[-1,j],

        marker="x",

        s=40

    )


plt.axvline(

    0.5,

    linestyle="--"

)


plt.scatter(

    np.ones_like(
        known_zeros
    )
    *
    0.5,

    known_zeros,

    marker="*",

    s=120

)


plt.xlim(
    0,
    1
)


plt.ylim(
    t_min,
    t_max
)


plt.xlabel(
    "sigma"
)

plt.ylabel(
    "t"
)

plt.title(
    "NIO Optimization Trajectories"
)

plt.grid(True)

plt.show()


# ============================================================
# BASIN OF ATTRACTION MAP
#
# INITIAL locations are colored according to which known zero
# their FINAL optimized state approaches.
#
# This may be the most interesting visualization.
# ============================================================

plt.figure(
    figsize=(8,10)
)


scatter = plt.scatter(

    sigma_start,

    t_start,

    c=nearest_zero_index,

    s=22

)


plt.axvline(

    0.5,

    linestyle="--"

)


plt.scatter(

    np.ones_like(
        known_zeros
    )
    *
    0.5,

    known_zeros,

    marker="*",

    s=130

)


plt.xlim(
    0,
    1
)


plt.ylim(
    t_min,
    t_max
)


plt.xlabel(
    "Initial sigma"
)

plt.ylabel(
    "Initial t"
)

plt.title(
    "NIO Basins of Attraction for Riemann Zeta Zeros"
)


plt.colorbar(
    scatter,
    label="Nearest Final Zero Index"
)


plt.grid(True)

plt.show()


# ============================================================
# SIGMA DISTRIBUTION
#
# Where did NIO put the real component?
# ============================================================

plt.figure(
    figsize=(8,5)
)


plt.hist(

    sigma_final_np,

    bins=40

)


plt.axvline(

    0.5,

    linestyle="--"

)


plt.xlabel(
    "Optimized sigma"
)

plt.ylabel(
    "Count"
)

plt.title(
    "Distribution of NIO-Optimized Real Components"
)

plt.grid(True)

plt.show()


# ============================================================
# DISTANCE TO CRITICAL LINE VS HEIGHT
# ============================================================

plt.figure(
    figsize=(8,5)
)


plt.scatter(

    t_final_np,

    np.abs(
        sigma_final_np
        -
        0.5
    ),

    s=20

)


plt.xlabel(
    "Imaginary Component t"
)

plt.ylabel(
    "|sigma - 0.5|"
)

plt.title(
    "NIO Distance From Critical Line"
)

plt.grid(True)

plt.show()


# ============================================================
# SAVE RESULTS
# ============================================================

df = pd.DataFrame({

    "sigma_initial":
        sigma_start,

    "t_initial":
        t_start,

    "sigma_final":
        sigma_final_np,

    "t_final":
        t_final_np,

    "abs_zeta":
        magnitude_final_np,

    "distance_from_critical_line":
        np.abs(
            sigma_final_np
            -
            0.5
        ),

    "nearest_zero_number":
        nearest_zero_index
        +
        1,

    "nearest_known_zero_t":
        known_zeros[
            nearest_zero_index
        ],

    "distance_to_nearest_known_zero":
        distance_to_zero

})


df.to_csv(

    "nio_riemann_zeta_results.csv",

    index=False

)


print(
    "\nSaved: nio_riemann_zeta_results.csv"
)


# ============================================================
# PRINT BEST NIO RESULTS
# ============================================================

order = np.argsort(
    magnitude_final_np
)


print("\n")
print("==============================================")
print("BEST NIO-DISCOVERED POINTS")
print("==============================================")


for rank in range(20):


    j = order[rank]


    print(

        rank + 1,

        "sigma =",
        sigma_final_np[j],

        "t =",
        t_final_np[j],

        "|zeta| =",
        magnitude_final_np[j],

        "nearest known t =",
        known_zeros[
            nearest_zero_index[j]
        ],

        "distance =",
        distance_to_zero[j]

    )


```



You want numbers close to things such as
$$ 0.5+14.1347i,\qquad 0.5+21.0220i,\qquad 0.5+25.0109i,\ldots $$
even though those known zeros were never used by the optimizer. They're only introduced afterward for validation.


maybe this is a better approximation of the zeta


```



import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mpmath as mp
import math

torch.manual_seed(7)
np.random.seed(7)

# ============================================================
# NIO + RIEMANN ZETA
#
# Optimize:
#
#       s = sigma + i t
#
# over the critical strip:
#
#       0 < sigma < 1
#
# Objective:
#
#       minimize |zeta(s)|^2
#
# Zeta is evaluated using a differentiable
# Euler-Maclaurin analytic continuation.
# ============================================================

dtype = torch.float64
device = "cuda" if torch.cuda.is_available() else "cpu"

print("device =", device)

# ============================================================
# SETTINGS
# ============================================================

Npoints = 500
iters   = 2000
lr      = 0.01

sigma_min = 0.01
sigma_max = 0.99

t_min = 10.0
t_max = 40.0


# ============================================================
# EULER-MACLAURIN SETTINGS
#
# Increasing these increases numerical accuracy.
# ============================================================

Nsum = 50

# Bernoulli numbers B2, B4, ..., B16

bernoulli = [

    1.0 / 6.0,             # B2
   -1.0 / 30.0,            # B4
    1.0 / 42.0,            # B6
   -1.0 / 30.0,            # B8
    5.0 / 66.0,            # B10
   -691.0 / 2730.0,        # B12
    7.0 / 6.0,             # B14
   -3617.0 / 510.0         # B16
]


# ============================================================
# DIFFERENTIABLE COMPLEX POWER
#
# For positive real x:
#
#       x^(-s)
#
#       = exp(-sigma log x)
#         [cos(t log x) - i sin(t log x)]
# ============================================================

def power_minus_s(x, sigma, t):

    logx = torch.log(x)

    magnitude = torch.exp(
        -sigma * logx
    )

    angle = t * logx

    real = magnitude * torch.cos(angle)
    imag = -magnitude * torch.sin(angle)

    return real, imag


# ============================================================
# COMPLEX MULTIPLICATION
# ============================================================

def cmul(ar, ai, br, bi):

    real = ar * br - ai * bi
    imag = ar * bi + ai * br

    return real, imag


# ============================================================
# COMPLEX DIVISION
# ============================================================

def cdiv(ar, ai, br, bi):

    denom = br**2 + bi**2 + 1e-30

    real = (ar*br + ai*bi) / denom
    imag = (ai*br - ar*bi) / denom

    return real, imag


# ============================================================
# DIFFERENTIABLE RIEMANN ZETA
#
# Euler-Maclaurin:
#
# zeta(s)
#
# = sum_{n=1}^{N-1} n^(-s)
#
#   + N^(1-s)/(s-1)
#
#   + 1/2 N^(-s)
#
#   + sum corrections using Bernoulli numbers.
#
#
# This is an analytic-continuation representation of
# the Riemann zeta function.
# ============================================================

def zeta_euler_maclaurin(sigma, t):

    batch = sigma.shape[0]

    zr = torch.zeros(
        batch,
        dtype=dtype,
        device=device
    )

    zi = torch.zeros(
        batch,
        dtype=dtype,
        device=device
    )


    # --------------------------------------------------------
    # finite sum
    #
    # sum n^(-s)
    # --------------------------------------------------------

    for n in range(1, Nsum):

        x = torch.tensor(
            float(n),
            dtype=dtype,
            device=device
        )

        r, im = power_minus_s(
            x,
            sigma,
            t
        )

        zr = zr + r
        zi = zi + im


    # ========================================================
    # N^(-s)
    # ========================================================

    Ntorch = torch.tensor(
        float(Nsum),
        dtype=dtype,
        device=device
    )

    Nr, Ni = power_minus_s(
        Ntorch,
        sigma,
        t
    )


    # ========================================================
    # 1/2 N^(-s)
    # ========================================================

    zr = zr + 0.5 * Nr
    zi = zi + 0.5 * Ni


    # ========================================================
    # N^(1-s)/(s-1)
    #
    # N^(1-s) = N * N^(-s)
    # ========================================================

    numerator_r = Nsum * Nr
    numerator_i = Nsum * Ni

    denominator_r = sigma - 1.0
    denominator_i = t

    tr, ti = cdiv(

        numerator_r,
        numerator_i,

        denominator_r,
        denominator_i

    )

    zr = zr + tr
    zi = zi + ti


    # ========================================================
    # BERNOULLI CORRECTIONS
    #
    # B_(2k)/(2k)!
    #
    # *
    #
    # s(s+1)...(s+2k-2)
    #
    # *
    #
    # N^(-s-2k+1)
    # ========================================================

    for k, B in enumerate(
        bernoulli,
        start=1
    ):

        order = 2*k

        # ----------------------------------------------------
        # rising factorial
        #
        # s(s+1)...(s+2k-2)
        # ----------------------------------------------------

        pr = torch.ones(
            batch,
            dtype=dtype,
            device=device
        )

        pi = torch.zeros(
            batch,
            dtype=dtype,
            device=device
        )

        for j in range(
            order - 1
        ):

            br = sigma + float(j)
            bi = t

            pr, pi = cmul(
                pr,
                pi,
                br,
                bi
            )


        # ----------------------------------------------------
        # N^(-s-(2k-1))
        #
        # = N^(-s) * N^(-(2k-1))
        # ----------------------------------------------------

        scale = (
            float(Nsum)
            **
            (-(order - 1))
        )

        pow_r = Nr * scale
        pow_i = Ni * scale


        cr, ci = cmul(

            pr,
            pi,

            pow_r,
            pow_i

        )


        coefficient = (
            B
            /
            math.factorial(order)
        )


        zr = zr + coefficient * cr
        zi = zi + coefficient * ci


    return zr, zi


# ============================================================
# NIO PARAMETERIZATION
#
# unconstrained z
#
#       ↓
#
# sigmoid
#
#       ↓
#
# critical strip
# ============================================================

def decode(z):

    sigma = (

        sigma_min

        +

        (sigma_max - sigma_min)

        *

        torch.sigmoid(z[:,0])

    )

    t = (

        t_min

        +

        (t_max - t_min)

        *

        torch.sigmoid(z[:,1])

    )

    return sigma, t


# ============================================================
# INITIALIZE UNIFORMLY THROUGHOUT CRITICAL STRIP
# ============================================================

sigma0 = (

    sigma_min

    +

    (sigma_max - sigma_min)

    *

    torch.rand(
        Npoints,
        dtype=dtype,
        device=device
    )

)

t0 = (

    t_min

    +

    (t_max - t_min)

    *

    torch.rand(
        Npoints,
        dtype=dtype,
        device=device
    )

)


def logit(p):

    p = torch.clamp(
        p,
        1e-6,
        1-1e-6
    )

    return torch.log(
        p/(1-p)
    )


ps = (

    (sigma0 - sigma_min)

    /

    (sigma_max - sigma_min)

)

pt = (

    (t0 - t_min)

    /

    (t_max - t_min)

)


z = torch.stack(

    [
        logit(ps),
        logit(pt)
    ],

    dim=1

)

z = z.detach().clone().requires_grad_(True)


sigma_start = sigma0.detach().cpu().numpy()
t_start = t0.detach().cpu().numpy()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    [z],
    lr=lr
)

loss_history = []

track_N = 30

track_sigma = []
track_t = []


# ============================================================
# NIO
# ============================================================

print()
print("========================================")
print("NIO SEARCH OF RIEMANN ZETA")
print("========================================")


for iteration in range(iters):

    sigma, t = decode(z)

    zr, zi = zeta_euler_maclaurin(
        sigma,
        t
    )

    abs_zeta_squared = (

        zr**2
        +
        zi**2

    )

    loss = abs_zeta_squared.mean()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    loss_history.append(
        loss.item()
    )


    if iteration % 10 == 0:

        with torch.no_grad():

            ss, tt = decode(z)

            track_sigma.append(
                ss[:track_N].cpu().numpy()
            )

            track_t.append(
                tt[:track_N].cpu().numpy()
            )


    if iteration % 100 == 0:

        distance = torch.abs(
            sigma - 0.5
        ).mean()

        print(
            iteration,
            "loss =",
            loss.item(),
            "mean |sigma-.5| =",
            distance.item()
        )


# ============================================================
# FINAL RESULTS
# ============================================================

with torch.no_grad():

    sigma_final, t_final = decode(z)

    zr, zi = zeta_euler_maclaurin(
        sigma_final,
        t_final
    )

    abs_zeta = torch.sqrt(
        zr**2 + zi**2
    )


sigma_final = sigma_final.cpu().numpy()
t_final = t_final.cpu().numpy()
abs_zeta = abs_zeta.cpu().numpy()

track_sigma = np.array(track_sigma)
track_t = np.array(track_t)


# ============================================================
# KNOWN ZEROS
#
# ONLY USED AFTER NIO FOR COMPARISON
# ============================================================

known_zeros = np.array([

    14.134725141734693,
    21.022039638771555,
    25.010857580145688,
    30.424876125859513,
    32.935061587739189,
    37.586178158825671

])


# ============================================================
# ASSIGN TO NEAREST KNOWN ZERO
# ============================================================

nearest_index = []
distance_known = []


for j in range(Npoints):

    d = np.sqrt(

        (sigma_final[j] - 0.5)**2

        +

        (t_final[j] - known_zeros)**2

    )

    k = np.argmin(d)

    nearest_index.append(k)
    distance_known.append(d[k])


nearest_index = np.array(nearest_index)
distance_known = np.array(distance_known)


# ============================================================
# INDEPENDENT HIGH-PRECISION VALIDATION
#
# mpmath evaluates zeta independently of our
# differentiable PyTorch implementation.
# ============================================================

mp.mp.dps = 50

mp_zeta = []

for sigma, t in zip(
    sigma_final,
    t_final
):

    s = mp.mpc(
        float(sigma),
        float(t)
    )

    value = mp.zeta(s)

    mp_zeta.append(
        float(abs(value))
    )


mp_zeta = np.array(mp_zeta)


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("RESULTS")
print("========================================")

print(
    "Mean |sigma - 0.5| =",
    np.mean(
        np.abs(sigma_final - 0.5)
    )
)

print(
    "Median |sigma - 0.5| =",
    np.median(
        np.abs(sigma_final - 0.5)
    )
)

print(
    "Mean NIO |zeta| =",
    np.mean(abs_zeta)
)

print(
    "Mean independent mpmath |zeta| =",
    np.mean(mp_zeta)
)

print(
    "Median independent mpmath |zeta| =",
    np.median(mp_zeta)
)


for tolerance in [
    0.1,
    0.05,
    0.02,
    0.01
]:

    count = np.sum(
        distance_known < tolerance
    )

    print(
        "Within",
        tolerance,
        "of known zero:",
        count,
        "/",
        Npoints
    )


# ============================================================
# LOSS
# ============================================================

plt.figure(figsize=(8,5))

plt.semilogy(
    loss_history
)

plt.xlabel("NIO iteration")
plt.ylabel("|zeta(s)|^2")
plt.title("NIO Optimization of Riemann Zeta")

plt.grid(True)

plt.show()


# ============================================================
# INITIAL -> FINAL
# ============================================================

plt.figure(figsize=(8,10))

plt.scatter(
    sigma_start,
    t_start,
    s=15,
    alpha=.3,
    label="Initial"
)

plt.scatter(
    sigma_final,
    t_final,
    s=20,
    label="NIO"
)

plt.axvline(
    .5,
    linestyle="--",
    label="sigma = 0.5"
)

plt.scatter(
    np.ones_like(known_zeros)*.5,
    known_zeros,
    marker="*",
    s=150,
    label="Known zeros"
)

plt.xlabel("sigma")
plt.ylabel("t")

plt.xlim(0,1)
plt.ylim(t_min,t_max)

plt.title(
    "NIO Search of the Critical Strip"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# OPTIMIZATION TRAJECTORIES
# ============================================================

plt.figure(figsize=(8,10))

for j in range(track_N):

    plt.plot(
        track_sigma[:,j],
        track_t[:,j],
        alpha=.7
    )

    plt.scatter(
        track_sigma[0,j],
        track_t[0,j],
        s=20
    )

    plt.scatter(
        track_sigma[-1,j],
        track_t[-1,j],
        marker="x",
        s=40
    )


plt.axvline(
    .5,
    linestyle="--"
)

plt.scatter(
    np.ones_like(known_zeros)*.5,
    known_zeros,
    marker="*",
    s=150
)

plt.xlabel("sigma")
plt.ylabel("t")

plt.xlim(0,1)
plt.ylim(t_min,t_max)

plt.title(
    "NIO Trajectories Through the Critical Strip"
)

plt.grid(True)

plt.show()


# ============================================================
# BASINS
# ============================================================

plt.figure(figsize=(8,10))

sc = plt.scatter(

    sigma_start,
    t_start,

    c=nearest_index,

    s=25

)

plt.axvline(
    .5,
    linestyle="--"
)

plt.scatter(

    np.ones_like(known_zeros)*.5,
    known_zeros,

    marker="*",
    s=150

)

plt.xlabel("Initial sigma")
plt.ylabel("Initial t")

plt.xlim(0,1)
plt.ylim(t_min,t_max)

plt.title(
    "NIO Basins of Attraction"
)

plt.colorbar(
    sc,
    label="Final Zero"
)

plt.grid(True)

plt.show()


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame({

    "sigma_initial":
        sigma_start,

    "t_initial":
        t_start,

    "sigma_final":
        sigma_final,

    "t_final":
        t_final,

    "NIO_abs_zeta":
        abs_zeta,

    "mpmath_abs_zeta":
        mp_zeta,

    "distance_sigma_half":
        np.abs(
            sigma_final-.5
        ),

    "nearest_known_zero":
        known_zeros[
            nearest_index
        ],

    "distance_known_zero":
        distance_known

})


df.to_csv(
    "nio_riemann_zeta.csv",
    index=False
)


# ============================================================
# BEST DISCOVERIES
# ============================================================

order = np.argsort(
    mp_zeta
)


print()
print("========================================")
print("BEST NIO DISCOVERIES")
print("========================================")


for rank in range(20):

    j = order[rank]

    print(
        rank+1,
        "s =",
        sigma_final[j],
        "+",
        t_final[j],
        "i",
        "NIO |zeta| =",
        abs_zeta[j],
        "mpmath |zeta| =",
        mp_zeta[j]
    )


```


try some proving



```


print("\n==============================================")
print("SCIENTIFIC INTERPRETATION")
print("==============================================")

j = np.argmin(mp_zeta)

sigma_best = sigma_final[j]
t_best     = t_final[j]
zeta_best  = mp_zeta[j]

distance = abs(sigma_best - 0.5)

print("Best candidate:")
print("sigma =", sigma_best)
print("t     =", t_best)
print("|zeta| =", zeta_best)
print("|sigma - 0.5| =", distance)

if zeta_best < 1e-20 and distance > 1e-6:

    print("\n*** POTENTIAL OFF-CRITICAL-LINE ZERO ***")
    print("NIO has found a numerical candidate away from sigma = 0.5.")
    print("IF rigorous independent mathematics confirms zeta(s) = 0,")
    print("this would constitute a counterexample to the Riemann Hypothesis.")
    print("DO NOT interpret this numerical result itself as a disproof.")
    print("The candidate requires arbitrary-precision and rigorous verification.")

elif zeta_best < 1e-20 and distance <= 1e-6:

    print("\nNIO found a candidate zero on the critical line.")
    print("This is consistent with the Riemann Hypothesis,")
    print("but does not provide a proof.")

else:

    print("\nNo convincing zero candidate was discovered.")
    print("This result neither proves nor disproves the Riemann Hypothesis.")



```





try this too


```



import torch
import numpy as np
import mpmath as mp
import math

torch.manual_seed(7)
np.random.seed(7)

dtype  = torch.float64
device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# NIO SEARCH FOR AN OFF-CRITICAL-LINE ZETA ZERO
#
# Goal:
#   make |zeta(s)| small
#   while pushing Re(s) AWAY from 0.5
#
# IMPORTANT:
#   A numerical candidate is NOT a disproof of RH.
# ============================================================

NPOINTS = 500
ITERS   = 3000
LR      = 0.005

SIGMA_MIN = 0.01
SIGMA_MAX = 0.99

# Start with a region where we know the numerics can be tested.
# Later increase these windows.
T_MIN = 10.0
T_MAX = 100.0

# pressure away from critical line
LAMBDA_OFF = 0.05

# Euler-Maclaurin
NSUM = 100

BERNOULLI = [
     1/6,
    -1/30,
     1/42,
    -1/30,
     5/66,
    -691/2730,
     7/6,
    -3617/510,
     43867/798,
    -174611/330
]


# ============================================================
# COMPLEX ARITHMETIC
# ============================================================

def cmul(ar, ai, br, bi):
    return ar*br-ai*bi, ar*bi+ai*br


def cdiv(ar, ai, br, bi):

    d = br*br + bi*bi + 1e-30

    return (
        (ar*br + ai*bi)/d,
        (ai*br - ar*bi)/d
    )


# ============================================================
# DIFFERENTIABLE EULER-MACLAURIN ZETA
# ============================================================

def zeta_em(sigma, t):

    batch = sigma.shape[0]

    zr = torch.zeros(batch, dtype=dtype, device=device)
    zi = torch.zeros(batch, dtype=dtype, device=device)

    # --------------------------------------------------------
    # finite sum
    # --------------------------------------------------------

    n = torch.arange(
        1,
        NSUM,
        dtype=dtype,
        device=device
    )

    logn = torch.log(n)

    mag = torch.exp(
        -sigma[:,None] * logn[None,:]
    )

    ang = (
        t[:,None] * logn[None,:]
    )

    zr += torch.sum(
        mag * torch.cos(ang),
        dim=1
    )

    zi += torch.sum(
        -mag * torch.sin(ang),
        dim=1
    )

    # --------------------------------------------------------
    # N^(-s)
    # --------------------------------------------------------

    logN = math.log(NSUM)

    magN = torch.exp(
        -sigma * logN
    )

    angN = t * logN

    Nr = magN * torch.cos(angN)
    Ni = -magN * torch.sin(angN)

    # 1/2 N^-s

    zr += 0.5 * Nr
    zi += 0.5 * Ni

    # --------------------------------------------------------
    # N^(1-s)/(s-1)
    # --------------------------------------------------------

    tr, ti = cdiv(
        NSUM*Nr,
        NSUM*Ni,
        sigma-1.0,
        t
    )

    zr += tr
    zi += ti

    # --------------------------------------------------------
    # Bernoulli corrections
    # --------------------------------------------------------

    for k, B in enumerate(BERNOULLI, start=1):

        order = 2*k

        pr = torch.ones(
            batch,
            dtype=dtype,
            device=device
        )

        pi = torch.zeros(
            batch,
            dtype=dtype,
            device=device
        )

        # rising factorial
        # s(s+1)...(s+2k-2)

        for j in range(order-1):

            pr, pi = cmul(
                pr, pi,
                sigma + j,
                t
            )

        scale = NSUM ** (-(order-1))

        cr, ci = cmul(
            pr, pi,
            Nr*scale,
            Ni*scale
        )

        coefficient = (
            B / math.factorial(order)
        )

        zr += coefficient * cr
        zi += coefficient * ci

    return zr, zi


# ============================================================
# NIO PARAMETERIZATION
# ============================================================

def decode(z):

    sigma = (
        SIGMA_MIN
        +
        (SIGMA_MAX-SIGMA_MIN)
        *
        torch.sigmoid(z[:,0])
    )

    t = (
        T_MIN
        +
        (T_MAX-T_MIN)
        *
        torch.sigmoid(z[:,1])
    )

    return sigma, t


# ============================================================
# INITIALIZATION
# ============================================================

z = torch.randn(
    NPOINTS,
    2,
    dtype=dtype,
    device=device,
    requires_grad=True
)

optimizer = torch.optim.Adam(
    [z],
    lr=LR
)


# ============================================================
# NIO
# ============================================================

print("\n==============================================")
print("NIO OFF-CRITICAL-LINE SEARCH")
print("==============================================")

for iteration in range(ITERS):

    sigma, t = decode(z)

    zr, zi = zeta_em(
        sigma,
        t
    )

    zeta2 = zr*zr + zi*zi

    distance2 = (
        sigma - 0.5
    )**2

    # --------------------------------------------------------
    # Find zero, BUT reward distance from 0.5
    # --------------------------------------------------------

    loss_each = (
        zeta2
        -
        LAMBDA_OFF * distance2
    )

    loss = loss_each.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iteration % 200 == 0:

        print(
            iteration,
            "loss =",
            loss.item(),
            "mean |zeta| =",
            torch.sqrt(zeta2).mean().item(),
            "mean distance from .5 =",
            torch.abs(sigma-.5).mean().item()
        )


# ============================================================
# COLLECT RESULTS
# ============================================================

with torch.no_grad():

    sigma, t = decode(z)

    zr, zi = zeta_em(
        sigma,
        t
    )

    abs_zeta = torch.sqrt(
        zr*zr + zi*zi
    )


sigma = sigma.cpu().numpy()
t = t.cpu().numpy()
abs_zeta = abs_zeta.cpu().numpy()


# ============================================================
# FIND INTERESTING CANDIDATES
#
# Prefer:
#   small |zeta|
#   AND far from 0.5
# ============================================================

distance = np.abs(
    sigma - 0.5
)

score = (
    abs_zeta
    /
    (distance + 1e-12)
)

order = np.argsort(score)


# ============================================================
# INDEPENDENT HIGH-PRECISION VERIFICATION
# ============================================================

print("\n==============================================")
print("INDEPENDENT VERIFICATION")
print("==============================================")

best_candidate = None

for rank in range(20):

    j = order[rank]

    # Ignore points essentially on critical line

    if distance[j] < 1e-5:
        continue

    print("\nCandidate", rank+1)

    print("sigma =", sigma[j])
    print("t     =", t[j])
    print("NIO |zeta| =", abs_zeta[j])
    print("|sigma-.5| =", distance[j])

    values = []

    # --------------------------------------------------------
    # Completely independent mpmath evaluation
    # --------------------------------------------------------

    for digits in [50, 100, 200]:

        mp.mp.dps = digits

        s = mp.mpc(
            str(sigma[j]),
            str(t[j])
        )

        value = abs(
            mp.zeta(s)
        )

        values.append(value)

        print(
            digits,
            "digit |zeta| =",
            mp.nstr(value, 25)
        )

    # --------------------------------------------------------
    # candidate condition
    #
    # This ONLY means "interesting enough to investigate."
    # --------------------------------------------------------

    if values[-1] < mp.mpf("1e-20"):

        best_candidate = (
            sigma[j],
            t[j],
            values[-1]
        )

        break


# ============================================================
# SCIENTIFIC INTERPRETATION
# ============================================================

print("\n\n==============================================")
print("SCIENTIFIC INTERPRETATION")
print("==============================================")


if best_candidate is None:

    print("""
NO COUNTEREXAMPLE FOUND.

NIO did not identify a credible off-critical-line zero
in the region searched.

This result neither proves nor disproves the
Riemann Hypothesis.
""")


else:

    sb, tb, zb = best_candidate

    print("\n*** POTENTIAL OFF-CRITICAL-LINE CANDIDATE ***")

    print("\nsigma =", sb)
    print("t     =", tb)
    print("|zeta| =", mp.nstr(zb, 30))

    print("""
NIO has identified a numerical point away from
Re(s) = 0.5 where the independently evaluated
Riemann zeta function is extremely small.

THIS IS NOT YET A COUNTEREXAMPLE TO THE
RIEMANN HYPOTHESIS.

If rigorous independent mathematics establishes
that zeta(s) is exactly zero at this off-critical-line
location, then this would constitute a counterexample
and the Riemann Hypothesis would be false.

The candidate must therefore be independently and
rigorously verified before any mathematical claim
is made.
""")


# ============================================================
# SHOW BEST NUMERICAL POINT REGARDLESS
# ============================================================

j = order[0]

print("==============================================")
print("BEST NIO POINT")
print("==============================================")

print("sigma =", sigma[j])
print("t     =", t[j])
print("NIO |zeta| =", abs_zeta[j])
print("|sigma-.5| =", distance[j])


```









