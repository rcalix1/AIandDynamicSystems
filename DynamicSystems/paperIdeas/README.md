## Paper Ideas

* link

## NIO + cybersecurity/malware dynamics


Expectation is that MAX will drive roughly toward

$$ I_0\uparrow,\qquad \beta\uparrow,\qquad q\downarrow,\qquad u\downarrow $$

and MIN will do approximately the reverse.



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



```


## NIO + geometry / vibrating string


```



```

## Information theory + dynamical systems




```




```








