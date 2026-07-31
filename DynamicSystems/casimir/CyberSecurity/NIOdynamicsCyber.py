#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# In[2]:


torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"


# In[3]:


###############################################################
# Simulation parameters
###############################################################

dt         = 0.01
Time_steps = 500
iters      = 1000


# In[5]:


###############################################################
# Enterprise network size
###############################################################

TOTAL_HOSTS = 1000.0


# In[6]:


###############################################################
# Fixed model parameters
###############################################################

eta = 0.20          # quarantine restoration rate


# In[7]:


###############################################################
# Policy parameter ranges (NIO searches these)
###############################################################

beta_min  = 0.10
beta_max  = 0.80

delta_min = 0.05
delta_max = 0.90

gamma_min = 0.05
gamma_max = 0.90


# In[9]:


###############################################################
# Initial network state
###############################################################

S0 = 990.0
I0 = 10.0
Q0 = 0.0
R0 = 0.0



# In[10]:


###############################################################
# Cyber Dynamical System
###############################################################

class CyberStep(nn.Module):

    def __init__(self, dt=0.01):
        super().__init__()
        self.dt = dt

    def forward(self, state, beta, delta, gamma):

        S = state[:,0]
        I = state[:,1]
        Q = state[:,2]
        R = state[:,3]

        ########################################################
        # Malware propagation equations
        ########################################################

        dS = -beta*S*I/TOTAL_HOSTS

        dI = beta*S*I/TOTAL_HOSTS \
             - delta*I \
             - gamma*I

        dQ = delta*I - eta*Q

        dR = gamma*I + eta*Q

        ########################################################
        # Euler step
        ########################################################

        S = S + self.dt*dS
        I = I + self.dt*dI
        Q = Q + self.dt*dQ
        R = R + self.dt*dR

        ########################################################
        # Keep states non-negative
        ########################################################

        S = torch.clamp(S,min=0)
        I = torch.clamp(I,min=0)
        Q = torch.clamp(Q,min=0)
        R = torch.clamp(R,min=0)

        return torch.stack([S,I,Q,R],dim=1)




# In[11]:


model = CyberStep(dt).to(device)



# In[12]:


###############################################################
# Simulate system
###############################################################

def simulate(beta, delta, gamma):

    state = torch.tensor(
        [[S0,I0,Q0,R0]],
        dtype=torch.float32,
        device=device
    )

    traj = []

    traj.append(state.clone())

    for k in range(Time_steps):

        state = model(state,beta,delta,gamma)

        traj.append(state.clone())

    return torch.stack(traj)


# In[13]:


###############################################################
# NIO variables
###############################################################

z = torch.randn(3,device=device,requires_grad=True)

optimizer = optim.Adam([z],lr=0.05)


# In[15]:


###############################################################
# History
###############################################################

loss_history=[]

beta_history=[]
delta_history=[]
gamma_history = []


# In[16]:


###############################################################
# NIO Optimization
###############################################################

for i in range(iters):

    beta  = beta_min  + (beta_max-beta_min)*torch.sigmoid(z[0])

    delta = delta_min + (delta_max-delta_min)*torch.sigmoid(z[1])

    gamma = gamma_min + (gamma_max-gamma_min)*torch.sigmoid(z[2])

    traj = simulate(beta,delta,gamma)

    ###########################################################
    # Total infected over entire outbreak
    ###########################################################

    infected = traj[:,0,1].sum()

    ###########################################################
    # Simple policy cost
    ###########################################################

    policy_cost = \
        20*delta + \
        20*gamma + \
        15*(1-beta)

    ###########################################################
    # Loss
    ###########################################################

    loss = infected + policy_cost

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    loss_history.append(loss.item())

    beta_history.append(beta.item())
    delta_history.append(delta.item())
    gamma_history.append(gamma.item())

    if i%100==0:

        print(
            i,
            loss.item(),
            beta.item(),
            delta.item(),
            gamma.item()
        )



# In[17]:


###############################################################
# Best trajectory
###############################################################

best_traj = simulate(beta,delta,gamma)




# In[18]:


###############################################################
# Random search baseline
###############################################################

random_scores=[]

Nrandom=5000

for k in range(Nrandom):

    beta = beta_min + (beta_max-beta_min)*torch.rand(1,device=device)

    delta = delta_min + (delta_max-delta_min)*torch.rand(1,device=device)

    gamma = gamma_min + (gamma_max-gamma_min)*torch.rand(1,device=device)

    traj = simulate(beta,delta,gamma)

    infected = traj[:,0,1].sum()

    policy_cost = \
        20*delta + \
        20*gamma + \
        15*(1-beta)

    score = infected + policy_cost

    random_scores.append(score.item())


# In[19]:


###############################################################
# Print results
###############################################################

print()

print("Best policy")

print("beta :",beta.item())
print("delta:",delta.item())
print("gamma:",gamma.item())


# In[20]:


###############################################################
# Plots
###############################################################

plt.figure(figsize=(6,4))

plt.plot(loss_history)

plt.grid()

plt.title("Optimization Loss")

plt.xlabel("Iteration")

plt.ylabel("Loss")

plt.show()


# In[22]:


###############################################################

plt.figure(figsize=(6,4))

plt.plot(best_traj[:,0,0].detach().cpu(),label="Susceptible")

plt.plot(best_traj[:,0,1].detach().cpu(),label="Infected")

plt.plot(best_traj[:,0,2].detach().cpu(),label="Quarantined")

plt.plot(best_traj[:,0,3].detach().cpu(),label="Recovered")

plt.legend()

plt.grid()

plt.title("Best NIO Trajectory")

plt.show()


# In[23]:


###############################################################

plt.figure(figsize=(6,4))

plt.hist(random_scores,bins=50)

plt.axvline(loss_history[-1],
            color='red',
            linewidth=3,
            label="NIO")

plt.legend()

plt.grid()

plt.title("Random Policies vs NIO")

plt.show()


# In[24]:


###############################################################

plt.figure(figsize=(6,4))

plt.plot(beta_history,label="beta")

plt.plot(delta_history,label="delta")

plt.plot(gamma_history,label="gamma")

plt.legend()

plt.grid()

plt.title("Policy Evolution")

plt.show()



# In[ ]:





# 
# ## Version 2 - Minimize case
# 

# In[ ]:





# In[35]:


###############################################################
#
# cyber_nio.py
#
# Neural Input Optimization (NIO)
# for Cybersecurity Policy Discovery
#
###############################################################

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# In[36]:


torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"


# In[37]:


###############################################################
# Simulation
###############################################################

dt = 0.05
Time_steps = 400
iters = 1000


# In[38]:


###############################################################
# Enterprise Network
###############################################################

TOTAL_HOSTS = 1000.0


# In[39]:


###############################################################
# Fixed Parameters
###############################################################

alpha = 0.30        # exposed -> infected
eta   = 0.15        # quarantine -> recovered


# In[40]:


###############################################################
# Policy Bounds
###############################################################

beta_min  = 0.10
beta_max  = 0.90

delta_min = 0.05
delta_max = 0.90

gamma_min = 0.05
gamma_max = 0.90


# In[42]:


###############################################################
# Initial Network
###############################################################

S0 = 985.0
E0 = 5.0
I0 = 10.0
Q0 = 0.0
R0 = 0.0


# In[43]:


###############################################################
# Cyber Dynamical System
###############################################################

class CyberStep(nn.Module):

    def __init__(self,dt=0.05):
        super().__init__()
        self.dt = dt

    def forward(self,state,beta,delta,gamma):

        S = state[:,0]
        E = state[:,1]
        I = state[:,2]
        Q = state[:,3]
        R = state[:,4]

        #######################################################
        # SEIQR Equations
        #######################################################

        dS = -beta*S*I/TOTAL_HOSTS

        dE = beta*S*I/TOTAL_HOSTS \
             - alpha*E

        dI = alpha*E \
             - delta*I \
             - gamma*I

        dQ = delta*I \
             - eta*Q

        dR = gamma*I \
             + eta*Q

        #######################################################
        # Euler
        #######################################################

        S = S + self.dt*dS
        E = E + self.dt*dE
        I = I + self.dt*dI
        Q = Q + self.dt*dQ
        R = R + self.dt*dR

        #######################################################
        # Clamp
        #######################################################

        S = torch.clamp(S,min=0)
        E = torch.clamp(E,min=0)
        I = torch.clamp(I,min=0)
        Q = torch.clamp(Q,min=0)
        R = torch.clamp(R,min=0)

        return torch.stack([S,E,I,Q,R],dim=1)



# In[44]:


model = CyberStep(dt).to(device)




# In[45]:


###############################################################
# Simulation
###############################################################

def simulate(beta, delta, gamma):

    state = torch.tensor(
        [[S0, E0, I0, Q0, R0]],
        dtype=torch.float32,
        device=device
    )

    trajectory = [state.clone()]

    for k in range(Time_steps):

        state = model(state, beta, delta, gamma)

        trajectory.append(state.clone())

    trajectory = torch.stack(trajectory)

    return trajectory


# In[46]:


###############################################################
# NIO variables
###############################################################

z = torch.randn(3, device=device, requires_grad=True)

optimizer = optim.Adam([z], lr=0.05)


# In[47]:


###############################################################
# History
###############################################################

loss_history = []

beta_history = []
delta_history = []
gamma_history = []


# In[48]:


###############################################################
# Optimization
###############################################################

print()
print("Running NIO Optimization")
print()

for iteration in range(iters):

    ###########################################################
    # Policy variables
    ###########################################################

    beta = beta_min + (beta_max-beta_min) * torch.sigmoid(z[0])

    delta = delta_min + (delta_max-delta_min) * torch.sigmoid(z[1])

    gamma = gamma_min + (gamma_max-gamma_min) * torch.sigmoid(z[2])

    ###########################################################
    # Simulate outbreak
    ###########################################################

    traj = simulate(beta, delta, gamma)

    ###########################################################
    # States
    ###########################################################

    S = traj[:,0,0]
    E = traj[:,0,1]
    I = traj[:,0,2]
    Q = traj[:,0,3]
    R = traj[:,0,4]

    ###########################################################
    # Objective
    #
    # Minimize total infected over the outbreak
    ###########################################################

    total_infected = torch.sum(I)

    loss = total_infected

    ###########################################################
    # Gradient step
    ###########################################################

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    ###########################################################
    # Save history
    ###########################################################

    loss_history.append(loss.item())

    beta_history.append(beta.item())
    delta_history.append(delta.item())
    gamma_history.append(gamma.item())

    ###########################################################
    # Progress
    ###########################################################

    if iteration % 100 == 0:

        print(
            "Iter:",
            iteration,
            "Loss:",
            round(loss.item(),3),
            "beta:",
            round(beta.item(),3),
            "delta:",
            round(delta.item(),3),
            "gamma:",
            round(gamma.item(),3)
        )



# In[49]:


###############################################################
# Best policy
###############################################################

best_beta = beta.detach()

best_delta = delta.detach()

best_gamma = gamma.detach()

best_traj = simulate(best_beta, best_delta, best_gamma)

print()
print("Optimization Finished")
print()

print("Best Policy")

print("beta :", best_beta.item())
print("delta:", best_delta.item())
print("gamma:", best_gamma.item())


# In[50]:


###############################################################
# Random Policy Search
###############################################################

print()
print("Running Random Search")
print()

Nrandom = 5000

random_scores = []

random_beta = []
random_delta = []
random_gamma = []

best_random_score = 1e30

best_random_beta = None
best_random_delta = None
best_random_gamma = None

for k in range(Nrandom):

    beta = beta_min + (beta_max-beta_min)*torch.rand(1,device=device)

    delta = delta_min + (delta_max-delta_min)*torch.rand(1,device=device)

    gamma = gamma_min + (gamma_max-gamma_min)*torch.rand(1,device=device)

    traj = simulate(beta,delta,gamma)

    I = traj[:,0,2]

    score = torch.sum(I)

    random_scores.append(score.item())

    random_beta.append(beta.item())
    random_delta.append(delta.item())
    random_gamma.append(gamma.item())

    if score.item() < best_random_score:

        best_random_score = score.item()

        best_random_beta = beta.item()
        best_random_delta = delta.item()
        best_random_gamma = gamma.item()



# In[51]:


###############################################################
# Print comparison
###############################################################

print()
print("---------------------------------------")
print("Random Search")
print("---------------------------------------")
print("Best Score :", best_random_score)
print("beta       :", best_random_beta)
print("delta      :", best_random_delta)
print("gamma      :", best_random_gamma)

print()

print("---------------------------------------")
print("NIO")
print("---------------------------------------")
print("Score :", loss_history[-1])
print("beta  :", best_beta.item())
print("delta :", best_delta.item())
print("gamma :", best_gamma.item())


# In[52]:


###############################################################
# Plot Optimization Loss
###############################################################

plt.figure(figsize=(7,4))

plt.plot(loss_history)

plt.grid(True)

plt.xlabel("Iteration")

plt.ylabel("Loss")

plt.title("NIO Optimization")

plt.tight_layout()

plt.show()


# In[53]:


###############################################################
# Best Trajectory
###############################################################

S = best_traj[:,0,0].cpu().numpy()
E = best_traj[:,0,1].cpu().numpy()
I = best_traj[:,0,2].cpu().numpy()
Q = best_traj[:,0,3].cpu().numpy()
R = best_traj[:,0,4].cpu().numpy()

plt.figure(figsize=(8,5))

plt.plot(S,label="Susceptible")
plt.plot(E,label="Exposed")
plt.plot(I,label="Infected")
plt.plot(Q,label="Quarantined")
plt.plot(R,label="Recovered")

plt.grid(True)

plt.legend()

plt.xlabel("Time")

plt.ylabel("Hosts")

plt.title("SEIQR Dynamics Using NIO Policy")

plt.tight_layout()

plt.show()


# In[54]:


###############################################################
# Histogram
###############################################################

plt.figure(figsize=(7,4))

plt.hist(random_scores,
         bins=50)

plt.axvline(loss_history[-1],
            color='red',
            linewidth=3,
            label="NIO")

plt.legend()

plt.xlabel("Total Infection")

plt.ylabel("Frequency")

plt.title("Random Policies vs NIO")

plt.grid(True)

plt.tight_layout()

plt.show()


# In[55]:


###############################################################
# Policy Evolution
###############################################################

plt.figure(figsize=(8,5))

plt.plot(beta_history,label="beta")

plt.plot(delta_history,label="delta")

plt.plot(gamma_history,label="gamma")

plt.legend()

plt.grid(True)

plt.xlabel("Iteration")

plt.ylabel("Value")

plt.title("Policy Parameters During Optimization")

plt.tight_layout()

plt.show()


# In[56]:


###############################################################
# Scatter Plot of Policy Space
###############################################################

plt.figure(figsize=(7,6))

sc = plt.scatter(random_beta,
                 random_delta,
                 c=random_scores,
                 s=8,
                 cmap="viridis")

plt.scatter(best_beta.cpu().item(),
            best_delta.cpu().item(),
            color="red",
            s=150,
            marker="*",
            label="NIO")

plt.xlabel("Beta (Propagation Rate)")

plt.ylabel("Delta (Detection Rate)")

plt.title("Policy Space")

plt.legend()

plt.colorbar(sc,label="Total Infection")

plt.grid(True)

plt.tight_layout()

plt.show()


# In[57]:


###############################################################
# Delta vs Gamma
###############################################################

plt.figure(figsize=(7,6))

sc = plt.scatter(random_delta,
                 random_gamma,
                 c=random_scores,
                 s=8,
                 cmap="plasma")

plt.scatter(best_delta.cpu().item(),
            best_gamma.cpu().item(),
            color="red",
            marker="*",
            s=150)

plt.xlabel("Detection Rate")

plt.ylabel("Recovery Rate")

plt.title("Detection vs Recovery Policies")

plt.colorbar(sc,label="Total Infection")

plt.grid(True)

plt.tight_layout()

plt.show()


# In[58]:


###############################################################
# Beta vs Gamma
###############################################################

plt.figure(figsize=(7,6))

sc = plt.scatter(random_beta,
                 random_gamma,
                 c=random_scores,
                 s=8,
                 cmap="inferno")

plt.scatter(best_beta.cpu().item(),
            best_gamma.cpu().item(),
            color="cyan",
            marker="*",
            s=150)

plt.xlabel("Propagation Rate")

plt.ylabel("Recovery Rate")

plt.title("Propagation vs Recovery")

plt.colorbar(sc,label="Total Infection")

plt.grid(True)

plt.tight_layout()

plt.show()


# In[59]:


###############################################################
# Statistics
###############################################################

import numpy as np

random_scores = np.array(random_scores)

print()

print("================================================")

print("Random Search Statistics")

print("================================================")

print("Mean :", np.mean(random_scores))

print("Std  :", np.std(random_scores))

print("Min  :", np.min(random_scores))

print("Max  :", np.max(random_scores))

print()

print("================================================")

print("NIO")

print("================================================")

print("Score :", loss_history[-1])

print("Improvement over Random Mean (%)")

improvement = 100.0 * (
    np.mean(random_scores) - loss_history[-1]
) / np.mean(random_scores)

print(improvement)



# In[60]:


###############################################################
# Save Best Trajectory
###############################################################

torch.save(best_traj.cpu(),"best_trajectory.pt")

print()

print("Trajectory saved.")


# In[62]:


###############################################################
#
# Multiple NIO Experiments
#
###############################################################

Nruns = 10  ### 100

nio_scores = []

nio_beta = []
nio_delta = []
nio_gamma = []

print()
print("==========================================")
print("Running Multiple NIO Experiments")
print("==========================================")
print()

for run in range(Nruns):

    ###########################################################
    # New optimization variables
    ###########################################################

    z = torch.randn(3,device=device,requires_grad=True)

    optimizer = optim.Adam([z],lr=0.05)

    ###########################################################

    for iteration in range(iters):

        beta = beta_min + (beta_max-beta_min) * torch.sigmoid(z[0])

        delta = delta_min + (delta_max-delta_min) * torch.sigmoid(z[1])

        gamma = gamma_min + (gamma_max-gamma_min) * torch.sigmoid(z[2])

        traj = simulate(beta,delta,gamma)

        I = traj[:,0,2]

        loss = torch.sum(I)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    ###########################################################

    nio_scores.append(loss.item())

    nio_beta.append(beta.item())

    nio_delta.append(delta.item())

    nio_gamma.append(gamma.item())

    if run%1==0:

        print(run,loss.item())





# In[63]:


###############################################################
#
# Statistics
#
###############################################################

import numpy as np

nio_scores = np.array(nio_scores)

random_scores = np.array(random_scores)

print()

print("===================================")

print("NIO")

print("===================================")

print("Mean :",np.mean(nio_scores))

print("Std  :",np.std(nio_scores))

print("Min  :",np.min(nio_scores))

print("Max  :",np.max(nio_scores))

print()

print("===================================")

print("Random")

print("===================================")

print("Mean :",np.mean(random_scores))

print("Std  :",np.std(random_scores))

print("Min  :",np.min(random_scores))

print("Max  :",np.max(random_scores))




# In[64]:


###############################################################
#
# Boxplot
#
###############################################################

plt.figure(figsize=(6,5))

plt.boxplot(
    [random_scores,nio_scores],
    labels=["Random","NIO"]
)

plt.ylabel("Total Infection")

plt.title("Random Policies vs NIO")

plt.grid(True)

plt.tight_layout()

plt.show()


# In[65]:


###############################################################
#
# Learned Policies
#
###############################################################

plt.figure(figsize=(8,5))

plt.subplot(311)

plt.hist(nio_beta,bins=20)

plt.ylabel("Count")

plt.title("Beta")

plt.subplot(312)

plt.hist(nio_delta,bins=20)

plt.ylabel("Count")

plt.title("Delta")

plt.subplot(313)

plt.hist(nio_gamma,bins=20)

plt.ylabel("Count")

plt.title("Gamma")

plt.tight_layout()

plt.show()




# In[66]:


###############################################################
#
# Save Results
#
###############################################################

results = {

    "nio_scores":nio_scores,

    "random_scores":random_scores,

    "beta":nio_beta,

    "delta":nio_delta,

    "gamma":nio_gamma

}

torch.save(results,"cyber_results.pt")

print()

print("Finished.")

print("Results saved.")




# In[ ]:





# 
# ## Version 2 - Maximization case
# 

# In[ ]:





# In[67]:


###############################################################
#
# cyber_nio.py
#
# Neural Input Optimization (NIO)
# for Cybersecurity Policy Discovery
#
###############################################################

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# In[68]:


torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"



# In[69]:


###############################################################
# Simulation
###############################################################

dt = 0.05
Time_steps = 400
iters = 1000


# In[70]:


###############################################################
# Enterprise Network
###############################################################

TOTAL_HOSTS = 1000.0



# In[71]:


###############################################################
# Fixed Parameters
###############################################################

alpha = 0.30        # exposed -> infected
eta   = 0.15        # quarantine -> recovered



# In[72]:


###############################################################
# Policy Bounds
###############################################################

beta_min  = 0.10
beta_max  = 0.90

delta_min = 0.05
delta_max = 0.90

gamma_min = 0.05
gamma_max = 0.90


# In[73]:


###############################################################
# Initial Network
###############################################################

S0 = 985.0
E0 = 5.0
I0 = 10.0
Q0 = 0.0
R0 = 0.0


# In[74]:


###############################################################
# Cyber Dynamical System
###############################################################

class CyberStep(nn.Module):

    def __init__(self,dt=0.05):
        super().__init__()
        self.dt = dt

    def forward(self,state,beta,delta,gamma):

        S = state[:,0]
        E = state[:,1]
        I = state[:,2]
        Q = state[:,3]
        R = state[:,4]

        #######################################################
        # SEIQR Equations
        #######################################################

        dS = -beta*S*I/TOTAL_HOSTS

        dE = beta*S*I/TOTAL_HOSTS \
             - alpha*E

        dI = alpha*E \
             - delta*I \
             - gamma*I

        dQ = delta*I \
             - eta*Q

        dR = gamma*I \
             + eta*Q

        #######################################################
        # Euler
        #######################################################

        S = S + self.dt*dS
        E = E + self.dt*dE
        I = I + self.dt*dI
        Q = Q + self.dt*dQ
        R = R + self.dt*dR

        #######################################################
        # Clamp
        #######################################################

        S = torch.clamp(S,min=0)
        E = torch.clamp(E,min=0)
        I = torch.clamp(I,min=0)
        Q = torch.clamp(Q,min=0)
        R = torch.clamp(R,min=0)

        return torch.stack([S,E,I,Q,R],dim=1)




# In[75]:


model = CyberStep(dt).to(device)



# In[76]:


###############################################################
# Simulation
###############################################################

def simulate(beta, delta, gamma):

    state = torch.tensor(
        [[S0, E0, I0, Q0, R0]],
        dtype=torch.float32,
        device=device
    )

    trajectory = [state.clone()]

    for k in range(Time_steps):

        state = model(state, beta, delta, gamma)

        trajectory.append(state.clone())

    trajectory = torch.stack(trajectory)

    return trajectory




# In[77]:


###############################################################
# NIO variables
###############################################################

z = torch.randn(3, device=device, requires_grad=True)

optimizer = optim.Adam([z], lr=0.05)



# In[78]:


###############################################################
# History
###############################################################

loss_history = []

beta_history = []
delta_history = []
gamma_history = []



# In[79]:


###############################################################
# Optimization
###############################################################

print()
print("Running NIO Optimization")
print()

for iteration in range(iters):

    ###########################################################
    # Policy variables
    ###########################################################

    beta = beta_min + (beta_max-beta_min) * torch.sigmoid(z[0])

    delta = delta_min + (delta_max-delta_min) * torch.sigmoid(z[1])

    gamma = gamma_min + (gamma_max-gamma_min) * torch.sigmoid(z[2])

    ###########################################################
    # Simulate outbreak
    ###########################################################

    traj = simulate(beta, delta, gamma)

    ###########################################################
    # States
    ###########################################################

    S = traj[:,0,0]
    E = traj[:,0,1]
    I = traj[:,0,2]
    Q = traj[:,0,3]
    R = traj[:,0,4]

    ###########################################################
    # Objective
    #
    # When positive you Minimize total infected over the outbreak
    ###########################################################
    
    ## When negative Now NIO searches for policies that produce the largest outbreak.
    ## loss = -torch.sum(I)

    total_infected = -torch.sum(I)

    loss = total_infected

    ###########################################################
    # Gradient step
    ###########################################################

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    ###########################################################
    # Save history
    ###########################################################

    loss_history.append(loss.item())

    beta_history.append(beta.item())
    delta_history.append(delta.item())
    gamma_history.append(gamma.item())

    ###########################################################
    # Progress
    ###########################################################

    if iteration % 100 == 0:

        print(
            "Iter:",
            iteration,
            "Loss:",
            round(loss.item(),3),
            "beta:",
            round(beta.item(),3),
            "delta:",
            round(delta.item(),3),
            "gamma:",
            round(gamma.item(),3)
        )


# In[80]:


###############################################################
# Best policy
###############################################################

best_beta = beta.detach()

best_delta = delta.detach()

best_gamma = gamma.detach()

best_traj = simulate(best_beta, best_delta, best_gamma)

print()
print("Optimization Finished")
print()

print("Best Policy")

print("beta :", best_beta.item())
print("delta:", best_delta.item())
print("gamma:", best_gamma.item())


# In[81]:


###############################################################
# Random Policy Search
###############################################################

print()
print("Running Random Search")
print()

Nrandom = 5000

random_scores = []

random_beta = []
random_delta = []
random_gamma = []

best_random_score = 1e30

best_random_beta = None
best_random_delta = None
best_random_gamma = None

for k in range(Nrandom):

    beta = beta_min + (beta_max-beta_min)*torch.rand(1,device=device)

    delta = delta_min + (delta_max-delta_min)*torch.rand(1,device=device)

    gamma = gamma_min + (gamma_max-gamma_min)*torch.rand(1,device=device)

    traj = simulate(beta,delta,gamma)

    I = traj[:,0,2]

    score = torch.sum(I)

    random_scores.append(score.item())

    random_beta.append(beta.item())
    random_delta.append(delta.item())
    random_gamma.append(gamma.item())

    if score.item() < best_random_score:

        best_random_score = score.item()

        best_random_beta = beta.item()
        best_random_delta = delta.item()
        best_random_gamma = gamma.item()



# In[82]:


###############################################################
# Print comparison
###############################################################

print()
print("---------------------------------------")
print("Random Search")
print("---------------------------------------")
print("Best Score :", best_random_score)
print("beta       :", best_random_beta)
print("delta      :", best_random_delta)
print("gamma      :", best_random_gamma)

print()

print("---------------------------------------")
print("NIO")
print("---------------------------------------")
print("Score :", loss_history[-1])
print("beta  :", best_beta.item())
print("delta :", best_delta.item())
print("gamma :", best_gamma.item())


# In[83]:


###############################################################
# Plot Optimization Loss
###############################################################

plt.figure(figsize=(7,4))

plt.plot(loss_history)

plt.grid(True)

plt.xlabel("Iteration")

plt.ylabel("Loss")

plt.title("NIO Optimization")

plt.tight_layout()

plt.show()


# In[84]:


###############################################################
# Best Trajectory
###############################################################

S = best_traj[:,0,0].cpu().numpy()
E = best_traj[:,0,1].cpu().numpy()
I = best_traj[:,0,2].cpu().numpy()
Q = best_traj[:,0,3].cpu().numpy()
R = best_traj[:,0,4].cpu().numpy()

plt.figure(figsize=(8,5))

plt.plot(S,label="Susceptible")
plt.plot(E,label="Exposed")
plt.plot(I,label="Infected")
plt.plot(Q,label="Quarantined")
plt.plot(R,label="Recovered")

plt.grid(True)

plt.legend()

plt.xlabel("Time")

plt.ylabel("Hosts")

plt.title("SEIQR Dynamics Using NIO Policy")

plt.tight_layout()

plt.show()



# In[85]:


###############################################################
# Histogram
###############################################################

plt.figure(figsize=(7,4))

plt.hist(random_scores,
         bins=50)

plt.axvline(loss_history[-1],
            color='red',
            linewidth=3,
            label="NIO")

plt.legend()

plt.xlabel("Total Infection")

plt.ylabel("Frequency")

plt.title("Random Policies vs NIO")

plt.grid(True)

plt.tight_layout()

plt.show()


# In[86]:


###############################################################
# Policy Evolution
###############################################################

plt.figure(figsize=(8,5))

plt.plot(beta_history,label="beta")

plt.plot(delta_history,label="delta")

plt.plot(gamma_history,label="gamma")

plt.legend()

plt.grid(True)

plt.xlabel("Iteration")

plt.ylabel("Value")

plt.title("Policy Parameters During Optimization")

plt.tight_layout()

plt.show()



# In[87]:


###############################################################
# Scatter Plot of Policy Space
###############################################################

plt.figure(figsize=(7,6))

sc = plt.scatter(random_beta,
                 random_delta,
                 c=random_scores,
                 s=8,
                 cmap="viridis")

plt.scatter(best_beta.cpu().item(),
            best_delta.cpu().item(),
            color="red",
            s=150,
            marker="*",
            label="NIO")

plt.xlabel("Beta (Propagation Rate)")

plt.ylabel("Delta (Detection Rate)")

plt.title("Policy Space")

plt.legend()

plt.colorbar(sc,label="Total Infection")

plt.grid(True)

plt.tight_layout()

plt.show()


# In[88]:


###############################################################
# Delta vs Gamma
###############################################################

plt.figure(figsize=(7,6))

sc = plt.scatter(random_delta,
                 random_gamma,
                 c=random_scores,
                 s=8,
                 cmap="plasma")

plt.scatter(best_delta.cpu().item(),
            best_gamma.cpu().item(),
            color="red",
            marker="*",
            s=150)

plt.xlabel("Detection Rate")

plt.ylabel("Recovery Rate")

plt.title("Detection vs Recovery Policies")

plt.colorbar(sc,label="Total Infection")

plt.grid(True)

plt.tight_layout()

plt.show()


# In[89]:


###############################################################
# Beta vs Gamma
###############################################################

plt.figure(figsize=(7,6))

sc = plt.scatter(random_beta,
                 random_gamma,
                 c=random_scores,
                 s=8,
                 cmap="inferno")

plt.scatter(best_beta.cpu().item(),
            best_gamma.cpu().item(),
            color="cyan",
            marker="*",
            s=150)

plt.xlabel("Propagation Rate")

plt.ylabel("Recovery Rate")

plt.title("Propagation vs Recovery")

plt.colorbar(sc,label="Total Infection")

plt.grid(True)

plt.tight_layout()

plt.show()


# In[90]:


###############################################################
# Statistics
###############################################################

import numpy as np

random_scores = np.array(random_scores)

print()

print("================================================")

print("Random Search Statistics")

print("================================================")

print("Mean :", np.mean(random_scores))

print("Std  :", np.std(random_scores))

print("Min  :", np.min(random_scores))

print("Max  :", np.max(random_scores))

print()

print("================================================")

print("NIO")

print("================================================")

print("Score :", loss_history[-1])

print("Improvement over Random Mean (%)")

improvement = 100.0 * (
    np.mean(random_scores) - loss_history[-1]
) / np.mean(random_scores)

print(improvement)



# In[91]:


###############################################################
# Save Best Trajectory
###############################################################

torch.save(best_traj.cpu(),"best_trajectory.pt")

print()

print("Trajectory saved.")


# In[93]:


###############################################################
#
# Multiple NIO Experiments
#
###############################################################

Nruns = 4

nio_scores = []

nio_beta = []
nio_delta = []
nio_gamma = []

print()
print("==========================================")
print("Running Multiple NIO Experiments")
print("==========================================")
print()

for run in range(Nruns):

    ###########################################################
    # New optimization variables
    ###########################################################

    z = torch.randn(3,device=device,requires_grad=True)

    optimizer = optim.Adam([z],lr=0.05)

    ###########################################################

    for iteration in range(iters):

        beta = beta_min + (beta_max-beta_min) * torch.sigmoid(z[0])

        delta = delta_min + (delta_max-delta_min) * torch.sigmoid(z[1])

        gamma = gamma_min + (gamma_max-gamma_min) * torch.sigmoid(z[2])

        traj = simulate(beta,delta,gamma)

        I = traj[:,0,2]

        loss = -torch.sum(I)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    ###########################################################

    nio_scores.append(loss.item())

    nio_beta.append(beta.item())

    nio_delta.append(delta.item())

    nio_gamma.append(gamma.item())

    if run%1==0:

        print(run,loss.item())





# In[94]:


###############################################################
#
# Statistics
#
###############################################################

import numpy as np

nio_scores = np.array(nio_scores)

random_scores = np.array(random_scores)

print()

print("===================================")

print("NIO")

print("===================================")

print("Mean :",np.mean(nio_scores))

print("Std  :",np.std(nio_scores))

print("Min  :",np.min(nio_scores))

print("Max  :",np.max(nio_scores))

print()

print("===================================")

print("Random")

print("===================================")

print("Mean :",np.mean(random_scores))

print("Std  :",np.std(random_scores))

print("Min  :",np.min(random_scores))

print("Max  :",np.max(random_scores))




# In[95]:


###############################################################
#
# Boxplot
#
###############################################################

plt.figure(figsize=(6,5))

plt.boxplot(
    [random_scores,nio_scores],
    labels=["Random","NIO"]
)

plt.ylabel("Total Infection")

plt.title("Random Policies vs NIO")

plt.grid(True)

plt.tight_layout()

plt.show()



# In[96]:


###############################################################
#
# Learned Policies
#
###############################################################

plt.figure(figsize=(8,5))

plt.subplot(311)

plt.hist(nio_beta,bins=20)

plt.ylabel("Count")

plt.title("Beta")

plt.subplot(312)

plt.hist(nio_delta,bins=20)

plt.ylabel("Count")

plt.title("Delta")

plt.subplot(313)

plt.hist(nio_gamma,bins=20)

plt.ylabel("Count")

plt.title("Gamma")

plt.tight_layout()

plt.show()




# In[97]:


###############################################################
#
# Save Results
#
###############################################################

results = {

    "nio_scores":nio_scores,

    "random_scores":random_scores,

    "beta":nio_beta,

    "delta":nio_delta,

    "gamma":nio_gamma

}

torch.save(results,"cyber_results.pt")

print()

print("Finished.")

print("Results saved.")



# In[ ]:





# 
# ## Attacker vs. Defender
# 

# In[98]:


def run_nio(maximize=False):

    z = torch.randn(3, device=device, requires_grad=True)

    optimizer = optim.Adam([z], lr=0.05)

    loss_history = []

    for iteration in range(iters):

        beta = beta_min + (beta_max-beta_min) * torch.sigmoid(z[0])

        delta = delta_min + (delta_max-delta_min) * torch.sigmoid(z[1])

        gamma = gamma_min + (gamma_max-gamma_min) * torch.sigmoid(z[2])

        traj = simulate(beta, delta, gamma)

        I = traj[:,0,2]

        objective = torch.sum(I)

        if maximize:
            loss = -objective
        else:
            loss = objective

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    return (
        beta.detach(),
        delta.detach(),
        gamma.detach(),
        traj.detach(),
        loss_history
    )





# In[99]:


#######################################################
# Defender
#######################################################

beta_min_case, delta_min_case, gamma_min_case, traj_min, loss_min = run_nio(False)


# In[100]:


#######################################################
# Attacker
#######################################################

beta_max_case, delta_max_case, gamma_max_case, traj_max, loss_max = run_nio(True)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




