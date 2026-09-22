# Phase 07 — DQN Methodology Lock

## Explicitly stated in the manuscript

### State
s_t = (L_t, F_t, H_t, T_t)

Where:
- L_t = active truck location
- F_t = blockchain-verified bin fill level
- H_t = localized hazard severity
- T_t = elapsed full-bin time

### Action
The action selects the next bin for the active truck.

Unverified or spoofed telemetry is excluded from the candidate set.

### Reward
R_t = w_c C_t - w_d D_t - w_h H_t - w_s S_t

Weights:
- collection reward = 1.0
- distance penalty = 0.5
- hazard-delay penalty = 5.0
- SLA-violation penalty = 3.0

### Routing behavior
- Centralized sequential coordinator
- One target selected at a time for the active vehicle
- Critical hazard states override ordinary distance minimization
- Near-full bins are considered around the 90% threshold
- Bins predicted to overflow before the next collection cycle are also considered
- Representative scenario mentions 75-80% bins predicted to overflow

### DQN architecture
- Hidden layer 1: 256 ReLU
- Hidden layer 2: 128 ReLU
- Hidden layer 3: 64 ReLU
- Linear output
- Discount factor gamma = 0.95
- Replay buffer = 100,000
- Batch size = 64
- Epsilon = 1.0 -> 0.01
- Linear epsilon decay
- Optimizer = Adam
- 100 episodes per scenario

### Convergence
Learning is accepted after episodic reward flattens and training loss stabilizes.

## Missing / underspecified in the manuscript

The following must be explicitly reconstructed and then added to the revised methodology:

1. Learning rate
2. Exact numerical encoding of DQN state
3. Exact DQN action-output representation
4. Target-network update policy
5. Warm-up size before replay-buffer training
6. Exact epsilon-decay duration
7. Hazard-event generation distribution
8. Hazard severity scale
9. Fill-rate evolution between time steps
10. Overflow-prediction model
11. Exact SLA elapsed-time simulation
12. Exact episode termination criteria
13. Training/evaluation seed separation

These values must not be reverse-engineered to reproduce the old 45 L result.