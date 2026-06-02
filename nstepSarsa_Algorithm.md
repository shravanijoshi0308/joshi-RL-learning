# n-step Sarsa — Algorithm Notes
## Source: Sutton & Barto, Chapter 7.2, p.147

---

## The Exact Algorithm (Textbook p.147)

```
n-step Sarsa for estimating Q ~ q* or q_pi

Initialize Q(s, a) arbitrarily, for all s in S, a in A
Initialize pi to be epsilon-greedy with respect to Q, or to a fixed given policy
Algorithm parameters: step size alpha in (0, 1], small epsilon > 0, a positive integer n
All store and access operations (for St, At, and Rt) can take their index mod n + 1

Loop for each episode:
    Initialize and store S0 != terminal
    Select and store an action A0 ~ pi(·|S0)
    T <- infinity

    Loop for t = 0, 1, 2, ...:
    |
    |   If t < T, then:
    |       Take action At
    |       Observe and store the next reward as Rt+1 and the next state as St+1
    |       If St+1 is terminal, then:
    |           T <- t + 1
    |       else:
    |           Select and store an action At+1 ~ pi(·|St+1)
    |
    |   tau <- t - n + 1      (tau is the time whose estimate is being updated)
    |
    |   If tau >= 0:
    |       G <- sum from i = tau+1 to min(tau+n, T) of  gamma^{i-tau-1} * Ri
    |       If tau + n < T, then:
    |           G <- G + gamma^n * Q(S_{tau+n}, A_{tau+n})       (this is G_{tau:tau+n})
    |       Q(S_tau, A_tau) <- Q(S_tau, A_tau) + alpha * [G - Q(S_tau, A_tau)]
    |       If pi is being learned, then ensure pi(·|S_tau) is epsilon-greedy wrt Q
    |
    Until tau = T - 1
```

---

## Every Line Explained
---

### Line 1
```
Initialize Q(s, a) arbitrarily, for all s in S, a in A
```

**What it does:**
Creates the Q-table. One row per state, one column per action.
Every cell is filled with a number — usually zero, or a small random value.

**Why it is needed:**
The algorithm reads and writes Q-values constantly. Without this table nothing can be stored.

**What happens if you remove it:**
The algorithm crashes immediately. There is nowhere to store or look up values.

**Beginner note:**
Think of this as opening a blank spreadsheet before you start filling it in.
At the start the agent knows nothing — every cell is zero.

---

### Line 2
```
Initialize pi to be epsilon-greedy with respect to Q
```

**What it does:**
Sets the starting policy. A policy is the rule the agent uses to pick actions.
Epsilon-greedy means: with probability epsilon pick a random action, otherwise pick the action with the highest Q-value.

**Why it is needed:**
The agent needs a policy before it can take any action. Without a policy it cannot start moving.

**What happens if you remove it:**
The agent has no rule for choosing actions. It cannot begin.

**Beginner note:**
At the start all Q-values are zero, so all actions look equally good.
The agent will explore almost randomly at first — that is correct and expected.

---

### Line 3
```
Algorithm parameters: alpha in (0,1], epsilon > 0, n (positive integer)
```

**What it does:**
Defines the three numbers you must set before running the algorithm.

| Parameter | Name | What it controls |
|---|---|---|
| alpha | Learning rate | How much Q-values change per update. Small = slow stable. Large = fast unstable. |
| epsilon | Exploration rate | How often the agent tries random actions. 0.1 means 10% random. |
| n | Step count | How many real rewards to collect before each update. |

**Why it is needed:**
These three numbers control the entire behaviour of the algorithm. Change any one of them and the learning speed, stability, and final policy all change.

**Beginner note:**
A good starting point is alpha = 0.5, epsilon = 0.1, n = 4.
Your professor may ask you to change these — each one has a clear and testable effect.

---

### Line 4
```
All store and access operations (for St, At, and Rt) can take their index mod n + 1
```

**What it does:**
A memory-saving trick. Instead of storing the entire history of an episode, you only keep the last n+1 states, actions, and rewards in a circular buffer.

**Why it is needed:**
Episodes can be very long. Storing everything wastes memory. Since we only ever look n steps back, we only need to keep n+1 items at any time.

**What happens if you remove it:**
The algorithm still works — you just use more memory. For simple gridworlds this line is not critical. For large environments it matters a lot.

**Beginner note:**
In the Python code this is handled automatically by using lists and indexing. You do not need to implement the circular buffer manually for small problems.

---

### Line 5 — Episode Loop
```
Loop for each episode:
```

**What it does:**
Repeats the entire learning process from start to goal, over and over.
One pass = one episode = one attempt from start state to terminal state.

**Why it is needed:**
A single episode is not enough to learn. The agent needs hundreds or thousands of attempts before Q-values become accurate.

**Beginner note:**
Think of this as your driving lesson. Each episode is one full drive. You need many drives before you become a good driver.

---

### Line 6
```
Initialize and store S0 != terminal
Select and store an action A0 ~ pi(·|S0)
T <- infinity
```

**What it does:**
Three things happen before the episode begins:

1. Places the agent at the start state S0
2. Picks the first action A0 using the current policy
3. Sets T = infinity — meaning we do not know yet when the episode will end

**Why T starts as infinity:**
T is the time step when the agent reaches the terminal state. We do not know that in advance. It gets updated the moment a terminal state is reached.

**Why A0 is stored before the loop:**
The n-step algorithm needs an action stored at every time step. The first action must be chosen before t=0 begins.

---

### Line 7 — Time Step Loop
```
Loop for t = 0, 1, 2, ...:
```

**What it does:**
Advances time one step at a time. Each iteration is one step in the environment.
Two things happen inside this loop every iteration:
1. Move forward (if episode not done)
2. Update a Q-value from n steps ago (if enough steps have passed)

---

### Line 8
```
If t < T, then:
    Take action At
    Observe and store the next reward as Rt+1 and the next state as St+1
```

**What it does:**
Takes one step in the environment. The agent does action At, the environment responds with a new state St+1 and a reward Rt+1. Both are stored.

**Why the condition t < T:**
Once the episode ends (T is set), the agent stops moving. But the loop keeps running to finish remaining Q-updates. The condition prevents the agent from taking more steps after reaching the goal.

**What happens if you remove the condition:**
The agent keeps moving after the goal is reached. The episode never ends properly.

---

### Line 9
```
If St+1 is terminal, then:
    T <- t + 1
else:
    Select and store an action At+1 ~ pi(·|St+1)
```

**What it does:**
Two branches:

- If the new state is the goal (terminal): record T = t+1. The episode is over. No next action is needed.
- If not terminal: choose the next action using epsilon-greedy and store it.

**Why the next action is stored:**
The bootstrap term in G uses Q(S_{tau+n}, A_{tau+n}). That action A_{tau+n} must have been chosen by the same epsilon-greedy policy. Storing it now ensures the update is on-policy.

**This is the on-policy guarantee:**
The action A_{t+1} was chosen by the current epsilon-greedy policy — the same policy we are improving. This is what makes n-step Sarsa on-policy.

---

### Line 10 — The Most Important Line
```
tau <- t - n + 1
```

**What it does:**
Computes tau — the time step whose Q-value will be updated in this iteration.
tau is always n steps behind the current time t.

**Why the delay:**
To update tau correctly, you need rewards R_{tau+1}, R_{tau+2}, ..., R_{tau+n}.
Those rewards only become available n steps later. So the update must wait.

**Example:**

```
n = 4

t = 0  -->  tau = -3   (no update yet, tau < 0)
t = 1  -->  tau = -2   (no update yet)
t = 2  -->  tau = -1   (no update yet)
t = 3  -->  tau =  0   (first update — for S0, A0)
t = 4  -->  tau =  1   (update for S1, A1)
t = 5  -->  tau =  2   (update for S2, A2)
...
```

For the first n-1 steps there are no updates at all. This is normal.

---

### Line 11
```
If tau >= 0:
    G <- sum from i = tau+1 to min(tau+n, T) of  gamma^{i-tau-1} * Ri
```

**What it does:**
Computes the first part of G — the sum of n discounted real rewards.

Breaking down the sum:

```
i = tau+1:   gamma^0  * R_{tau+1}   =  1.0  * R_{tau+1}
i = tau+2:   gamma^1  * R_{tau+2}   =  γ    * R_{tau+2}
i = tau+3:   gamma^2  * R_{tau+3}   =  γ²   * R_{tau+3}
...
i = tau+n:   gamma^{n-1} * R_{tau+n}
```

**Why min(tau+n, T):**
If the episode ends before n steps (tau+n >= T), you only sum up to T — as many real rewards as are available. You do not go beyond the terminal state.

**Beginner note:**
Each reward is discounted more the further it is in the future. A reward 3 steps away is worth gamma^2 as much as an immediate reward. If gamma = 1.0 (which is common in gridworlds), all rewards are worth the same regardless of when they happen.

---

### Line 12
```
If tau + n < T, then:
    G <- G + gamma^n * Q(S_{tau+n}, A_{tau+n})     (this is G_{tau:tau+n})
```

**What it does:**
Adds the bootstrap term to G. This completes the n-step return (Equation 7.4).

**Why the condition tau + n < T:**
If tau+n >= T, the episode already ended. The terminal state has Q = 0 by definition. Adding zero changes nothing, so the bootstrap term is simply skipped.

**What the bootstrap term means:**
`gamma^n * Q(S_{tau+n}, A_{tau+n})` is the agent's current estimate of all future rewards beyond step tau+n. It is not a real reward — it is an approximation. This is what introduces bias into the algorithm.

**Two cases:**

```
tau + n < T   (episode still running at step tau+n)
    G = n real rewards  +  gamma^n * Q(S_{tau+n}, A_{tau+n})

tau + n >= T  (episode ended within n steps)
    G = real rewards only  (no bootstrap — same as Monte Carlo for these steps)
```

---

### Line 13 — The Learning Step
```
Q(S_tau, A_tau) <- Q(S_tau, A_tau) + alpha * [G - Q(S_tau, A_tau)]
```

**What it does:**
Updates the Q-value for state S_tau and action A_tau. This is the actual learning step.
This is Equation 7.5 from the textbook.

**Breaking it down:**

```
new Q  =  old Q  +  alpha  x  (G  -  old Q)
                               └──────────┘
                                 TD error
```

| TD error (G - Q) | Meaning | What happens |
|---|---|---|
| Positive | G > Q — you underestimated | Q moves up toward G |
| Negative | G < Q — you overestimated | Q moves down toward G |
| Zero | G = Q — perfect estimate | Nothing changes |

**Why multiply by alpha:**
Alpha controls the step size. Large alpha = big jump toward G. Small alpha = small nudge.
Too large and Q-values overshoot and oscillate. Too small and learning is very slow.

---

### Line 14
```
If pi is being learned, then ensure pi(·|S_tau) is epsilon-greedy wrt Q
```

**What it does:**
After Q(S_tau, A_tau) changes, the policy at state S_tau may need to change too.
This line regenerates the epsilon-greedy policy at S_tau based on the updated Q-values.

**Why it is needed:**
The policy must stay consistent with Q. If Q(S_tau, Right) just became the highest value in that row, the policy should now favour Right. Without this line the policy never improves.

**What happens if you remove it:**
The Q-table improves but the policy stays fixed. The agent learns internally but never acts on what it learned.

---

### Line 15
```
Until tau = T - 1
```

**What it does:**
The episode ends when tau reaches T-1 — meaning every time step from 0 to T-1 has been updated.

**Why not stop at t = T:**
When t = T the agent has finished moving. But tau = T - n, which means the last n states have not been updated yet. The loop keeps running with t increasing and tau catching up, until tau = T-1 and all updates are done.

**The drain phase:**

```
Episode ends at T = 10, n = 4

After reaching goal:
    t=10, tau=7  -->  update step 7
    t=11, tau=8  -->  update step 8
    t=12, tau=9  -->  update step 9  (= T-1, loop ends)
```

---

## Summary Table — Every Line at a Glance

| Line | Code | Plain English |
|---|---|---|
| 1 | `Initialize Q(s,a)` | Create blank Q-table, all zeros |
| 2 | `Initialize pi` | Set starting policy to epsilon-greedy |
| 3 | `Parameters: alpha, epsilon, n` | Set learning rate, exploration, step count |
| 4 | `index mod n+1` | Memory saving — only keep last n+1 items |
| 5 | `Loop for each episode` | Repeat many times to learn |
| 6 | `S0, A0, T=inf` | Place agent at start, pick first action, T unknown |
| 7 | `Loop for t = 0,1,2...` | Advance time one step at a time |
| 8 | `If t < T: take At` | Move forward only while episode is active |
| 9 | `If terminal: T=t+1 else: store At+1` | Record episode end or choose next action |
| 10 | `tau = t - n + 1` | The delayed update index — n steps behind |
| 11 | `G = sum of discounted rewards` | Collect n real rewards into target G |
| 12 | `G += gamma^n * Q(...)` | Add bootstrap term if episode not done |
| 13 | `Q(S_tau, A_tau) += alpha*[G-Q]` | The actual learning step |
| 14 | `Update pi to epsilon-greedy` | Keep policy in sync with Q-table |
| 15 | `Until tau = T-1` | Run until all steps in episode are updated |

---
## lets apply our Driving Analogy intution to the algorithm 
## The Driving Analogy Mapped to the Code

| Algorithm line | Driving analogy |
|---|---|
| `Initialize Q-table` | Coach starts with no knowledge of your driving ability |
| `Initialize pi` | Coach decides to give random feedback at first (exploring) |
| `alpha, epsilon, n` | Coach sets: how much weight to give new observations, how adventurous to be, how many actions to watch before giving feedback |
| `Store S, A, R in lists` | Coach is watching and writing notes on each action |
| `T <- infinity` | Coach does not know when the drive will end |
| `tau = t - n + 1` | Coach waits until they have seen n actions before giving any feedback |
| `t < T check` | Coach is still watching — the drive is not over yet |
| `T <- t+1` | You reached the destination — drive is over |
| `Store At+1` | Coach notes which action you are about to take next |
| `Compute G` | Coach forms their verdict based on n observations they recorded |
| `Bootstrap term` | Coach estimates how the rest of the drive would have gone from that point |
| `Q update` | Coach gives you feedback — nudges your understanding of that action |
| `Policy update` | You adjust your driving decisions based on the feedback |
| `Until tau = T-1` | Coach finishes giving feedback on the last few actions from the end of the drive |

---

## Reference

Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
MIT Press. Chapter 7: n-step Bootstrapping, Section 7.2: n-step Sarsa. p.147.