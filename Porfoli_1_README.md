# RL Portfolio — n-step Sarsa vs Gridworld Algorithms

> **Course:** Reinforcement Learning · Portfolio Task 1  
> **Chapter:** Sutton & Barto — *Reinforcement Learning* (2nd Ed.) · Chapter 7.2  
> **Topic:** Evaluating n-step Sarsa against earlier learning algorithms across Gridworld variants

---

## Intuition First — The Driving Lesson Analogy

Before any equations, here is the core idea in plain language.

Imagine you are taking driving lessons. Your coach can give feedback in **three different ways**:

---

### Scenario 1 — Feedback Only at the End of the Trip

> *"The trip was 60 minutes. Your coach tells you how it went at the very end."*

You drove for an hour. Some turns were great, some were bad. But by the time your coach gives feedback, you have **completely forgotten which specific action at which moment** produced the best outcome. You only know the final result — good trip or bad trip.

**This is Monte Carlo.**  
The agent waits until the episode ends, then updates everything backwards using the final return. It learns from the complete picture but with no fine-grained credit.

#### Return (target)

```
Gt = R_{t+1} + γ·R_{t+2} + γ²·R_{t+3} + ... + γ^{T-t-1}·R_T
     <────────────────────────────────────────────────────────>
              ALL real rewards until episode ends (no bootstrap)
```

#### Update Rule

```
Q(St, At) ← Q(St, At) + α [ Gt − Q(St, At) ]
                             <──────────────>
                              full return target
```

#### Symbol Reference

| Symbol | Name | What It Means |
|---|---|---|
| `Gt` | Full return | Sum of ALL discounted rewards from step t to end of episode |
| `R_{t+k}` | Reward | Real reward k steps after time t |
| `T` | Terminal time | Final time step of the episode |
| `γ^{T-t-1}` | Discount | Discount factor applied to the final reward |
| `α` | Learning rate | How much to move toward the new estimate |

---

### Scenario 2 — Feedback After Every Single Action

> *"Your coach gives feedback after every turn, every brake, every steering input — immediately."*

Now imagine: you take a turn at a junction. Your coach says *"bad"* immediately. But three turns later, it turns out that junction was actually the right route — the coach judged too early with too little information. Feedback is quick, but it is based on only the very next moment. **Learning is fast per-step but shallow.**

**This is 1-step TD / Sarsa(0).**  
The agent updates after every single action using only the immediate next reward. Fast, but the signal is narrow — only one step of real information before bootstrapping.

#### Return (target)

```
G_{t:t+1} = R_{t+1}  +  γ · Q(S_{t+1}, A_{t+1})
             ────────     ───────────────────────
             1 real         bootstrap estimate
             reward         (not a real reward!)
```

#### Update Rule — Sarsa (on-policy)

```
Q(St, At) ← Q(St, At) + α [ R_{t+1} + γ·Q(S_{t+1}, A_{t+1}) − Q(St, At) ]
                             <──────────────────────────────>   <─────────>
                                       TD target                 current Q
                             └─────────────────────────────────────────────┘
                                              TD error  δt
```

#### Update Rule — Q-learning (off-policy variant)

```
Q(St, At) ← Q(St, At) + α [ R_{t+1} + γ · max_a Q(S_{t+1}, a) − Q(St, At) ]
                                               <────────────────>
                                               greedy max over all
                                               next actions (off-policy)
```

#### Symbol Reference

| Symbol | Name | What It Means |
|---|---|---|
| `G_{t:t+1}` | 1-step return | 1 real reward + bootstrap |
| `R_{t+1}` | Reward | The one real reward after taking action `At` |
| `Q(S_{t+1}, A_{t+1})` | Bootstrap (Sarsa) | Next state-action value chosen by ε-greedy policy |
| `max_a Q(S_{t+1}, a)` | Bootstrap (Q-learning) | Best possible next action value — greedy, off-policy |
| `δt` | TD error | How wrong the current estimate was |
| `α` | Learning rate | How much to move toward the new estimate |

---

### Scenario 3 — Feedback After Observing a Few Actions (the sweet spot)

> *"Your coach watches you make 4 actions — Action 1 -> Action 2 -> Action 3 -> Action 4 — and then gives you feedback based on what they observed."*

Now your coach has seen enough to make a **meaningful, informed judgment**. Not so little information that feedback is shallow. Not so much waiting that you forgot what happened. The feedback covers a **sequence of actions**, so credit flows back across multiple steps at once.

**This is n-step Sarsa.**  
The agent collects `n` real rewards, then updates. It propagates credit `n` steps backward in a single episode — faster than 1-step TD, more stable than Monte Carlo.

```
+----------+----------+----------+--------------+
| Action 1 | Action 2 | Action 3 |   Feedback   |
+----------+----------+----------+--------------+
     R1          R2        R3     -> update Q(S0, A0)
```

#### Return (target) — Equation 7.4

```
G_{t:t+n} = R_{t+1} + γ·R_{t+2} + γ²·R_{t+3} + ... + γ^{n-1}·R_{t+n} + γⁿ·Q(S_{t+n}, A_{t+n})
             <──────────────────────────────────────────────────────────>   <────────────────────>
                           n real rewards (discounted)                           bootstrap term
                                                                            (only if episode not done)
```

#### Update Rule — Equation 7.5

```
Q(Sτ, Aτ) ← Q(Sτ, Aτ) + α [ G_{τ:τ+n} − Q(Sτ, Aτ) ]
                              <───────────────────────>
                                      TD error

where  τ = t − n + 1   (the state being updated is n steps BEHIND the current time)
```

#### Symbol Reference

| Symbol | Name | What It Means |
|---|---|---|
| `G_{t:t+n}` | n-step return | The update target — what we want Q to become |
| `R_{t+1}` | Reward at t+1 | The real reward received after taking action at time t |
| `γ` (gamma) | Discount factor | How much future rewards are worth (0 to 1). γ=1 means care equally about all future rewards |
| `γⁱ` | Discount power | Reward i steps away is worth γⁱ as much as the immediate reward |
| `n` | Step count | How many real rewards to collect before bootstrapping |
| `Q(S_{t+n}, A_{t+n})` | Bootstrap term | Our current estimate of future value after n steps. Not a real reward — an estimate |
| `t` | Current time | The time step whose Q-value we are updating (n steps ago) |
| `α` | Learning rate | How much to move toward the new estimate |
| `τ` (tau) | Update time | The time step being updated — τ = t − n + 1 (n steps behind current time) |

---

### The Three Formulas Side by Side

The **only difference** between all three algorithms is what goes inside `G` (the target).  
The update rule shape is **identical** for all three:

```
Q(S, A) <- Q(S, A) + α [ G  −  Q(S, A) ]
                         ^        ^
                       target   current estimate
```

```
+--------------+-------------------------------------+------------------------------+
|  Algorithm   |        Real Rewards Used           |         Bootstrap?            |
+--------------+-------------------------------------+------------------------------+
| Monte Carlo  |  ALL rewards R_{t+1} ... R_T       |  None — waits for end        |
|   (n = inf)  |                                     |                              |
+--------------+-------------------------------------+------------------------------+
|  n-step      |  n rewards R_{t+1} ... R_{t+n}     |  y^n * Q(S_{t+n}, A_{t+n})  |
|  Sarsa       |                                     |  after n steps               |
+--------------+-------------------------------------+------------------------------+
|  1-step TD   |  1 reward  R_{t+1} only            |  y * Q(S_{t+1}, A_{t+1})    |
|  Sarsa(0)    |                                     |  immediately                 |
+--------------+-------------------------------------+------------------------------+
```

---

### The Spectrum

```
  Monte Carlo          n-step Sarsa           1-step TD / Sarsa
    (n = inf)       (n = 4, 8, 16 ...)             (n = 1)

  Wait for            Sweet spot:               Update after
 episode end       n real rewards then          every single
                   update n steps back             action

 Low variance  <------------------------------------------->  High variance
 High bias     <------------------------------------------->  Low bias
 Slow online   <------------------------------------------->  Fast online
```

> "n-step methods enable bootstrapping to occur over multiple steps, freeing us from the tyranny of the single time step."  
> — Sutton & Barto, p. 141

---

## Limitations of the Two Extremes

### Monte Carlo

| Limitation | Explanation |
|---|---|
| **Too late** | Waits for the episode to end before making any updates |
| **High variance** | Long episodes produce very noisy returns — small random events early in the episode change the total dramatically |
| **Useless online** | Cannot learn anything until the episode terminates |
| **Slow in practice** | Needs many full episodes before useful Q-values emerge |

### 1-step TD / Sarsa(0)

| Limitation | Explanation |
|---|---|
| **Too narrow** | Only the immediately preceding action gets updated each step |
| **Slow credit** | In a 20-step path to the goal, 1-step Sarsa needs 20 episodes just to propagate the goal reward back to the start |
| **Tyranny of the single time step** | The update interval and the action interval are forced to be equal |
| **Shallow signal** | Bootstrap estimate after just 1 reward — high bias |

### n-step Sarsa solves both

- Updates at every step (unlike Monte Carlo)
- Uses `n` real rewards before bootstrapping (unlike 1-step TD)
- Tunable: choose `n` to match your environment's reward structure

---

## Reference

> Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).  
> MIT Press. Chapter 7: n-step Bootstrapping — Section 7.2: n-step Sarsa.

---

*Made as part of the Reinforcement Learning university course portfolio.*