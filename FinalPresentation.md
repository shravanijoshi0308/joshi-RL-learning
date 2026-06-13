# n-step Sarsa — Reinforcement Learning Portfolio
### Chapter 7.2 | Sutton & Barto | THWS Würzburg-Schweinfurt

> *"n-step methods enable bootstrapping to occur over multiple steps, freeing us from the tyranny of the single time step."*
> — Sutton & Barto, Reinforcement Learning (2nd Ed.), p.141

---

## Table of Contents

1. [The Driving Analogy — Intuition](#1-the-driving-analogy--intuition)
2. [Limitations of the Two Extremes](#2-limitations-of-the-two-extremes)
3. [Why n-step Sarsa](#3-why-n-step-sarsa)
4. [The n-step Spectrum](#4-the-n-step-spectrum)
5. [Bias and Variance](#5-bias-and-variance)
6. [Can We Reduce the Bias-Variance Tradeoff?](#6-can-we-reduce-the-bias-variance-tradeoff)
7. [The Formula]
(#7-the-formula)
---

## 1. The Driving Analogy — Intuition

Before any formula, think about **when a coach should give feedback to a driver.**

There are three coaching styles. Each one maps exactly to one algorithm.

---

### Scenario 1 — Feedback at the End of the Trip = Monte Carlo

> "Your coach sits quietly for the entire 60-minute drive. They watch everything but say nothing. At the very end of the trip they give you one verdict — good drive or bad drive. The problem is by then you have forgotten which specific turn, which specific brake, which specific decision actually mattered. You only know the final outcome."

**This is Monte Carlo.**

The agent runs the complete episode, collects every reward, and only updates its knowledge after the episode is completely finished. It learns from the full picture but it waits — and in a long episode that waiting is expensive.

```
Episode:   step 1 → step 2 → step 3 → ... → step T → GOAL
Updates:   nothing   nothing   nothing         nothing   ← ALL updates here
```

---

### Scenario 2 — Feedback After Every Action = 1-step Sarsa

> "Your coach gives you feedback after every single action. You steer — feedback. You brake — feedback. You take a turn — feedback. The reaction is immediate. But here is the problem. You take a left turn at a junction and your coach immediately says bad. But three turns later it becomes clear that left turn was actually the correct route. The coach judged too early, with too little information. Feedback is fast but shallow."

**This is 1-step TD Sarsa.**

The agent updates its Q-table after every single step using only one real reward. It is fast but the signal is narrow. In a gridworld with 20 steps to the goal, 1-step Sarsa needs **20 episodes** just to carry the goal reward back to the starting position.

> The textbook calls this **the tyranny of the single time step.**

```
Episode:   step 1 → step 2 → step 3 → ... → step T → GOAL
Updates:      ↑         ↑         ↑                      ↑
           update    update    update                  update
           (only 1 reward each — narrow signal)
```

---

### Scenario 3 — Feedback After n Actions = n-step Sarsa

> "Your coach watches you make exactly n actions — say four actions — and then gives you feedback based on what they observed. Not one action, not the entire trip. Just four. Enough information to make a meaningful judgment. Not so much that you are waiting forever."

**This is n-step Sarsa. The sweet spot.**

The agent collects n real rewards, then updates. And the textbook on page 141 says exactly this:

> *"n-step methods free us from the tyranny of the single time step."*

```
Episode:   step 1 → step 2 → step 3 → step 4 → step 5 → ...
                                          ↑
                              update state (0,0) using
                              steps 1 to 4 (n=4 real rewards)
```

---

## 2. Limitations of the Two Extremes

These limitations directly motivate why n-step Sarsa was invented.

### Monte Carlo — Three Problems

| Problem | Explanation |
|---|---|
| **Too late** | No learning during the episode. If the episode is 200 steps long, you wait 200 steps before updating a single Q-value. |
| **High variance** | One unlucky early event — say the agent hits a trap by chance — changes **every Q-value** in the entire episode. |
| **Useless online** | If the episode never terminates, Monte Carlo learns nothing at all. |

### 1-step Sarsa — One Problem

**Too narrow.** Each update uses only one real reward. The goal signal travels backwards one step per episode.

In our 4×4 gridworld where the path is 6 steps long:

```
Episode 1:   Goal signal reaches (2,3)   ← 1 step from goal
Episode 2:   Goal signal reaches (1,3)   ← 2 steps from goal
Episode 3:   Goal signal reaches (0,3)   ← 3 steps from goal
Episode 4:   Goal signal reaches (0,2)   ← 4 steps from goal
Episode 5:   Goal signal reaches (0,1)   ← 5 steps from goal
Episode 6:   Goal signal reaches (0,0)   ← START — finally!
```

On a sparse reward environment this is very slow.

---

## 3. Why n-step Sarsa

n-step Sarsa was introduced specifically to fix **both** problems at the same time.

**It fixes the Monte Carlo problem** by updating during the episode — you do not wait for the episode to end. After every n steps an update happens. The agent is learning while it moves.

**It fixes the 1-step Sarsa problem** by using n real rewards instead of just one. In our gridworld with n=4, four Q-values get updated every time the agent reaches the goal, instead of just one. That is **four times more learning** from the same episode.

**n is a tunable dial:**

```
n = 1             →   exactly 1-step Sarsa
n = 4, 8, 16...   →   n-step Sarsa (sweet spot)
n = episode       →   exactly Monte Carlo
```

You choose where on the spectrum you want to be based on your environment.

---

## 4. The n-step Spectrum

```
1-step Sarsa          n-step Sarsa           Monte Carlo
    n = 1           n = 4, 8, 16...            n = ∞

Update after            Sweet spot            Wait for
every action                                episode end

High bias     ←─────────────────────────→  Low bias
Low variance  ←─────────────────────────→  High variance
```

### When does n-step win?

| Environment | n-step advantage | Why |
|---|---|---|
| Sparse reward (only at goal) | **Largest** | n credits per episode vs 1 for Sarsa |
| Windy Gridworld (long paths) | **Moderate** | Wind delays paths, n-step helps |
| Dense reward (-1 every step) | **Small** | Signal at every step — Sarsa sufficient |
| Cliff Walking (-100 penalty) | **Smallest** | Large n spreads cliff penalty backward |

---

## 5. Bias and Variance

### Where bias comes from

**Bias comes from bootstrapping** — using a Q-table estimate instead of real rewards.

```
1-step Sarsa:   G = R  +  γ · Q(S', A')
                           ─────────────
                           bootstrap (estimate)  ← HIGH bias

Monte Carlo:    G = R₁ + R₂ + ... + Rₜ
                    all real rewards             ← zero bias
```

### Where variance comes from

**Variance comes from accumulating many random rewards.**

Monte Carlo includes every reward from the full episode. One lucky or unlucky early event affects every Q-value update — HIGH variance.

1-step Sarsa uses only one real reward — LOW variance.

### The correct picture

| Algorithm | Bias | Variance |
|---|---|---|
| 1-step Sarsa | High | Low |
| n-step Sarsa (n=4) | Medium | Medium |
| Monte Carlo | Low | High |

> **More real rewards in G = lower bias, higher variance**
> **More bootstrapping in G = higher bias, lower variance**

---

## 6. Can We Reduce the Bias-Variance Tradeoff?

**Honest answer: No — not with n-step Sarsa alone.**

Bias and variance pull in opposite directions:

```
Reduce bias    →  use MORE real rewards  →  variance increases
Reduce variance →  use FEWER real rewards →  bias increases
```

This is a **fundamental mathematical property**, not a design flaw of n-step Sarsa.

### What n-step Sarsa actually does

It gives you a **dial to find the best balance** for your environment — not a way to eliminate the tradeoff.

```
n=1    →  high bias,   low variance   (1-step Sarsa)
n=4    →  medium bias, medium variance ← sweet spot
n=∞    →  no bias,     high variance   (Monte Carlo)
```

### What can genuinely reduce both

**Eligibility traces (Chapter 12 — TD(λ))** combines all n-step returns simultaneously using a weighted average. It gets closer to the ideal than any single fixed n.

**More experience** — with enough episodes both bias and variance naturally decrease as Q-values converge to their true values.

> n-step Sarsa reduces the tradeoff by letting you find the best balance point.
> To reduce the tradeoff itself — you need eligibility traces.

---

## 7. The Formula 

### The n-step Return (Equation 7.4)

```
G_{t:t+n} = R_{t+1} + γ·R_{t+2} + γ²·R_{t+3} + ... + γ^{n-1}·R_{t+n}  +  γⁿ·Q(S_{t+n}, A_{t+n})
             ───────────────────────────────────────────────────────────     ──────────────────────
                              n real rewards (discounted)                         bootstrap term
                                                                             (dropped if ep ended)
```

#### Every piece explained

| Piece | Meaning |
|---|---|
| `G_{t:t+n}` | The target value. Return starting at time t, looking n steps ahead. |
| `R_{t+1}` | First real reward. Discount = γ⁰ = 1. Full value. Came from the environment. |
| `γ · R_{t+2}` | Second real reward, discounted once. One step further = worth slightly less. |
| `γ^{n-1} · R_{t+n}` | The nth real reward. Last real reward before switching to estimate. |
| `γⁿ · Q(S_{t+n}, A_{t+n})` | **Bootstrap term.** Not a real reward — our Q-table guess of future value. Only added if episode still running at step t+n. |

#### The bootstrap condition

```
τ + n ≥ T   →   episode ended before n steps   →   NO bootstrap (drop term)
τ + n < T   →   episode still running           →   ADD bootstrap term
```


---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
  - Chapter 6 — Sarsa, Q-learning, Cliff Walking (Example 6.6)
  - Chapter 7 — n-step Bootstrapping, Equation 7.4 and 7.5
  - Figure 7.2 — n-step performance on 19-state random walk
  - p.141 — tyranny of the single time step

---

*Portfolio Task 1 — Group 16 | Reasoning and Decision Making under Uncertainty*
*Prof. Dr. Frank Deinzer | THWS Würzburg-Schweinfurt | SS 2026*