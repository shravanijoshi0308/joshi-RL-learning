# Gridworld Mathematical Comparison
### Monte Carlo | 1-step Sarsa | n-step Sarsa
**Environment 1: Standard 4×4 Gridworld | Environment 2: Windy 4×4 Gridworld**

> α = 0.5 | γ = 1.0 | ε = 0.1 | n = 4

---

## Table of Contents

1. [Part 1 — Standard 4×4 Gridworld](#part-1--standard-4×4-gridworld)
   - [Environment Setup](#11--environment-setup)
   - [The Path We Solve](#12--the-path-we-solve)
   - [Method 1 — Monte Carlo](#13--method-1--monte-carlo)
   - [Method 2 — 1-step Sarsa](#14--method-2--1-step-sarsa)
   - [Method 3 — n-step Sarsa](#15--method-3--n-step-sarsa-n--4)
   - [Part 1 Comparison Table](#16--part-1-comparison--q-values-after-episode-1)
2. [Part 2 — Windy 4×4 Gridworld](#part-2--windy-4×4-gridworld)
   - [Environment Setup](#21--environment-setup)
   - [How Wind Changes the Path](#22--how-wind-changes-the-path)
   - [The Path We Solve](#23--the-path-we-will-solve)
   - [Method 1 — Monte Carlo on Windy](#24--method-1--monte-carlo-on-windy-gridworld)
   - [Method 2 — 1-step Sarsa on Windy](#25--method-2--1-step-sarsa-on-windy-gridworld)
   - [Method 3 — n-step Sarsa on Windy](#26--method-3--n-step-sarsa-n4-on-windy-gridworld)
   - [Part 2 Comparison Table](#27--part-2-comparison--q-values-after-episode-1)
3. [Part 3 — Master Comparison Table](#part-3--master-comparison-table)
   - [Q-values Both Environments](#31--q-values-after-episode-1--both-environments)
   - [Algorithm Feature Comparison](#32--algorithm-feature-comparison)
   - [When Each Algorithm Wins](#33--when-each-algorithm-wins)
   - [Core Answer](#34--the-core-question)

---

# Part 1 — Standard 4×4 Gridworld

## 1.1  Environment Setup

The standard 4×4 gridworld is the simplest test environment. It has a clear start, a clear goal, two traps to avoid, and a sparse reward (only at the goal). This makes it the ideal environment for showing n-step Sarsa's advantage.

```
+-------+-------+-------+-------+
|  S    |       |       |       |    Row 0
| (0,0) | (0,1) | (0,2) | (0,3) |
+-------+-------+-------+-------+
|       |  TRAP |       |       |    Row 1
| (1,0) | (1,1) | (1,2) | (1,3) |
+-------+-------+-------+-------+
|       |       |  TRAP |       |    Row 2
| (2,0) | (2,1) | (2,2) | (2,3) |
+-------+-------+-------+-------+
|       |       |       |  G    |    Row 3
| (3,0) | (3,1) | (3,2) | (3,3) |
+-------+-------+-------+-------+
```

| Property | Value | Explanation |
|---|---|---|
| Start S | (0,0) | Top-left corner |
| Goal G | (3,3) | Bottom-right corner |
| Traps | (1,1) and (2,2) | Step penalty, no reset |
| Actions | Up Down Left Right | 4 actions per state |
| Goal reward | +10 | Episode ends |
| Trap penalty | −5 | Episode continues |
| Step penalty | −1 | Every non-goal step |
| Reward type | **SPARSE** | Only +10 at goal — n-step advantage is LARGEST here |

---

## 1.2  The Path We Solve

We trace the same 6-step path through all three algorithms. Same starting conditions, same sequence, same rewards. **Only the algorithm changes.**

```
Step 1:  (0,0) → Right → (0,1)   reward = -1
Step 2:  (0,1) → Right → (0,2)   reward = -1
Step 3:  (0,2) → Right → (0,3)   reward = -1
Step 4:  (0,3) → Down  → (1,3)   reward = -1
Step 5:  (1,3) → Down  → (2,3)   reward = -1
Step 6:  (2,3) → Down  → (3,3)   reward = +10  ← GOAL
```

```
R1=-1   R2=-1   R3=-1   R4=-1   R5=-1   R6=+10
```

Q-table at the start of Episode 1: **all zeros**. The agent knows nothing yet.

---

## 1.3  Method 1 — Monte Carlo

**How it works:** Run the full episode first. Collect all 6 rewards. Then update every state **backwards** from the goal using the full real return G. No updates happen during the episode.

**Formula:**
```
G  =  R_{t+1} + γ·R_{t+2} + ... + γ^{T-t-1}·R_T
Q(S,A)  ←  Q(S,A)  +  α × [ G  −  Q(S,A) ]
```

**Step-by-step calculation (working backwards from goal):**

γ = 1.0 so every reward counts equally regardless of when it happened.

**State (2,3) — Action Down — 1 step before goal:**
```
G  =  R6  =  +10
Q( (2,3), Down )  <--  0  +  0.5 × (10 - 0)  =  +5.0
```

**State (1,3) — Action Down — 2 steps before goal:**
```
G  =  R5 + R6  =  -1 + 10  =  +9
Q( (1,3), Down )  <--  0  +  0.5 × (9 - 0)   =  +4.5
```

**State (0,3) — Action Down — 3 steps before goal:**
```
G  =  R4 + R5 + R6  =  -1 + -1 + 10  =  +8
Q( (0,3), Down )  <--  0  +  0.5 × (8 - 0)   =  +4.0
```

**State (0,2) — Action Right — 4 steps before goal:**
```
G  =  R3 + R4 + R5 + R6  =  -1-1-1+10  =  +7
Q( (0,2), Right ) <--  0  +  0.5 × (7 - 0)   =  +3.5
```

**State (0,1) — Action Right — 5 steps before goal:**
```
G  =  R2+R3+R4+R5+R6  =  -1-1-1-1+10  =  +6
Q( (0,1), Right ) <--  0  +  0.5 × (6 - 0)   =  +3.0
```

**State (0,0) — Action Right — 6 steps before goal (START):**
```
G  =  R1+R2+R3+R4+R5+R6  =  -1-1-1-1-1+10  =  +5
Q( (0,0), Right ) <--  0  +  0.5 × (5 - 0)   =  +2.5
```

> **Monte Carlo Result:** All 6 states updated with positive Q-values. The start state (0,0) immediately knows going Right is worth +2.5. Every state learned something useful from just one episode.

---

## 1.4  Method 2 — 1-step Sarsa

**How it works:** Update Q after every single step using one real reward plus a bootstrap estimate. The update happens immediately — no waiting.

**Formula:**
```
G  =  R  +  γ · Q(S_next, A_next)
Q(S,A)  ←  Q(S,A)  +  α × [ G  −  Q(S,A) ]
```

**Step 1 — at (0,0), takes Right, arrives at (0,1):**
```
R = -1
Next action at (0,1): Right  (epsilon-greedy, all zeros)
Q_next  =  Q( (0,1), Right )  =  0
G       =  -1  +  1.0 × 0     =  -1
Q( (0,0), Right )  <--  0  +  0.5 × (-1 - 0)  =  -0.5
```

**Steps 2–5 — all identical (Q_next = 0 everywhere):**
```
Each step:  G = -1 + 0 = -1
Q(state, action) <-- 0 + 0.5 × (-1) = -0.5
States (0,1)(0,2)(0,3)(1,3) all get Q = -0.5
```

**Step 6 — at (2,3), takes Down, arrives at GOAL (3,3):**
```
R = +10
Terminal state — Q(goal) = 0 by definition
G  =  +10  +  1.0 × 0  =  +10
Q( (2,3), Down )  <--  0  +  0.5 × (10 - 0)  =  +5.0
```

> **1-step Sarsa Result:** Only 1 state learned the goal is reachable (+5.0). Every other state got Q = −0.5. The start state (0,0) has **no idea the goal exists** after Episode 1.

---

## 1.5  Method 3 — n-step Sarsa (n = 4)

**How it works:** Collect 4 real rewards, then update the state from 4 steps ago. **tau = t − n + 1** is the state being updated. This allows 4 real rewards to contribute to each update instead of just 1.

**Formula:**
```
G_{τ:τ+4}  =  R_{τ+1} + γR_{τ+2} + γ²R_{τ+3} + γ³R_{τ+4}  +  γ⁴·Q(S_{τ+4}, A_{τ+4})
Q(S_τ, A_τ)  ←  Q(S_τ, A_τ)  +  α × [ G  −  Q(S_τ, A_τ) ]
```

**The delay table (n = 4):**

| Time t | τ = t−4+1 | Agent moves to | Update which state? |
|---|---|---|---|
| 0 | -3 | (0,1) R=-1 | No update yet (τ < 0) |
| 1 | -2 | (0,2) R=-1 | No update yet (τ < 0) |
| 2 | -1 | (0,3) R=-1 | No update yet (τ < 0) |
| 3 | 0 | (1,3) R=-1 | **UPDATE (0,0) → Right** |
| 4 | 1 | (2,3) R=-1 | **UPDATE (0,1) → Right** |
| 5 | 2 | (3,3) R=+10 | **UPDATE (0,2) → Right** (T=6) |
| drain | 3 | — | **UPDATE (0,3) → Down** |
| drain | 4 | — | **UPDATE (1,3) → Down** |
| drain | 5 | — | **UPDATE (2,3) → Down** |

**τ=0, t=3 — Update (0,0) Action Right:**
```
Rewards: R1=-1  R2=-1  R3=-1  R4=-1
Bootstrap: Q( (1,3), Down ) = 0   (τ+4=4 < T=6, include bootstrap)
G  =  -1 + -1 + -1 + -1 + 0  =  -4
Q( (0,0), Right )  <--  0  +  0.5 × (-4)  =  -2.0
Note: goal not seen yet — pessimistic but corrects in Episode 2
```

**τ=1, t=4 — Update (0,1) Action Right:**
```
Rewards: R2=-1  R3=-1  R4=-1  R5=-1
Bootstrap: Q( (2,3), Down ) = 0
G  =  -4
Q( (0,1), Right )  <--  -2.0
```

**τ=2, t=5 — Update (0,2) Action Right:**
```
Rewards: R3=-1  R4=-1  R5=-1  R6=+10
τ+4 = 6 = T  →  NO bootstrap (episode ended)
G  =  -1 + -1 + -1 + +10  =  +7
Q( (0,2), Right )  <--  0  +  0.5 × 7  =  +3.5   ← First update with goal!
```

**τ=3 (drain) — Update (0,3) Action Down:**
```
G  =  -1 + -1 + 10  =  +8
Q( (0,3), Down )  <--  +4.0
```

**τ=4 (drain) — Update (1,3) Action Down:**
```
G  =  -1 + 10  =  +9
Q( (1,3), Down )  <--  +4.5
```

**τ=5 (drain) — Update (2,3) Action Down:**
```
G  =  +10
Q( (2,3), Down )  <--  +5.0
```

> **n-step Result:** 4 out of 6 states got positive Q-values in one episode. The first two states got −2.0 because the goal reward had not been seen yet when they were updated. **This self-corrects in Episode 2** as bootstrap values improve.

---

## 1.6  Part 1 Comparison — Q-values After Episode 1

| State | Action | Q before | Monte Carlo | 1-step Sarsa | n-step (n=4) |
|---|---|---|---|---|---|
| (0,0) | Right | 0 | **+2.5** | −0.5 | −2.0 |
| (0,1) | Right | 0 | **+3.0** | −0.5 | −2.0 |
| (0,2) | Right | 0 | **+3.5** | −0.5 | **+3.5** |
| (0,3) | Down  | 0 | **+4.0** | −0.5 | **+4.0** |
| (1,3) | Down  | 0 | **+4.5** | −0.5 | **+4.5** |
| (2,3) | Down  | 0 | **+5.0** | **+5.0** | **+5.0** |

| Metric | Monte Carlo | 1-step Sarsa | n-step Sarsa (n=4) |
|---|---|---|---|
| States with positive Q | **6 / 6** | **1 / 6** | **4 / 6** |
| Start state learned goal? | Yes (+2.5) | No (−0.5) | Partial (−2.0, corrects Ep2) |
| Updates during episode | No — all after | Yes — each step | Yes — delayed n steps |
| Goal signal reaches start? | **Yes — Episode 1** | No — needs 6+ episodes | Partial — corrects Ep2 |
| Conclusion | Most learning Ep1 | Slowest on sparse | **Best balance** |

---

# Part 2 — Windy 4×4 Gridworld

## 2.1  Environment Setup

The windy gridworld adds a wind force. Wind in certain columns pushes the agent **upward** every step. The agent must account for wind to reach the goal.

```
+-------+-------+-------+-------+
|       |       |       |  G    |    Row 0  (wind pushes agent UP)
| (0,0) | (0,1) | (0,2) | (0,3) |
+-------+-------+-------+-------+
|       |       |       |       |    Row 1
| (1,0) | (1,1) | (1,2) | (1,3) |
+-------+-------+-------+-------+
|       |       |       |       |    Row 2
| (2,0) | (2,1) | (2,2) | (2,3) |
+-------+-------+-------+-------+
|  S    |       |       |       |    Row 3
| (3,0) | (3,1) | (3,2) | (3,3) |
+-------+-------+-------+-------+
Wind:     0       1       1       0
```

| Property | Value | Explanation |
|---|---|---|
| Start S | (3,0) | Bottom-left corner |
| Goal G | (0,3) | Top-right corner |
| Wind | cols 1,2 push up 1 row | Columns 1 and 2 have wind strength 1 |
| Effect of wind | nr = row + dr - wind | Agent moves intended direction MINUS wind push |
| Goal reward | +10 | Episode ends |
| Step penalty | −1 | Every non-goal step |
| Reward type | **MODERATE** | Steps give −1 so 1-step TD gets some signal too |

---

## 2.2  How Wind Changes the Path

- Moving Right through column 1: agent goes Right AND 1 step Up (wind push)
- Moving Right through column 2: agent goes Right AND 1 step Up (wind push)
- The agent can **use the wind to its advantage** — reaching the goal faster

---

## 2.3  The Path We Will Solve

```
Step 1:  (3,0) → Right → col 1 has wind
         Actual: Right + wind(1) UP = (2,1)   reward = -1

Step 2:  (2,1) → Right → col 2 has wind
         Actual: Right + wind = (1,2)          reward = -1

Step 3:  (1,2) → Right → col 3 no wind
         (1,2) + Right = (1,3)                 reward = -1

Step 4:  (1,3) → Up → col 3 no wind
         (1,3) + Up = (0,3) = GOAL             reward = +10
```

A **4-step episode** — wind helped the agent reach the goal in fewer steps!

```
R1=-1   R2=-1   R3=-1   R4=+10
```

---

## 2.4  Method 1 — Monte Carlo on Windy Gridworld

Working backwards from goal:

**State (1,3) — 1 step before goal:**
```
G  =  R4  =  +10
Q( (1,3), Up )  <--  0  +  0.5 × 10  =  +5.0
```

**State (1,2) — 2 steps before goal:**
```
G  =  R3 + R4  =  -1 + 10  =  +9
Q( (1,2), Right )  <--  +4.5
```

**State (2,1) — 3 steps before goal:**
```
G  =  R2 + R3 + R4  =  -1 + -1 + 10  =  +8
Q( (2,1), Right )  <--  +4.0
```

**State (3,0) — 4 steps before goal (START):**
```
G  =  R1 + R2 + R3 + R4  =  -1 + -1 + -1 + 10  =  +7
Q( (3,0), Right )  <--  +3.5
```

> **Monte Carlo on Windy:** All 4 states updated with positive Q-values. Shorter episode = less variance.

---

## 2.5  Method 2 — 1-step Sarsa on Windy Gridworld

**Step 1 — at (3,0), takes Right, wind pushes to (2,1):**
```
R = -1,  Q_next = Q( (2,1), Right ) = 0
G  =  -1 + 0  =  -1
Q( (3,0), Right )  <--  -0.5
```

**Steps 2–3 — similar (Q_next = 0):**
```
Q( (2,1), Right )  <--  -0.5
Q( (1,2), Right )  <--  -0.5
```

**Step 4 — at (1,3), takes Up, arrives at GOAL:**
```
R = +10,  terminal Q = 0
G  =  +10
Q( (1,3), Up )  <--  +5.0
```

> **1-step Sarsa on Windy:** Only the last state (1,3) learned the goal. Wind means paths are shorter (4 steps instead of 6), so credit propagates faster than on a longer grid.

---

## 2.6  Method 3 — n-step Sarsa (n=4) on Windy Gridworld

**The delay table (n=4, episode length=4):**

| Time t | τ = t−4+1 | Agent moves to | Update which state? |
|---|---|---|---|
| 0 | -3 | (2,1) R=-1 | No update yet |
| 1 | -2 | (1,2) R=-1 | No update yet |
| 2 | -1 | (1,3) R=-1 | No update yet |
| 3 | 0 | (0,3) R=+10 GOAL (T=4) | **UPDATE (3,0) — τ+4=4=T, no bootstrap** |
| drain | 1 | — | **UPDATE (2,1)** |
| drain | 2 | — | **UPDATE (1,2)** |
| drain | 3 | — | **UPDATE (1,3)** |

> **Key insight:** Episode ends at T=4. With n=4, τ+4 ≥ T for ALL tau → **NO bootstrap added for any update.** n-step Sarsa behaves exactly like Monte Carlo for this short episode.

**τ=0 — Update (3,0):**
```
Rewards: R1=-1  R2=-1  R3=-1  R4=+10   τ+4=4=T → no bootstrap
G  =  -1 + -1 + -1 + 10  =  +7
Q( (3,0), Right )  <--  +3.5
```

**τ=1 — Update (2,1):**
```
G  =  -1 + -1 + 10  =  +8    →    Q  <--  +4.0
```

**τ=2 — Update (1,2):**
```
G  =  -1 + 10  =  +9          →    Q  <--  +4.5
```

**τ=3 — Update (1,3):**
```
G  =  +10                      →    Q  <--  +5.0
```

> **n-step on Windy:** ALL 4 states get positive Q-values — **same as Monte Carlo.** When n ≥ episode length, n-step equals MC but updates online.

---

## 2.7  Part 2 Comparison — Q-values After Episode 1

| State | Action | Monte Carlo | 1-step Sarsa | n-step (n=4) |
|---|---|---|---|---|
| (3,0) | Right | **+3.5** | −0.5 | **+3.5** |
| (2,1) | Right | **+4.0** | −0.5 | **+4.0** |
| (1,2) | Right | **+4.5** | −0.5 | **+4.5** |
| (1,3) | Up    | **+5.0** | **+5.0** | **+5.0** |

| Metric | Monte Carlo | 1-step Sarsa | n-step Sarsa (n=4) |
|---|---|---|---|
| States with positive Q | **4 / 4** | **1 / 4** | **4 / 4** |
| Start state learned goal? | Yes (+3.5) | No (−0.5) | **Yes (+3.5) — same as MC!** |
| n ≥ episode length? | n/a | n/a | Yes (4 ≥ 4) → equals MC |
| Conclusion | Excellent Ep1 | Slowest | **Matches MC on short episodes** |

---

# Part 3 — Master Comparison Table

## 3.1  Q-values After Episode 1 — Both Environments

### 4×4 Standard Gridworld

| State | MC | 1-step Sarsa | n-step (n=4) | Winner |
|---|---|---|---|---|
| (0,0) Right | +2.5 | −0.5 | −2.0 | Monte Carlo |
| (0,1) Right | +3.0 | −0.5 | −2.0 | Monte Carlo |
| (0,2) Right | +3.5 | −0.5 | +3.5 | MC = n-step |
| (0,3) Down  | +4.0 | −0.5 | +4.0 | MC = n-step |
| (1,3) Down  | +4.5 | −0.5 | +4.5 | MC = n-step |
| (2,3) Down  | +5.0 | +5.0 | +5.0 | All equal |
| **Useful updates** | **6/6** | **1/6** | **4/6** | MC > n-step > Sarsa |

### 4×4 Windy Gridworld

| State | MC | 1-step Sarsa | n-step (n=4) | Winner |
|---|---|---|---|---|
| (3,0) Right | +3.5 | −0.5 | +3.5 | MC = n-step |
| (2,1) Right | +4.0 | −0.5 | +4.0 | MC = n-step |
| (1,2) Right | +4.5 | −0.5 | +4.5 | MC = n-step |
| (1,3) Up    | +5.0 | +5.0 | +5.0 | All equal |
| **Useful updates** | **4/4** | **1/4** | **4/4** | MC = n-step > Sarsa |

---

## 3.2  Algorithm Feature Comparison

| Feature | Monte Carlo | 1-step Sarsa | n-step Sarsa (n=4) |
|---|---|---|---|
| Policy type | On-policy | On-policy | On-policy |
| Update timing | After episode ends | Every step (immediate) | Every step (n-step delay) |
| Bootstrap | None — pure real rewards | After 1 reward | After n rewards |
| Rewards per update | All (full return) | 1 | n = 4 |
| Bias | Low | **High** | Medium |
| Variance | **High** | Low | Medium |
| Works online? | **No — must wait** | Yes | Yes |
| 4×4 useful updates Ep1 | **6/6** | 1/6 | 4/6 |
| Windy useful updates Ep1 | **4/4** | 1/4 | **4/4** |
| Credit propagation speed | Full episode in one pass | 1 step per episode | n steps per episode |
| On sparse rewards | Best single-episode | **Worst** | **Best online** |
| On windy (short episodes) | Best | Worst | Equal to MC (n≥len) |
| Memory required | Full episode | O(1) | O(n) |

---

## 3.3  When Each Algorithm Wins

| Situation | Best Algorithm | Why |
|---|---|---|
| Sparse reward (goal only) | **n-step Sarsa** | Credits n states per episode vs 1. Online — no waiting. |
| Dense reward (every step) | **1-step Sarsa** | Signal at every step — no credit delay problem. |
| Short episodes (len ≤ n) | **n-step = Monte Carlo** | When n ≥ episode length, n-step equals MC but updates online. |
| Long episodes | **n-step Sarsa** | MC has very high variance. n-step limits it via bootstrap. |
| Learning accuracy (unbiased) | **Monte Carlo** | No bootstrap = no bias from estimated Q-values. |
| Online learning (can't wait) | **n-step or Sarsa** | MC cannot update until episode ends. |
| Windy environment | **n-step Sarsa** | Wind shortens paths, n-step often equals MC but works online. |
| Near cliffs / dense penalties | **1-step Sarsa** | On-policy + immediate updates = safer near large penalties. |

---

## 3.4  The Core Question

> **Which gridworld shows the BIGGEST n-step Sarsa advantage?**

**Answer: The 4×4 Standard Gridworld.**

Reward is sparse (only at goal), episode is long (6 steps), and n=4 updates **4 states per episode** vs Sarsa's 1.

The Windy Gridworld shows n-step advantage is **equal to Monte Carlo** when n ≥ episode length — but n-step is still better than 1-step Sarsa.

| Gridworld | n-step vs 1-step | n-step vs MC | Why |
|---|---|---|---|
| 4×4 Standard | **LARGE**: 4/6 vs 1/6 useful updates | MC wins Ep1 but n-step better online | Sparse reward, long path |
| 4×4 Windy | **LARGE**: 4/4 vs 1/4 useful updates | Equal (n ≥ episode length) | Short episodes, n-step = MC |

---

*Sutton & Barto (2018) — Reinforcement Learning (2nd Ed.) — Chapters 6 & 7*

*Portfolio Task 1 — Group 16 | Prof. Dr. Frank Deinzer | THWS Würzburg-Schweinfurt | SS 2026*