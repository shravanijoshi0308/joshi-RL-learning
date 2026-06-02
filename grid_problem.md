# Gridworld Comparison — Three Methods
## 1-step Sarsa vs Monte Carlo vs n-step Sarsa
### Same grid. Same path. Same rewards. Only the algorithm changes.

---

## The Setup — 4x4 Gridworld

```
+-------+-------+-------+-------+
|  S    |       |       |       |
| (0,0) | (0,1) | (0,2) | (0,3) |
+-------+-------+-------+-------+
|       |  TRAP |       |       |
| (1,0) | (1,1) | (1,2) | (1,3) |
+-------+-------+-------+-------+
|       |       |  TRAP |       |
| (2,0) | (2,1) | (2,2) | (2,3) |
+-------+-------+-------+-------+
|       |       |       |  G    |
| (3,0) | (3,1) | (3,2) | (3,3) |
+-------+-------+-------+-------+

S = Start  (0,0)
G = Goal   (3,3)
TRAP = -5 reward
```

### Fixed parameters for all three methods

| Parameter | Value | Reason |
|---|---|---|
| Alpha (learning rate) | 0.5 | Standard starting value |
| Gamma (discount) | 1.0 | All rewards worth equally |
| Epsilon | 0.1 | 10% random exploration |
| Grid size | 4 x 4 = 16 states | Small enough to solve by hand |
| Actions | Up, Down, Left, Right | 4 actions per state |

### Reward structure

| Event | Reward |
|---|---|
| Reaching Goal G | +10 |
| Stepping into Trap | -5 |
| Every other step | -1 |

---

## One Episode — The Path We Will Solve

We will trace the exact same episode through all three methods.

The agent takes this path from S to G:

```
Step 0:  Start at (0,0)
Step 1:  Move Right  →  reach (0,1)   reward = -1
Step 2:  Move Right  →  reach (0,2)   reward = -1
Step 3:  Move Right  →  reach (0,3)   reward = -1
Step 4:  Move Down   →  reach (1,3)   reward = -1
Step 5:  Move Down   →  reach (2,3)   reward = -1
Step 6:  Move Down   →  reach (3,3)   reward = +10  GOAL
```

The agent took 6 steps to reach the goal.

Rewards collected in order:

```
R1 = -1
R2 = -1
R3 = -1
R4 = -1
R5 = -1
R6 = +10
```

The Q-table starts completely at zero — the agent knows nothing yet.

```
All Q(s, a) = 0   for every state and every action
```

---

## Method 1 — 1-step Sarsa (n = 1)

The coach gives feedback after every single action.
After Step 1, the coach updates the score for Step 0.
After Step 2, the coach updates Step 1.
And so on.

Each update only uses ONE real reward then immediately relies on an estimate for everything beyond.

### How it updates — step by step

The update rule is:

```
Q(S, A)  <--  Q(S, A)  +  alpha  x  [ R  +  gamma x Q(S_next, A_next)  -  Q(S, A) ]
```

**After Step 1:**
```
Updating:   Q( (0,0), Right )
R           = -1
Q_next      = Q( (0,1), ? )  = 0   (Q-table is all zeros)

G  =  R  +  gamma x Q_next
   =  -1  +  1.0 x 0
   =  -1

Update:
Q( (0,0), Right )  <--  0  +  0.5 x [ -1  -  0 ]
                   <--  0  +  0.5 x (-1)
                   <--  -0.5
```

**After Step 2:**
```
Updating:   Q( (0,1), Right )
R           = -1
Q_next      = Q( (0,2), ? )  = 0

G  =  -1  +  1.0 x 0  =  -1

Q( (0,1), Right )  <--  0  +  0.5 x (-1 - 0)  =  -0.5
```

**After Step 3:**
```
Updating:   Q( (0,2), Right )
R           = -1
Q_next      = Q( (0,3), ? )  = 0

Q( (0,2), Right )  <--  0  +  0.5 x (-1 - 0)  =  -0.5
```

**After Step 4:**
```
Updating:   Q( (0,3), Down )
R           = -1
Q_next      = Q( (1,3), ? )  = 0

Q( (0,3), Down )  <--  0  +  0.5 x (-1 - 0)  =  -0.5
```

**After Step 5:**
```
Updating:   Q( (1,3), Down )
R           = -1
Q_next      = Q( (2,3), ? )  = 0

Q( (1,3), Down )  <--  0  +  0.5 x (-1 - 0)  =  -0.5
```

**After Step 6 — Goal reached:**
```
Updating:   Q( (2,3), Down )
R           = +10
Q_next      = Q( (3,3), ? )  = 0   (terminal state, Q = 0)

G  =  +10  +  1.0 x 0  =  +10

Q( (2,3), Down )  <--  0  +  0.5 x (10 - 0)  =  +5.0
```

### Q-table after Episode 1 — Method 1

```
State       Action      Q-value before    Q-value after
(0,0)       Right       0                 -0.5
(0,1)       Right       0                 -0.5
(0,2)       Right       0                 -0.5
(0,3)       Down        0                 -0.5
(1,3)       Down        0                 -0.5
(2,3)       Down        0                 +5.0
```

### What happened?

Only the state **one step before the goal** learned that the goal is nearby (+5.0).

Every other state only learned that its immediate next step gives -1.

The state at the START (0,0) has no idea the goal exists. It only knows moving right gives -1.

After Episode 1:
- States that learned goal is close: **1 out of 6**
- States that know nothing useful: **5 out of 6**

---

## Method 2 — Monte Carlo (n = infinity)



The coach waits until the entire trip is finished.
Then reviews every single action from start to finish.
Every state on the path gets updated using the FULL real return.

No estimates. No bootstrapping. Pure real experience.

### How it updates

The full episode return from any state is computed backwards.

```
Full return G  =  R1 + R2 + R3 + R4 + R5 + R6
               =  -1 + -1 + -1 + -1 + -1 + 10
               =  5
```

But each state is updated with G computed **from that state forward**, not the full total.

Working backwards from the goal:

**Step 6 — State (2,3), Action Down:**
```
G  =  R6  =  +10

Q( (2,3), Down )  <--  0  +  0.5 x (10 - 0)  =  +5.0
```

**Step 5 — State (1,3), Action Down:**
```
G  =  R5  +  gamma x R6
   =  -1  +  1.0 x 10
   =  +9

Q( (1,3), Down )  <--  0  +  0.5 x (9 - 0)  =  +4.5
```

**Step 4 — State (0,3), Action Down:**
```
G  =  R4  +  R5  +  R6
   =  -1  +  -1  +  10
   =  +8

Q( (0,3), Down )  <--  0  +  0.5 x (8 - 0)  =  +4.0
```

**Step 3 — State (0,2), Action Right:**
```
G  =  R3  +  R4  +  R5  +  R6
   =  -1  +  -1  +  -1  +  10
   =  +7

Q( (0,2), Right )  <--  0  +  0.5 x (7 - 0)  =  +3.5
```

**Step 2 — State (0,1), Action Right:**
```
G  =  R2  +  R3  +  R4  +  R5  +  R6
   =  -1  +  -1  +  -1  +  -1  +  10
   =  +6

Q( (0,1), Right )  <--  0  +  0.5 x (6 - 0)  =  +3.0
```

**Step 1 — State (0,0), Action Right:**
```
G  =  R1  +  R2  +  R3  +  R4  +  R5  +  R6
   =  -1  +  -1  +  -1  +  -1  +  -1  +  10
   =  +5

Q( (0,0), Right )  <--  0  +  0.5 x (5 - 0)  =  +2.5
```

### Q-table after Episode 1 — Method 2

```
State       Action      Q-value before    Q-value after
(0,0)       Right       0                 +2.5
(0,1)       Right       0                 +3.0
(0,2)       Right       0                 +3.5
(0,3)       Down        0                 +4.0
(1,3)       Down        0                 +4.5
(2,3)       Down        0                 +5.0
```

### What happened?

Every single state on the path learned something useful from just one episode.

The start state (0,0) already knows moving right is worth +2.5.

After Episode 1:
- States that learned goal is reachable: **6 out of 6**
- States that know nothing useful: **0 out of 6**

But remember — Monte Carlo had to wait for the **entire episode to finish** before making any of these updates. No learning happened during the episode itself.

---

## Method 3 — n-step Sarsa (n = 4)


The coach watches 4 actions then gives feedback.
After seeing Actions 1, 2, 3, 4 — the coach updates Action 1.
After seeing Actions 2, 3, 4, 5 — the coach updates Action 2.
And so on.

4 real rewards are used. Then one estimate is added at the bottom.

### The delay — tau = t - n + 1

With n = 4:

```
t = 0  -->  tau = 0 - 4 + 1 = -3   no update yet
t = 1  -->  tau = 1 - 4 + 1 = -2   no update yet
t = 2  -->  tau = 2 - 4 + 1 = -1   no update yet
t = 3  -->  tau = 3 - 4 + 1 =  0   UPDATE step 0 now
t = 4  -->  tau = 4 - 4 + 1 =  1   UPDATE step 1 now
t = 5  -->  tau = 5 - 4 + 1 =  2   UPDATE step 2 now
t = 6  -->  tau = 6 - 4 + 1 =  3   UPDATE step 3 now  (episode ends)
drain:  tau = 4   UPDATE step 4
drain:  tau = 5   UPDATE step 5
```

### How it updates — step by step

The update rule is:

```
G  =  R_{tau+1}  +  gamma x R_{tau+2}  +  gamma^2 x R_{tau+3}
   +  gamma^3 x R_{tau+4}  +  gamma^4 x Q(S_{tau+4}, A_{tau+4})
```

---

**At t=3 — Updating tau=0 — State (0,0), Action Right:**

Rewards from tau+1=1 to tau+n=4:
```
R1 = -1,   R2 = -1,   R3 = -1,   R4 = -1
Bootstrap from Step 4: Q( (1,3), Down ) = 0

G  =  -1  +  -1  +  -1  +  -1  +  1.0^4 x 0
   =  -4  +  0
   =  -4

Q( (0,0), Right )  <--  0  +  0.5 x (-4 - 0)  =  -2.0
```

Note: The goal reward has not been seen yet at this point.
This update is pessimistic — the agent only saw -1 rewards so far.

---

**At t=4 — Updating tau=1 — State (0,1), Action Right:**

Rewards from tau+1=2 to tau+n=5:
```
R2 = -1,   R3 = -1,   R4 = -1,   R5 = -1
Bootstrap from Step 5: Q( (2,3), Down ) = 0

G  =  -1  +  -1  +  -1  +  -1  +  0  =  -4

Q( (0,1), Right )  <--  0  +  0.5 x (-4 - 0)  =  -2.0
```

---

**At t=5 — Updating tau=2 — State (0,2), Action Right:**

Rewards from tau+1=3 to tau+n=6:
```
R3 = -1,   R4 = -1,   R5 = -1,   R6 = +10
Bootstrap: tau+n = 6 = T  so NO bootstrap (episode ended)

G  =  -1  +  -1  +  -1  +  10  =  +7

Q( (0,2), Right )  <--  0  +  0.5 x (7 - 0)  =  +3.5
```

This is the first update that includes the goal reward.

---

**At t=6 — Updating tau=3 — State (0,3), Action Down:**

Rewards from tau+1=4 to tau+n=7, but T=6 so cap at T:
```
R4 = -1,   R5 = -1,   R6 = +10
No bootstrap (episode ended at T=6)

G  =  -1  +  -1  +  10  =  +8

Q( (0,3), Down )  <--  0  +  0.5 x (8 - 0)  =  +4.0
```

---

**Drain — Updating tau=4 — State (1,3), Action Down:**

```
R5 = -1,   R6 = +10
No bootstrap

G  =  -1  +  10  =  +9

Q( (1,3), Down )  <--  0  +  0.5 x (9 - 0)  =  +4.5
```

---

**Drain — Updating tau=5 — State (2,3), Action Down:**

```
R6 = +10
No bootstrap

G  =  +10

Q( (2,3), Down )  <--  0  +  0.5 x (10 - 0)  =  +5.0
```

### Q-table after Episode 1 — Method 3

```
State       Action      Q-value before    Q-value after
(0,0)       Right       0                 -2.0
(0,1)       Right       0                 -2.0
(0,2)       Right       0                 +3.5
(0,3)       Down        0                 +4.0
(1,3)       Down        0                 +4.5
(2,3)       Down        0                 +5.0
```

### What happened?

6 out of 6 states updated — same as Monte Carlo.

But it happened **step by step during the episode**, not all at once at the end.

The first two states got a pessimistic update (-2.0) because the goal reward had not been seen yet when they were updated. This will self-correct in future episodes as Q-values propagate.

---

## Method 4 — Q-learning (off-policy)

Q-learning is similar to 1-step Sarsa with one important difference.

Sarsa updates using the action the agent **actually took** next.
Q-learning updates using the **best possible** action at the next state — even if the agent did not take it.

This makes Q-learning off-policy. It learns the optimal policy while behaving with epsilon-greedy.

### The key difference from Sarsa

```
Sarsa update uses:      R  +  gamma x Q(S_next, A_next)
                                        ────────────────
                                        the action actually taken (epsilon-greedy)

Q-learning update uses: R  +  gamma x max_a Q(S_next, a)
                                        ──────────────────
                                        the BEST action at S_next (greedy)
                                        regardless of what was actually done
```

### The update rule

```
Q(S, A)  <--  Q(S, A)  +  alpha x [ R  +  gamma x max_a Q(S_next, a)  -  Q(S, A) ]
                                             ──────────────────────────
                                             greedy max — not what agent did
```

### How it updates — step by step

Same path as before. Same rewards.

```
R1=-1  R2=-1  R3=-1  R4=-1  R5=-1  R6=+10
```

Because the Q-table starts at zero, `max_a Q(S_next, a) = 0` for every state at the start.
This means Q-learning gives the same numbers as 1-step Sarsa in Episode 1 on a blank Q-table.

**After Step 1:**
```
Updating:   Q( (0,0), Right )
R           = -1
max_a Q( (0,1), a )  =  max(0, 0, 0, 0)  =  0   (all zeros)

G  =  -1  +  1.0 x 0  =  -1

Q( (0,0), Right )  <--  0  +  0.5 x (-1 - 0)  =  -0.5
```

**After Step 2:**
```
Updating:   Q( (0,1), Right )
max_a Q( (0,2), a )  =  0

Q( (0,1), Right )  <--  0  +  0.5 x (-1 - 0)  =  -0.5
```

**After Step 3:**
```
Updating:   Q( (0,2), Right )
max_a Q( (0,3), a )  =  0

Q( (0,2), Right )  <--  0  +  0.5 x (-1 - 0)  =  -0.5
```

**After Step 4:**
```
Updating:   Q( (0,3), Down )
max_a Q( (1,3), a )  =  0

Q( (0,3), Down )  <--  0  +  0.5 x (-1 - 0)  =  -0.5
```

**After Step 5:**
```
Updating:   Q( (1,3), Down )
max_a Q( (2,3), a )  =  0

Q( (1,3), Down )  <--  0  +  0.5 x (-1 - 0)  =  -0.5
```

**After Step 6 — Goal reached:**
```
Updating:   Q( (2,3), Down )
R           = +10
max_a Q( (3,3), a )  =  0   (terminal state)

G  =  +10  +  1.0 x 0  =  +10

Q( (2,3), Down )  <--  0  +  0.5 x (10 - 0)  =  +5.0
```

### Q-table after Episode 1 — Method 4

```
State       Action      Q-value before    Q-value after
(0,0)       Right       0                 -0.5
(0,1)       Right       0                 -0.5
(0,2)       Right       0                 -0.5
(0,3)       Down        0                 -0.5
(1,3)       Down        0                 -0.5
(2,3)       Down        0                 +5.0
```

### What happened?

The numbers are identical to 1-step Sarsa for Episode 1.

This is because the Q-table was all zeros — `max_a Q(S_next, a) = 0` is the same as `Q(S_next, A_next) = 0`.

The difference between Q-learning and Sarsa only shows up in **later episodes** when Q-values are no longer zero.

### Where Q-learning behaves differently — Episode 2 example

Suppose after Episode 1, Q( (2,3), Down ) = +5.0 and everything else is still small.

Now the agent is at (1,3) and takes action Down to (2,3):

```
Sarsa update for (1,3):
    A_next  =  epsilon-greedy at (2,3)
            =  could be any action (exploring)
    If agent explores and picks Left:  Q( (2,3), Left ) = -0.5
    G  =  -1  +  1.0 x (-0.5)  =  -1.5
    Q( (1,3), Down ) stays low

Q-learning update for (1,3):
    max_a Q( (2,3), a )  =  max(-0.5, -0.5, -0.5, +5.0)  =  +5.0  (always Down)
    G  =  -1  +  1.0 x 5.0  =  +4.0
    Q( (1,3), Down )  <--  0  +  0.5 x (4.0 - 0)  =  +2.0
```

Q-learning propagates the goal signal faster because it always uses the best known next action — even during exploration. Sarsa may use a random exploratory action and get a weaker update.

---

## Side by Side Comparison — All Four Methods

### Q-values after ONE episode

```
State    Action   1-step Sarsa   Q-learning   Monte Carlo   n-step (n=4)
(0,0)    Right       -0.5          -0.5          +2.5           -2.0
(0,1)    Right       -0.5          -0.5          +3.0           -2.0
(0,2)    Right       -0.5          -0.5          +3.5           +3.5
(0,3)    Down        -0.5          -0.5          +4.0           +4.0
(1,3)    Down        -0.5          -0.5          +4.5           +4.5
(2,3)    Down        +5.0          +5.0          +5.0           +5.0
```

### States that learned something useful after Episode 1

```
1-step Sarsa:    1  out of 6    (only the last step before goal)
Q-learning:      1  out of 6    (same as Sarsa — difference shows later)
Monte Carlo:     6  out of 6    (all steps — but had to wait until end)
n-step (n=4):    4  out of 6    (steps 3-6 — updates during episode)
```

### Updates made DURING the episode vs AFTER

```
1-step Sarsa:    6 updates during episode   0 after
Q-learning:      6 updates during episode   0 after
Monte Carlo:     0 updates during episode   6 after
n-step (n=4):    4 updates during episode   2 after (drain phase)
```

---

## The Key Comparison Table

| Property | 1-step Sarsa | Q-learning | Monte Carlo | n-step Sarsa (n=4) |
|---|---|---|---|---|
| Policy type | On-policy | Off-policy | On-policy | On-policy |
| Updates per episode | 6 | 6 | 6 | 6 |
| Useful updates Ep 1 | 1 | 1 | 6 | 4 |
| Updates during episode | Yes | Yes | No | Yes |
| Real rewards used | 1 | 1 | All | 4 |
| Bootstrap target | Q(S', A') actual | max Q(S', a) greedy | None | Q(S_n, A_n) after n |
| Start state after Ep 1 | -0.5 | -0.5 | +2.5 | -2.0 (corrects) |
| Converges to | Safe policy | Optimal policy | Unbiased policy | Near-optimal |
| Risk near cliffs | Low | Higher during training | Low | Low |
| Episodes to converge | Slow | Moderate | Moderate | Fast |

---

## What This Tells Us

**1-step Sarsa** is too narrow. Credit travels 1 step per episode. On a 6-step path it takes 6 episodes just to get useful values back to the start.

**Q-learning** has the same first-episode performance as Sarsa but improves faster in later episodes because it always bootstraps from the best known action — not a random exploratory one. However near the cliff in Cliff Walking it can be risky during training.

**Monte Carlo** learns the most from Episode 1 but must wait until the episode ends. High variance from accumulating all rewards means it needs many episodes to stabilise.

**n-step Sarsa (n=4)** is the best balance. It updates 4 states usefully during the episode. The first two states get corrected in the next episode. It is on-policy so it is safe near cliffs. And it converges faster than 1-step Sarsa on sparse reward environments.

---

## Visual Summary

```
After 1 episode — Q-values on the path S to G

1-step Sarsa:
  S     →     →     →     →     →     G
[-0.5][-0.5][-0.5][-0.5][-0.5][+5.0]
  Only the last step knows the goal.

Q-learning (Episode 1 same as Sarsa):
  S     →     →     →     →     →     G
[-0.5][-0.5][-0.5][-0.5][-0.5][+5.0]
  Same as Sarsa now. Pulls ahead in Episode 2+.

Monte Carlo:
  S     →     →     →     →     →     G
[+2.5][+3.0][+3.5][+4.0][+4.5][+5.0]
  Every step knows. But waited until episode ended.

n-step Sarsa (n=4):
  S     →     →     →     →     →     G
[-2.0][-2.0][+3.5][+4.0][+4.5][+5.0]
  Most steps know. Updated during episode.
  First two correct themselves in Episode 2.
```

---

## Conclusion — Which Method Wins on This Gridworld?

For the 4x4 sparse gridworld the ranking is:

```
Rank 1:  n-step Sarsa (n=4)
         Best balance of speed, safety, and online learning

Rank 2:  Monte Carlo
         Most information per episode but must wait and has high variance

Rank 3:  Q-learning
         Matches Sarsa at first but pulls ahead in later episodes
         Slightly risky near traps due to off-policy greedy target

Rank 4:  1-step Sarsa
         Slowest credit propagation on sparse reward environment
```

**n-step Sarsa's advantage is largest here because:**
- The reward is only at the goal (sparse)
- Each episode that reaches the goal is very valuable
- n-step extracts 4x more learning from each such episode than 1-step Sarsa
- It updates online unlike Monte Carlo

---

## Reference

Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
MIT Press. Chapter 6: TD Learning. Chapter 7: n-step Bootstrapping, Section 7.2.