# Step by Step Walkthrough — Four Methods on 4x4 Gridworld
## Monte Carlo | 1-step Sarsa | Q-learning | n-step Sarsa

---

## The Grid

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
TRAP at (1,1) and (2,2)  =  -5 reward, no reset
```

---

## Fixed Settings for All Four Methods

| Setting | Value |
|---|---|
| Alpha (learning rate) | 0.5 |
| Gamma (discount) | 1.0 |
| n (for n-step Sarsa) | 4 |
| Starting Q-values | 0 for everything |

---

## The Episode We Will Solve

The agent takes this exact path:

```
Step 0   Start at (0,0)
Step 1   Move Right   reach (0,1)   reward = -1
Step 2   Move Right   reach (0,2)   reward = -1
Step 3   Move Right   reach (0,3)   reward = -1
Step 4   Move Down    reach (1,3)   reward = -1
Step 5   Move Down    reach (2,3)   reward = -1
Step 6   Move Down    reach (3,3)   reward = +10   GOAL
```

Rewards collected:

```
R1 = -1
R2 = -1
R3 = -1
R4 = -1
R5 = -1
R6 = +10
```

Q-table before the episode — everything is zero:

```
State      Up    Down   Left   Right
(0,0)       0      0      0      0
(0,1)       0      0      0      0
(0,2)       0      0      0      0
(0,3)       0      0      0      0
(1,3)       0      0      0      0
(2,3)       0      0      0      0
(3,3)       0      0      0      0   <- goal, always 0
```

---
---

# METHOD 1 — Monte Carlo

---

## What Monte Carlo Does

The agent runs the complete episode first.
Then it goes back and updates every state it visited.
No updates happen during the episode.
Updates happen only after the episode ends.

---

## Step 1 — Run the full episode, collect all rewards

```
(0,0) --R1=-1--> (0,1) --R2=-1--> (0,2) --R3=-1-->
(0,3) --R4=-1--> (1,3) --R5=-1--> (2,3) --R6=+10--> (3,3) GOAL
```

Episode is now finished. All rewards are stored.

---

## Step 2 — Compute G (return) for each state

Monte Carlo works BACKWARDS from the goal.

G means: total reward from this state to the end of the episode.

```
Start from the goal and work backwards.
```

**State (2,3) — one step before goal:**
```
G  =  R6
   =  +10
```

**State (1,3) — two steps before goal:**
```
G  =  R5  +  gamma x R6
   =  -1  +  1.0 x 10
   =  -1  +  10
   =  +9
```

**State (0,3) — three steps before goal:**
```
G  =  R4  +  R5  +  R6
   =  -1  +  -1  +  10
   =  +8
```

**State (0,2) — four steps before goal:**
```
G  =  R3  +  R4  +  R5  +  R6
   =  -1  +  -1  +  -1  +  10
   =  +7
```

**State (0,1) — five steps before goal:**
```
G  =  R2  +  R3  +  R4  +  R5  +  R6
   =  -1  +  -1  +  -1  +  -1  +  10
   =  +6
```

**State (0,0) — six steps before goal (the start):**
```
G  =  R1  +  R2  +  R3  +  R4  +  R5  +  R6
   =  -1  +  -1  +  -1  +  -1  +  -1  +  10
   =  +5
```

---

## Step 3 — Update Q-table for each state

Update rule:
```
Q(S, A)  <--  Q(S, A)  +  alpha x [ G  -  Q(S, A) ]
```

**Update (2,3) — Action Down:**
```
Q old  =  0
G      =  +10

Q new  =  0  +  0.5 x (10 - 0)
       =  0  +  5.0
       =  +5.0
```

**Update (1,3) — Action Down:**
```
Q old  =  0
G      =  +9

Q new  =  0  +  0.5 x (9 - 0)
       =  +4.5
```

**Update (0,3) — Action Down:**
```
Q old  =  0
G      =  +8

Q new  =  0  +  0.5 x (8 - 0)
       =  +4.0
```

**Update (0,2) — Action Right:**
```
Q old  =  0
G      =  +7

Q new  =  0  +  0.5 x (7 - 0)
       =  +3.5
```

**Update (0,1) — Action Right:**
```
Q old  =  0
G      =  +6

Q new  =  0  +  0.5 x (6 - 0)
       =  +3.0
```

**Update (0,0) — Action Right:**
```
Q old  =  0
G      =  +5

Q new  =  0  +  0.5 x (5 - 0)
       =  +2.5
```

---

## Monte Carlo — Q-table After Episode 1

```
State      Action    Q before    G      Q after
(0,0)      Right        0       +5      +2.5
(0,1)      Right        0       +6      +3.0
(0,2)      Right        0       +7      +3.5
(0,3)      Down         0       +8      +4.0
(1,3)      Down         0       +9      +4.5
(2,3)      Down         0       +10     +5.0
```

## What the agent learned

Every single state on the path now has a positive Q-value. The start state (0,0) already knows that going Right is worth +2.5.

The numbers increase as you get closer to the goal. This makes sense — states closer to the goal have a higher return because they need fewer steps.

## The problem with Monte Carlo

The agent learned a lot from this one episode. But it had to wait until the episode ended before learning anything. If the episode was 200 steps long, that is a long wait. And if the episode never reaches the goal, nothing is learned at all.

---
---

# METHOD 2 — 1-step Sarsa

---

## What 1-step Sarsa Does

The agent updates immediately after every single step.
It does not wait for the episode to end.
Each update uses only ONE real reward then relies on an estimate.

---

## Step 1 — Take action, observe reward, update immediately

Update rule:
```
Q(S, A)  <--  Q(S, A)  +  alpha x [ R  +  gamma x Q(S_next, A_next)  -  Q(S, A) ]
```

The agent chooses the next action A_next using epsilon-greedy BEFORE updating.
That next action is used in the formula.

---

**Step 1 — Agent at (0,0), takes Right, arrives at (0,1):**

```
Next action chosen at (0,1):  Right   (epsilon-greedy, all zeros so any action)

R             =  -1
Q(S_next, A_next)  =  Q( (0,1), Right )  =  0

G  =  R  +  gamma x Q(S_next, A_next)
   =  -1  +  1.0 x 0
   =  -1

Q( (0,0), Right )  <--  0  +  0.5 x (-1 - 0)
                   <--  -0.5
```

Q-table after Step 1:
```
(0,0) Right = -0.5     all others still 0
```

---

**Step 2 — Agent at (0,1), takes Right, arrives at (0,2):**

```
Next action chosen at (0,2):  Right

R             =  -1
Q(S_next, A_next)  =  Q( (0,2), Right )  =  0

G  =  -1  +  1.0 x 0  =  -1

Q( (0,1), Right )  <--  0  +  0.5 x (-1 - 0)
                   <--  -0.5
```

Q-table after Step 2:
```
(0,0) Right = -0.5
(0,1) Right = -0.5     all others still 0
```

---

**Step 3 — Agent at (0,2), takes Right, arrives at (0,3):**

```
Next action chosen at (0,3):  Down

R             =  -1
Q(S_next, A_next)  =  Q( (0,3), Down )  =  0

G  =  -1  +  1.0 x 0  =  -1

Q( (0,2), Right )  <--  0  +  0.5 x (-1 - 0)
                   <--  -0.5
```

---

**Step 4 — Agent at (0,3), takes Down, arrives at (1,3):**

```
Next action chosen at (1,3):  Down

R             =  -1
Q(S_next, A_next)  =  Q( (1,3), Down )  =  0

G  =  -1  +  1.0 x 0  =  -1

Q( (0,3), Down )  <--  0  +  0.5 x (-1 - 0)
                  <--  -0.5
```

---

**Step 5 — Agent at (1,3), takes Down, arrives at (2,3):**

```
Next action chosen at (2,3):  Down

R             =  -1
Q(S_next, A_next)  =  Q( (2,3), Down )  =  0

G  =  -1  +  1.0 x 0  =  -1

Q( (1,3), Down )  <--  0  +  0.5 x (-1 - 0)
                  <--  -0.5
```

---

**Step 6 — Agent at (2,3), takes Down, arrives at (3,3) — GOAL:**

```
Terminal state. Q of terminal state = 0.
No next action needed.

R             =  +10
Q(S_next, A_next)  =  Q( (3,3), any )  =  0

G  =  +10  +  1.0 x 0  =  +10

Q( (2,3), Down )  <--  0  +  0.5 x (10 - 0)
                  <--  +5.0
```

---

## 1-step Sarsa — Q-table After Episode 1

```
State      Action    Q before    G      Q after
(0,0)      Right        0       -1      -0.5
(0,1)      Right        0       -1      -0.5
(0,2)      Right        0       -1      -0.5
(0,3)      Down         0       -1      -0.5
(1,3)      Down         0       -1      -0.5
(2,3)      Down         0       +10     +5.0
```

## What the agent learned

Only the state right before the goal — (2,3) — learned that the goal is nearby.

Every other state only learned that its immediate next step gives -1.
The start state (0,0) has no idea the goal exists.

## The problem with 1-step Sarsa

The goal signal only travels ONE step back per episode.
To get the goal signal back to the start (0,0) — which is 6 steps away — the agent needs at least 6 more episodes.

This is the tyranny of the single time step.

---
---

# METHOD 3 — Q-learning

---

## What Q-learning Does

Almost identical to 1-step Sarsa with ONE key difference.

Sarsa uses the action the agent actually takes next.
Q-learning uses the BEST possible action at the next state.

This makes Q-learning off-policy.

---

## The Difference in the Formula

```
1-step Sarsa:
Q(S, A)  <--  Q(S, A)  +  alpha x [ R  +  gamma x Q(S_next, A_next)  -  Q(S, A) ]
                                              ────────────────────────
                                              action actually taken (on-policy)

Q-learning:
Q(S, A)  <--  Q(S, A)  +  alpha x [ R  +  gamma x max_a Q(S_next, a)  -  Q(S, A) ]
                                              ─────────────────────────
                                              best possible action (off-policy)
```

---

## Step by Step Updates

Because the Q-table starts at zero:
```
max_a Q(S_next, a)  =  max(0, 0, 0, 0)  =  0
```

So every update in Episode 1 gives the same numbers as 1-step Sarsa.

---

**Step 1 — Agent at (0,0), takes Right, arrives at (0,1):**

```
R                    =  -1
max_a Q( (0,1), a )  =  max(0, 0, 0, 0)  =  0

G  =  -1  +  1.0 x 0  =  -1

Q( (0,0), Right )  <--  0  +  0.5 x (-1 - 0)  =  -0.5
```

---

**Step 2 — Agent at (0,1), takes Right, arrives at (0,2):**

```
R                    =  -1
max_a Q( (0,2), a )  =  0

Q( (0,1), Right )  <--  0  +  0.5 x (-1 - 0)  =  -0.5
```

---

**Step 3 — Agent at (0,2), takes Right, arrives at (0,3):**

```
R                    =  -1
max_a Q( (0,3), a )  =  0

Q( (0,2), Right )  <--  0  +  0.5 x (-1 - 0)  =  -0.5
```

---

**Step 4 — Agent at (0,3), takes Down, arrives at (1,3):**

```
R                    =  -1
max_a Q( (1,3), a )  =  0

Q( (0,3), Down )  <--  0  +  0.5 x (-1 - 0)  =  -0.5
```

---

**Step 5 — Agent at (1,3), takes Down, arrives at (2,3):**

```
R                    =  -1
max_a Q( (2,3), a )  =  0

Q( (1,3), Down )  <--  0  +  0.5 x (-1 - 0)  =  -0.5
```

---

**Step 6 — Agent at (2,3), takes Down, arrives at (3,3) — GOAL:**

```
R                    =  +10
max_a Q( (3,3), a )  =  0   (terminal state)

G  =  +10  +  1.0 x 0  =  +10

Q( (2,3), Down )  <--  0  +  0.5 x (10 - 0)  =  +5.0
```

---

## Q-learning — Q-table After Episode 1

```
State      Action    Q before    G      Q after
(0,0)      Right        0       -1      -0.5
(0,1)      Right        0       -1      -0.5
(0,2)      Right        0       -1      -0.5
(0,3)      Down         0       -1      -0.5
(1,3)      Down         0       -1      -0.5
(2,3)      Down         0       +10     +5.0
```

Same as 1-step Sarsa in Episode 1.

---

## Where Q-learning Pulls Ahead — Episode 2

After Episode 1 we have:
```
Q( (2,3), Down ) = +5.0    everything else near 0
```

Now in Episode 2 the agent reaches state (1,3) and takes Down to (2,3).

**1-step Sarsa update for (1,3):**
```
Next action at (2,3) chosen by epsilon-greedy.
Suppose agent explores and picks Left.
Q( (2,3), Left ) = -0.5

G  =  -1  +  1.0 x (-0.5)  =  -1.5

Q( (1,3), Down )  <--  -0.5  +  0.5 x (-1.5 - (-0.5))
                  <--  -0.5  +  0.5 x (-1.0)
                  <--  -1.0
```
Sarsa got a bad update because it used a random exploratory action.

**Q-learning update for (1,3):**
```
max_a Q( (2,3), a )  =  max(-0.5, -0.5, -0.5, +5.0)
                     =  +5.0   (always picks Down — the best)

G  =  -1  +  1.0 x 5.0  =  +4.0

Q( (1,3), Down )  <--  -0.5  +  0.5 x (4.0 - (-0.5))
                  <--  -0.5  +  0.5 x 4.5
                  <--  -0.5  +  2.25
                  <--  +1.75
```

Q-learning jumped to +1.75. Sarsa dropped to -1.0.

This is why Q-learning propagates the goal signal faster than Sarsa. It always uses the best known next action regardless of what the agent actually explored.

---
---

# METHOD 4 — n-step Sarsa (n = 4)

---

## What n-step Sarsa Does

The agent collects n real rewards FIRST.
Then it goes back n steps and updates.
Updates are delayed by n steps but use n real rewards instead of just 1.

---

## The Delay — tau = t - n + 1

With n = 4:

```
t = 0   tau = 0 - 4 + 1 = -3   no update yet
t = 1   tau = 1 - 4 + 1 = -2   no update yet
t = 2   tau = 2 - 4 + 1 = -1   no update yet
t = 3   tau = 3 - 4 + 1 =  0   UPDATE state from step 0 now
t = 4   tau = 4 - 4 + 1 =  1   UPDATE state from step 1 now
t = 5   tau = 5 - 4 + 1 =  2   UPDATE state from step 2 now
t = 6   tau = 6 - 4 + 1 =  3   UPDATE state from step 3 now  (episode ends here)
drain   tau = 4              UPDATE state from step 4
drain   tau = 5              UPDATE state from step 5
```

---

## The Update Rule

```
G  =  R_{tau+1}  +  gamma x R_{tau+2}  +  gamma^2 x R_{tau+3}
   +  gamma^3 x R_{tau+4}  +  gamma^4 x Q(S_{tau+4}, A_{tau+4})

                                          only if tau+4 < T (episode not done)
                                          if tau+4 >= T  drop the bootstrap term
```

---

## Step by Step Updates

---

**t=0, t=1, t=2 — No updates yet**

```
Agent moves:
  (0,0) --R1=-1--> (0,1) --R2=-1--> (0,2) --R3=-1--> (0,3)

tau is negative. Just collecting rewards. No updates.
```

---

**t=3 — tau=0 — Update state (0,0), action Right**

```
We now have rewards R1, R2, R3, R4 available.

Rewards from tau+1=1 to tau+n=4:
  R1 = -1
  R2 = -1
  R3 = -1
  R4 = -1

Bootstrap from step tau+n = step 4:
  State at step 4 = (1,3)
  Action at step 4 = Down
  Q( (1,3), Down ) = 0

tau+n = 4   T = infinity (episode still running)
So tau+n < T   include bootstrap.

G  =  -1  +  -1  +  -1  +  -1  +  1.0^4 x 0
   =  -4  +  0
   =  -4

Q( (0,0), Right )  <--  0  +  0.5 x (-4 - 0)
                   <--  -2.0
```

Note: The goal reward has not been seen yet at this update.
This is why the start state gets a pessimistic value (-2.0).
It only saw four -1 rewards with no sign of the goal.

---

**t=4 — tau=1 — Update state (0,1), action Right**

```
Rewards from tau+1=2 to tau+n=5:
  R2 = -1
  R3 = -1
  R4 = -1
  R5 = -1

Bootstrap from step 5:
  State at step 5 = (2,3)
  Action at step 5 = Down
  Q( (2,3), Down ) = 0

G  =  -1  +  -1  +  -1  +  -1  +  0
   =  -4

Q( (0,1), Right )  <--  0  +  0.5 x (-4 - 0)
                   <--  -2.0
```

---

**t=5 — tau=2 — Update state (0,2), action Right**

```
Rewards from tau+1=3 to tau+n=6:
  R3 = -1
  R4 = -1
  R5 = -1
  R6 = +10   <- GOAL REWARD appears here

tau+n = 6 = T   episode ended exactly at step 6
So tau+n >= T   NO bootstrap term.

G  =  -1  +  -1  +  -1  +  10
   =  +7

Q( (0,2), Right )  <--  0  +  0.5 x (7 - 0)
                   <--  +3.5
```

This is the first update that includes the goal reward.

---

**t=6 — tau=3 — Update state (0,3), action Down**

```
Episode ended at T=6. Drain phase begins.
Agent stops moving but updates continue.

Rewards from tau+1=4 to min(tau+n, T) = min(7, 6) = 6:
  R4 = -1
  R5 = -1
  R6 = +10

tau+n = 7 > T = 6   NO bootstrap.

G  =  -1  +  -1  +  10
   =  +8

Q( (0,3), Down )  <--  0  +  0.5 x (8 - 0)
                  <--  +4.0
```

---

**Drain — tau=4 — Update state (1,3), action Down**

```
Rewards from tau+1=5 to min(8,6) = 6:
  R5 = -1
  R6 = +10

No bootstrap.

G  =  -1  +  10  =  +9

Q( (1,3), Down )  <--  0  +  0.5 x (9 - 0)
                  <--  +4.5
```

---

**Drain — tau=5 — Update state (2,3), action Down**

```
Rewards from tau+1=6 to min(9,6) = 6:
  R6 = +10

No bootstrap.

G  =  +10

Q( (2,3), Down )  <--  0  +  0.5 x (10 - 0)
                  <--  +5.0
```

Loop ends. tau = 5 = T-1 = 6-1. Done.

---

## n-step Sarsa — Q-table After Episode 1

```
State      Action    Q before    G      Q after
(0,0)      Right        0       -4      -2.0
(0,1)      Right        0       -4      -2.0
(0,2)      Right        0       +7      +3.5
(0,3)      Down         0       +8      +4.0
(1,3)      Down         0       +9      +4.5
(2,3)      Down         0       +10     +5.0
```

---

## Why (0,0) and (0,1) got -2.0

Those two states were updated at t=3 and t=4 — before the agent had reached the goal.

At t=3 the agent only had rewards R1 to R4, all of which were -1.
The bootstrap term Q( (1,3), Down ) = 0 at that moment.

So G = -4. The agent had not seen the goal yet.

In Episode 2, Q( (1,3), Down ) will be +4.5 from what we just learned.
The next time (0,0) is updated, the bootstrap term will be +4.5 instead of 0.

```
Episode 2 update for (0,0):
G  =  -1 + -1 + -1 + -1  +  1.0^4 x Q( (1,3), Down )
   =  -4  +  4.5
   =  +0.5

Q( (0,0), Right )  <--  -2.0  +  0.5 x (0.5 - (-2.0))
                   <--  -2.0  +  0.5 x 2.5
                   <--  -2.0  +  1.25
                   <--  -0.75
```

It is already moving in the right direction after just Episode 2.

---
---

# FINAL COMPARISON — All Four Methods

---

## Q-values After Episode 1

```
State    Action   Monte Carlo   1-step Sarsa   Q-learning   n-step (n=4)
(0,0)    Right       +2.5          -0.5          -0.5          -2.0
(0,1)    Right       +3.0          -0.5          -0.5          -2.0
(0,2)    Right       +3.5          -0.5          -0.5          +3.5
(0,3)    Down        +4.0          -0.5          -0.5          +4.0
(1,3)    Down        +4.5          -0.5          -0.5          +4.5
(2,3)    Down        +5.0          +5.0          +5.0          +5.0
```

---

## States That Learned the Goal is Reachable After Episode 1

```
Monte Carlo:     6 out of 6   all states updated with goal signal
n-step (n=4):    4 out of 6   steps 3 to 6 have positive values
1-step Sarsa:    1 out of 6   only the step before goal
Q-learning:      1 out of 6   same as Sarsa in episode 1
```

---

## Visual Summary on the Grid Path

```
         (0,0)  (0,1)  (0,2)  (0,3)  (1,3)  (2,3)  GOAL

MC:      +2.5   +3.0   +3.5   +4.0   +4.5   +5.0    G
         all states know goal is reachable

Sarsa:   -0.5   -0.5   -0.5   -0.5   -0.5   +5.0    G
         only last state knows

Q-learn: -0.5   -0.5   -0.5   -0.5   -0.5   +5.0    G
         same as Sarsa now, pulls ahead episode 2+

n-step:  -2.0   -2.0   +3.5   +4.0   +4.5   +5.0    G
         4 states useful, 2 correct themselves next episode
```

---

## Key Properties Compared

| Property | Monte Carlo | 1-step Sarsa | Q-learning | n-step Sarsa |
|---|---|---|---|---|
| Policy | On-policy | On-policy | Off-policy | On-policy |
| Updates during episode | No | Yes | Yes | Yes |
| Waits for episode end | Yes | No | No | No |
| Real rewards used | All | 1 | 1 | n = 4 |
| Bootstrap | None | After 1 step | After 1 step (greedy) | After n steps |
| Goal signal after Ep 1 | All states | 1 state | 1 state | 4 states |
| Corrects itself later | N/A | Slow | Faster | Yes Episode 2 |
| Safe near traps | Yes | Yes | Risky | Yes |

---

## Conclusion

For this sparse 4x4 gridworld the ranking after Episode 1 is:

```
1st   Monte Carlo     most states learned   but waited for episode end
2nd   n-step Sarsa    4 states learned      updated during episode
3rd   Q-learning      1 state learned       pulls ahead from Episode 2
4th   1-step Sarsa    1 state learned       slowest to propagate goal signal
```

The advantage of n-step Sarsa over 1-step Sarsa and Q-learning is clearest
in sparse reward environments like this one — where reward only appears at the goal.

If rewards appeared at every step the advantage would be much smaller.

---

## Reference

Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
MIT Press. Chapter 6: TD Learning. Chapter 7: n-step Bootstrapping, Section 7.2.