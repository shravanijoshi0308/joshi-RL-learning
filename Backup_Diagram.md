# n-Step Sarsa: Algorithm Workflow and Backup Diagram

## Purpose

Understanding the mathematical formula of n-Step Sarsa is important, but understanding how information flows through the algorithm is even more important.

This document explains:

1. What n-Step Sarsa does.
2. Why it works.
3. How the algorithm updates its Q-values.
4. How rewards propagate through the state-action sequence.

---

# What is n-Step Sarsa?

n-Step Sarsa is an on-policy Temporal Difference learning algorithm that combines ideas from:

- Monte Carlo methods
- One-Step Temporal Difference learning

Instead of:

- waiting until the end of the episode (Monte Carlo), or
- updating after every single action (1-Step Sarsa),

n-Step Sarsa waits for **n rewards** before performing an update.

This allows the agent to learn from more real experience while still learning online.

---

# Why Do We Need n-Step Sarsa?

Consider a long path to the goal.

```text
Start → S1 → S2 → S3 → S4 → Goal
```

Suppose the reward is received only at the goal.

## Problem with 1-Step Sarsa

Only one state-action pair is updated at a time.

```text
Goal reward
    ↑
   S4

Next episode

Goal reward
    ↑
   S3

Next episode

Goal reward
    ↑
   S2
```

The reward signal travels backward very slowly.

---

## Problem with Monte Carlo

Monte Carlo waits until:

```text
Start → S1 → S2 → S3 → S4 → Goal
```

is completely finished.

Only then does learning begin.

This causes delayed updates and high variance.

---

## n-Step Sarsa Solution

n-Step Sarsa allows rewards to travel backward multiple steps at once.

For example:

```text
Goal reward
      ↑
S2 ← S3 ← S4 ← Goal
```

A single update can influence several earlier decisions.

This accelerates learning.

---

# How n-Step Sarsa Works

Assume:

```text
n = 3
```

The agent experiences:

```text
S0,A0
   │
   ▼
  R1

S1,A1
   │
   ▼
  R2

S2,A2
   │
   ▼
  R3

S3,A3
```

The algorithm now has:

```text
R1
R2
R3
```

which is enough information to update.

---

# Backup Diagram

```text
S0,A0 ──R1──► S1,A1 ──R2──► S2,A2 ──R3──► S3,A3

   ▲
   │
   │
   └────────────── Update using
                   R1 + γR2 + γ²R3
                   + γ³Q(S3,A3)
```

This is called a **3-step backup**.

The update begins at:

```text
(S0,A0)
```

and looks ahead three rewards before bootstrapping.

---

# General n-Step Backup

```text
Sτ,Aτ
   │
   ▼
Rτ+1

   │
   ▼
Rτ+2

   │
   ▼

...

   │
   ▼

Rτ+n

   │
   ▼

Sτ+n,Aτ+n
```

The update target becomes:

```math
G_{τ:τ+n}
=
R_{τ+1}
+
γR_{τ+2}
+
...
+
γ^{n-1}R_{τ+n}
+
γ^nQ(S_{τ+n},A_{τ+n})
```

The target contains:

1. n real rewards
2. One bootstrap estimate

---

# Information Flow

The key idea is:

### 1-Step Sarsa

```text
Reward
   │
   ▼

One step backward
```

### n-Step Sarsa

```text
Reward
   │
   ▼

Several steps backward
```

### Monte Carlo

```text
Reward
   │
   ▼

Entire episode backward
```

n-Step Sarsa sits between these two extremes.

---

# Algorithm Workflow

```text
Start Episode
       │
       ▼

Choose Action
       │
       ▼

Take Action
       │
       ▼

Store Reward
       │
       ▼

Have n rewards been collected?
       │
   ┌───┴────┐
   │        │
  No       Yes
   │        │
   ▼        ▼

Continue   Compute n-Step Return
                │
                ▼

        Update Q-value
                │
                ▼

         Continue Episode
                │
                ▼

           Episode Ends
```

---

# Key Insight

The parameter n controls how far the algorithm looks into the future.

```text
n = 1
    ↓
1-Step Sarsa

n = 4
    ↓
4-Step Sarsa

n = 8
    ↓
8-Step Sarsa

n → ∞
    ↓
Monte Carlo
```

Therefore, n-Step Sarsa creates a bridge between Temporal Difference learning and Monte Carlo learning.

The choice of n determines the balance between:

- Learning speed
- Bias
- Variance
- Reward propagation

---

# Takeaway

Monte Carlo waits too long.

1-Step Sarsa looks too little into the future.

n-Step Sarsa provides a middle ground by collecting multiple real rewards before bootstrapping, allowing faster and more effective propagation of reward information through the state-action space.