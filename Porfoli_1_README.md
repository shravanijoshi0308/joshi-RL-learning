RL Portfolio — n-step Sarsa vs Gridworld Algorithms
Course: Reinforcement Learning · Portfolio Task 1Chapter: Sutton & Barto — Reinforcement Learning (2nd Ed.) · Chapter 7.2Topic: Evaluating n-step Sarsa against earlier learning algorithms across Gridworld variants

Intuition First : The Driving Lesson Analogy
Before any equations, here is the core idea in plain language.
Imagine you are taking driving lessons. Your coach can give feedback in three different ways:

Scenario 1 : Feedback Only at the End of the Trip
"The trip was 60 minutes. Your coach tells you how it went at the very end."
You drove for an hour. Some turns were great, some were bad. But by the time your coach gives feedback, you have completely forgotten which specific action at which moment produced the best outcome. You only know the final result — good trip or bad trip.
This is Monte Carlo.The agent waits until the episode ends, then updates everything backwards using the final return. It learns from the complete picture but with no fine-grained credit.

Scenario 2 : Feedback After Every Single Action
"Your coach gives feedback after every turn, every brake, every steering input — immediately."

Now imagine: you take a turn at a junction. Your coach says "bad" immediately. But three turns later, it turns out that junction was actually the right route — the coach judged too early with too little information. Feedback is quick, but it is based on only the very next moment. Learning is fast per-step but shallow.
This is 1-step TD / Sarsa(0).The agent updates after every single action using only the immediate next reward. Fast, but the signal is narrow — only one step of real information before bootstrapping.

Scenario 3 : Feedback After Observing a Few Actions (the sweet spot)
"Your coach watches you make 4 actions — Action 1 → Action 2 → Action 3 → Action 4 — and then gives you feedback based on what they observed."
Now your coach has seen enough to make a meaningful, informed judgment. Not so little information that feedback is shallow. Not so much waiting that you forgot what happened. The feedback covers a sequence of actions, so credit flows back across multiple steps at once.
This is n-step Sarsa.The agent collects n real rewards, then updates. It propagates credit n steps backward in a single episode — faster than 1-step TD, more stable than Monte Carlo.

The Spectrum 

Monte Carlo         n-step Sarsa         1-step TD / Sarsa
(n = ∞)          (n = 4, 8, 16 ...)        (n = 1)

Wait for         Sweet spot:              Update after
episode end   n real rewards then        every single
               update n steps back          action
               
Low variance ←────────────────────────→ High variance
High bias   ←────────────────────────→ Low bias


Limitations of the Two Extremes : 
Monte Carlo : 
Limitation	Explanation
Too late	Waits for the episode to end before making any updates
High variance	Long episodes produce very noisy returns — small random events early in the episode change the total dramatically
Useless online	Cannot learn anything until the episode terminates
Slow in practice	Needs many full episodes before useful Q-values emerge

1-step TD / Sarsa(0) : 
Limitation	Explanation
Too narrow	Only the immediately preceding action gets updated each step
Slow credit	In a 20-step path to the goal, 1-step Sarsa needs 20 episodes just to propagate the goal reward back to the start
Tyranny of the single time step	The update interval and the action interval are forced to be equal
Shallow signal	Bootstrap estimate after just 1 reward — high bias

n-step Sarsa solves both
* Updates at every step (unlike Monte Carlo) 
* Uses n real rewards before bootstrapping (unlike 1-step TD) 
* Tunable: choose n to match your environment's reward structure 