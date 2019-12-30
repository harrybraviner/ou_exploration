Adding Anti-correlation to the Ornstein-Uhlenbeck Process
=========================================================

In RL applications we often want an initial period in which our agent doesn't act according to a learned policy, but acts at random to explore.

In [Lillicrap et al. 2016](https://arxiv.org/abs/1509.02971) (the paper than introduced DDPG), the authors use Ornstein-Uhlenbeck noise for this exploration step.
[Spinning Up's DDPG page](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) claims that uniform noise is actually sufficient, but I found that this isn't always true.
For example, on the `MountainCarContinous-v0` problem, uniform noise results in very little exploration as the cart never manages to climb the hill, even after 50 episodes.
OU noise, on the other hand, results in several successful climbs within 25 episodes, and the first episode with the agent in the driving seat result in a near-perfect score.

# Discrete Ornstein-Uhlenbeck process

The Ornstein-Uhlenbeck process is really a continuous stochastic process.
Here's we'll consider a discrete stochastic recurrence relation, with fixed step sizes.
This really requires none of the machinery of stochastic calculus, and mostly can be attacked with common-sense reasoning.

Our state will be some vector `x`, and the process is parameterized by a decay constant `theta`, and a noise level `sigma`.
Our recurrence relation is then
```
x[t + 1] = x[t] + ( -theta * x[t] + sigma * Z[t] )
```
where `Z[t]` are independent standard Gaussians.

A little thought shows that we should have `theta > 0` for stability (otherwise the process diverges).
You probably also want `theta < 1`, since there isn't a continuous analogue otherwise.

This results in a random walk that has a tendency to 'decay' towards tthe origin.

# Can we make an 'anti-correlated' version of OU noise?

The Ornstein-Uhlenbeck process has a tendency to revert to the origin.
Once there, it's equally likely to explore in the same direction that it just explored.
Can we encourage it to explore in the *opposite* direction?

My idea here is to take rolling mean of the values, call it `mu[t]`, and decay the process towards `-mu[t]`.
i.e. the rule is "go look in the opposite direction to where you've looked before".

The update rule is then
```
x[t + 1] = x[t] + ( -theta * (x[t] - mu[t]) + sigma * Z[t] )
```
Note that this is no longer a Markov process in `x[t]`.

In my experiments I've taken `mu[t]` to be an exponentially weighted moving average, with a delay buffer.
i.e. we ignore the most recent few points, and take an EWMA over the points before that.

I do also allow attracting towards `mu[t]` (rather than `-mu[t]`).

Results are in [this notebook](processes.ipynb).

