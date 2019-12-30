Adding Anti-correlation to the Ornstein-Uhlenbeck Process
=========================================================

In RL applications we often want an initial period in which our agent doesn't act according to a learned policy, but acts at random to explore.

In [Lillicrap et al. 2016](https://arxiv.org/abs/1509.02971) (the paper than introduced DDPG), the authors use Ornstein-Uhlenbeck noise for this exploration step.
[Spinning Up's DDPG page](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) claims that uniform noise is actually sufficient, but I found that this isn't always true.
For example, on the `MountainCarContinous-v0` problem, uniform noise results in very little exploration as the cart never manages to climb the hill, even after 50 episodes.
OU noise, on the other hand, results in several successful climbs within 25 episodes, and the first episode with the agent in the driving seat result in a near-perfect score.

# Can we make an 'anti-correlated' version of OU noise?

The Ornstein-Uhlenbeck process has a tendency to revert to the origin.
Once there, it's equally likely to explore in the same direction that it just explored.
Can we encourage it to explore in the *opposite* direction?
