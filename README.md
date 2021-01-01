# SimplySAC: A Minimal Soft-Actor-Critic Pytorch Implementation

SimplySAC replicates SAC with minimum (~200) lines of code in clean, readable PyTorch style, while trying to use as little additional tricks and hyper-parameters as possible.

## Implementation details:
<li>
The actor's log_std output is clamped to lie within [-20, 2] according to the authors' source code.
</li>

<li>
Before learning, the replay buffer is warmed up with 1e4 transitions collected using an uniformly random policy.
</li>

<li>
The Q-value in the actor's loss averages from two critics.
</li>
<br>
That's it! All other things follow the original paper and pseudo code.
</li>

## Mujoco benchmarks:

![walker](./figures/walker2d.png)

![cheetah](./figures/halfcheetah.png)

![ant](./figures/ant.png)

![humanoid](./figures/humanoid.png)

Same as the figures in the original paper, these figures are produced with:
<li>
One evaluation episode every 1e3 steps.
</li>
<li>
5 random seeds, where the mean return is represented by the solid line, and max/min return by the shaded area.
</li>
<br>

To reproduce the results, simply run:
```
sh exp.sh
```
Assuming 4 GPU cards, this runs 5 seeds for one environment per card.

