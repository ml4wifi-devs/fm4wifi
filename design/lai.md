# Large AI for MAPC

The problem of MAPC scheduling can be deterministically solved given the exact station and APs positions as well as the all to all attenuation.
In practice, these are unknown and typically changing over time.

## Formal problem statement

Let us denote the position of all devices (STAs and APs) by $X$ and the configuration of attenuators (mostly walls) by $W$.
Under uncertainty, the model must describe both in terms of probability measures $d_\theta(X)$ and $a_\theta(W)$ with unknown parameters $\theta$
An all to all propagation loss $L$ can be expressed as conditional $p(L|W,X)$ - i.e. given walls and positions it is possible to estimate distances and  the losses.

We argue that a successful approach to MAPC problem can be framed as an inference problem.
An agent schedules APs for simultaneous operation and observes the transmission rate $R$, which is determined by the $W,X$ and some unknown factors.
In the Bayesian point of view, this can break the network control int two steps.

1. Inference problem of getting posterior over $X$ and $W$ given rate $p(X_{t+1},W_{t+1}|R_t)\sim~p(R_t|X_{t+1},W_{t+1})p(X_{t},W_{t})$
2. Approximate deterministic solver $\hat a = \arg\max_{A} R$, where $A$ is the set of actions

## Practical considerations

The inference problem can be solved using a variety of methods, such as MCMC, variational inference, or deep learning-based approaches.
The common property of the inference methods is that they produce distribution. We propose to relay on stochastic models that can incorporate this aspect of inherent uncertainty.
The second step can be approximated by a deterministic model evaluated on samples from the posterior distribution.

### Training improvement

- Use diffusion model to introduce stochasticity to the output of the model.
- Add device position as an auxiliary targen for the model.
- Increase context lengths as this sharpens the posterior distribution.

## Distribution over $X$

The loss $L$ can be expressed as a function of the positions of the devices and the configuration of the attenuators, but its invariant to global isometric transformation of the positions.
To reduce variance, we propose the following distributions of the positions:

1. Let the first device be at the origin, i.e. $X_1 = (0,0)$.
2. Let the second device be at a distance $d$ from the first device along the `x` axis, i.e. $X_2 = (d,0)$.
3. The remaining devices are normally distributed with some mean and covariance matrix.
4. The remaining reflection symetry can be expressed by taking the mixture of two distributions described above
5. One can relax `3` and instead of taking the distance $d$ one may use `d` as  `x` coordinate and use the mixture of four components representing reelections over `x` and `y` axes.

Such a distribution can be represented as z mixture bijective transformations of multivariate normal distribution that greatly simplify parametrization.

## Distribution over $W$

This is probably not required by the walls can be expressed as rectangles with material attenuation.
As such, two points can be used to define a rectangular wall with optional rotation.
Complicated environment can be expressed as infinite mixtures of such rectangles.
The Number of walls can be deterministic of distributed according to e.g. Poisson distribution.