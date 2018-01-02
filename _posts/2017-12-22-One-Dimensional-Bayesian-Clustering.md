---
title: "Bayesian Nonparametric Clustering"
layout: post
---

A common task in data mining is clustering data. A class of clustering models, known as mixture models, represent the clusters as a mixture of distributions, with each distribution describing a cluster. I will discuss Bayesian treatments of these models and some applications.

First, we need to load packages.


```python
import numpy as np
from theano import tensor as tt

import scipy as sp
import pymc3 as pm

import matplotlib.pyplot as plt
```

For demonstrative purposes, we will cluster an artificial dataset in one dimension.


```python
SEED = 238902
np.random.seed(SEED)

X = np.random.permutation(np.concatenate([np.random.normal(6, 1, 100), np.random.normal(2,.8,50), np.random.normal(12, 1.4, 100), np.random.normal(18, .9, 10)]))

fig, ax = plt.subplots(figsize=(10,5))

ax.hist(X, bins=50, alpha=1)
ax.set_xlabel("X")
plt.show()
```


![png]({{"images/One-Dimensional-Bayesian-Clustering/output_4_0.png"|absolute_url}})


We will use a Bayesian treatment of a nonparametric Gaussian mixture model. This is a popular technique for clustering that uses a Dirichlet Process as a prior distribution over possible probability measures. This distribution is updated by applying a likelihood term, which is simply the fit of the sampled mixture model to the data.



```python
N = X.shape[0] # number of atoms
K = 30 # number of (distribution) samples from DP
```


```python
def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1-beta)[:-1]])
    return beta * portion_remaining
```

To estimate the final distribution, we can sample from the posterior by regarding its fit to the given data.


```python
with pm.Model() as model:
    alpha = pm.Gamma('alpha', 1., 1.)
    beta = pm.Beta('beta', 1., alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta)) # mixture weights

    tau = pm.Gamma('tau', 1., 1., shape=K)
    lambda_ = pm.Uniform('lambda', 0, 5, shape=K)
    mu = pm.Normal('mu', 0, tau=tau*lambda_, shape=K)

    obs = pm.NormalMixture('obs', w, mu, tau=lambda_*tau, observed=X) # likelihood
```


```python
with model:
    trace = pm.sample(500, tune=1000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    100%|█████████▉| 1498/1500 [01:08<00:00, 21.85it/s]/home/chris/anaconda3/lib/python3.6/site-packages/pymc3/step_methods/hmc/nuts.py:467: UserWarning: Chain 0 contains 51 diverging samples after tuning. If increasing `target_accept` does not help try to reparameterize.
      % (self._chain_id, n_diverging))
    100%|██████████| 1500/1500 [01:08<00:00, 21.86it/s]


We can plot posterior of the mixture means generated and the mixture parameter $$\alpha$$.


```python
pm.traceplot(trace, varnames=["mu", "alpha"])
plt.show()
```


![png]({{"images/One-Dimensional-Bayesian-Clustering/output_12_0.png"|absolute_url}})


We can ensure that truncating the number of components at 30 does not interfere with the process by visualizing the weights. This visualization clearly shows convergence to 3 main components, so we have nothing to worry about.


```python
plot_w = np.arange(K)
fig, ax = plt.subplots(figsize=(10,5))

ax.bar(plot_w, trace["w"].mean(axis=0))
ax.set_xlabel("Component")
ax.set_ylabel("Weight")
ax.grid(axis="y")
plt.show()
```


![png]({{"images/One-Dimensional-Bayesian-Clustering/output_14_0.png"|absolute_url}})


Using the samples, we can compute the expected value of the mixture pdf and each of the components. Below is a visualization of the expected value.


```python
x_plot = np.arange(np.min(X),np.max(X),.1)

pdfs = []
components = []

for i in range(len(trace["mu"])):
    mu = trace["mu"][i]
    sigma = 1/np.sqrt(trace["tau"][i] * trace["lambda"][i])
    weight = trace["w"][i]

    pdf = np.zeros_like(x_plot)
    comps = []
    for m, s, w in zip(mu, sigma, weight):
        p = w * sp.stats.norm.pdf(x_plot, m, s)

        pdf += p
        comps.append(p)
    pdfs.append(pdf)
    components.append(comps)

pdfs = np.array(pdfs)
components = np.array(components)
```


```python
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

samples = mlines.Line2D([], [], color='gray', label='Samples', alpha=.5)
exp_value = mlines.Line2D([], [], color='black', label='Sample Mean')
comps = mlines.Line2D([], [], linestyle="--", color='black', label='Components')
hgram = mpatches.Patch(color="blue", label="Observed Data")

fig, ax = plt.subplots(figsize=(10,5))

scale = 1.3
ax.hist(X, weights=scale * np.ones_like(X)/len(X), bins=50, label="data") # observed data

for i in range(len(pdfs)):
    ax.plot(x_plot, pdfs[i], color="gray", alpha=.05) # DP samples

for pdf in components.mean(axis=0):
    ax.plot(x_plot, pdf, "--", color="black")

ax.plot(x_plot, pdfs.mean(axis=0), color="black", label="expected density") # expected value
ax.legend(handles=[samples, exp_value, comps, hgram])
ax.set_xlabel("X")
ax.set_ylabel("Density")

plt.show()
```


![png]({{"images/One-Dimensional-Bayesian-Clustering/output_17_0.png"|absolute_url}})


We can see that the nonparametric result identifies 3 distinct clusters in the data that can be modeled as a mixture of 3 Gaussians. I would like to extend this result to the multidimensional case, but that will require some extra techniques that are not shipped as builtins with PyMC3.
