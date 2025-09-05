# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 08:40:59 2025

@author: Admin
"""

'''



This code implements a more verstile Neuronal Posterior Estimator (NPE) within the SBI framework. Are implemented both an amortized 
and a non-amortized version. Indeed the foundamental idea is to initially train an amortized ensemble network. Given a target observation
initial prior distribution is obtain from it. Afterward the sequential paradigm is used. 

####----------- AMORTIZED NETWORK -----------####

This network is based on Neural Spline Flow architecture. An ensemble of such networks is trained through grid search in a deep manner as suggested
by Deistler et al (2022)[2]. The number of networks making up the ensemble is arbitrarly choosen. Be aware that the performance increases with the number 
of networks whitin the ensemble. In this setting the Simulation-based calibration (SBC) is used as both a diagnostic tool of the posterior estimation itself
and as way to alleviate the issue of overconfidence as done by Hermans et al. (2020) [6]







####----------- SEQUENTIAL NETWORK -----------####

THe architecture is inspired from the one proposed by Deistler et al (2022) [1]. THe article addresses
two widespread issues in sequential NPE:
    
    a) the sequential scheme of SNPE can be unstable. SNPE requires a modification of the loss function 
        compared to NPE, which suffers from issues that can limit its effectiveness on 
        (or even prevent their application to) complex problems.
        
    b) several commonly used diagnostic tools for SBI rely on performing inference across multiple observations. 
        In SNPE (in contrast to NPE), this requires generating new simulations and network retraining for each observation, 
        which often prohibits the use of such diagnostic tools.
        
        
To solve the above stated problems the authors introduce the Truncated sequential NPE (TSNPE) and the daignostic tool called
simulation-based coverage calibration (SBCC).


For a wrapped coincise implementation of TSNPE look up at [5]


--- TSNPE --- [1]
TSNPE follows the SNPE formalism, but uses a proposal which is a truncated version of the prior: TSNPE draws simulations from 
the prior, but rejects them before simulation if they lie outside of the support of the approximate posterior. Thus, the proposal 
is (within its support) proportional to the prior, which allows us to train the neural network with maximum-likelihood in every 
round and, therefore, sidesteps the instabilities (and hence ‘hassle’) of previous SNPE methods.
-------------

---- SBCC ---- [1]
TSNPE allows direct sampling and density evaluation of the approximate posterior, and thus permits computing expected coverage 
of the full posterior quickly (without MCMC) and at every iteration of the algorithm, thus allowing to diagnose failures of the
method even for high-dimensional parameter spaces. 
In this framework the SBCC only measures the QUALITY of the training procedure.
--------------


--- Ensemble models --- [2]
Ensembles of models constitute a standard method to improve predictive performance. In this work, we consider an ensemble model 
that averages the approximated posteriors of n posterior estimators that are either trained independently on the same dataset 
(deep ensemble). It is found that deep ensembling constitutes an immediately applicable and easy way to mitigate the overconfidence 
issue and build more reliable posterior estimators.

--- Neuronal Spline Flow --- [3][4]
Neural Spline Flow (NSF) is a type of normalizing flow, which is a generative model that learns to transform a simple probability 
distribution (like a standard Gaussian) into a complex, multi-modal distribution that matches the training data.
This transformation is achieved by chaining together a series of simple, invertible, and differentiable functions. 
The "flow" refers to the movement of probability mass from the simple distribution to the complex one.
NSF's key innovation is its use of monotonic rational-quadratic splines as the core transformation. Earlier normalizing flows used 
simpler transformations like affine or coupling layers, which are less flexible. The monotonic rational-quadratic spline allows 
the flow to model highly complex, non-linear relationships with high precision. This at the cost of an higher computational requirements.
----------------------------






Briefly the architecture is made of:
    
    1) A Neuronal Spline Flow deep network as the estimator [3] embedded in an ensmeble pool [2]
    2) The sampling rejection paradigm for truncation [1]
    3) Diagnotic tool [1]
    4) 


    
    
NOTES:
    1) Spline transformations are defined over the domain :math:`[-5, 5]`. Any feature outside of this domain is not transformed. 
        It is recommended to standardize features (zero mean, unit variance) before training.
    
    
    
    
    
    

References:
    
    [1] Truncated proposals for scalable and hassle-free simulation-based inference
    
    [2] A Trust Crisis In Simulation-Based Inference? Your  Posterior Approximations Can Be Unfaithful
    
    [3] https://github.com/sbi-dev/sbi/blob/main/sbi/neural_nets/net_builders/flow.py
    
    [4] Neural Spline Flows
    
    [5] https://sbi-dev.github.io/sbi/latest/tutorials/16_implemented_methods/

    [6] Towards constraining warm dark matter with stellar streams through neural simulation-based inference
'''

import numpy as np
from sbi.neural_nets.net_builders import build_zuko_nsf
import torch
from sbi.inference import NPE
from sbi.utils import RestrictedPrior, get_density_thresholder
from sbi.utils import BoxUniform

from sbi.inference import NPE, NRE
from sbi.analysis import pairplot
# Define the prior
num_dims = 2
num_sims = 3000
num_rounds = 3
prior = BoxUniform(low=torch.zeros(num_dims), high=torch.ones(num_dims))
simulator = lambda theta: theta + torch.randn_like(theta) * 0.01
x_o = torch.tensor([0.5, 0.5])


# Generate some samples:
theta = prior.sample((num_sims,))
x = simulator(theta)


prior = BoxUniform(torch.zeros(2), torch.ones(2))

from sbi.neural_nets import posterior_nn

density_estimator_build_fun = posterior_nn(
    model="zuko_nsf", hidden_features=60, num_transforms=10
)





inference = NPE(prior=prior, density_estimator=density_estimator_build_fun)
proposal = prior

# Sequential approach:
    
for _ in range(num_rounds):
    theta = proposal.sample((num_sims,))
    x = simulator(theta)
    _ = inference.append_simulations(theta, x).train(force_first_round_loss=True)
    posterior = inference.build_posterior().set_default_x(x_o)

    accept_reject_fn = get_density_thresholder(posterior, quantile=1e-4)
    proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")


#%%
%matplotlib
x_o = torch.tensor([0.5, 0.5])
print(f"Shape of x_o: {x_o.shape}            # Must have a batch dimension")

samples = posterior.sample((1000,), x=x_o)
print(f"Shape of samples: {samples.shape}  # Samples are returned with a batch dimension.")

samples = samples.squeeze(dim=1)
print(f"Shape of samples: {samples.shape}     # Removed batch dimension.")
fig, ax = pairplot(
    samples, limits=[[0, 1], [0, 1]], figsize=(5, 5)
)

