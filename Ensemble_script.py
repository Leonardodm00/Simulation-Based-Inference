
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:36:02 2025

@author: Admin
"""


import torch
from joblib import Parallel, delayed
from sbi.examples.minimal import simple
from sbi.utils import BoxUniform
from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior
from sbi.inference import NPE
from sbi.utils import RestrictedPrior, get_density_thresholder
import torch
import itertools
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

'''


To use the Parallel processing the networks must be reinitialized in each parallell sub-process. Just loading and 
trainig the network is not enought because child networks' internal state is cleared. Therefore a temporary folder is 
created and cleaned afterwards, in which temporary network internal states are loaded.








'''












def check_posterior_diversity(posteriors, x_o, num_samples=1000):
    """
    Approximates the Kullback-Leibler (KL) divergence between pairs of
    posterior distributions from an ensemble of trained NPEs.

    This function leverages a Monte Carlo approximation to determine if the
    posterior distributions are significantly different for a given observation.
    A higher KL divergence value indicates a greater difference between the
    two distributions.

    Args:
        posteriors (list): A list of trained NPE posterior objects from an sbi ensemble.
        x_o (torch.Tensor): The target observation for which to evaluate the posteriors.
                            Must be a 1D tensor, e.g., torch.tensor([1.0, 2.0]).
        num_samples (int): The number of samples to draw from the posterior
                           for the Monte Carlo approximation. A higher number
                           improves the approximation's accuracy.

    Returns:
        dict: A dictionary where each key is a tuple of posterior indices
              (e.g., (0, 1)) and the value is the approximate KL divergence.
    """
    
    num_posteriors = len(posteriors)
    if num_posteriors < 2:
        print("Warning: The ensemble must contain at least two posteriors to compare.")
        return {}

    kl_divergences = {}
    
    
    # Load all the models
    
    # Iterate through all unique pairs of posterior indices
    for i, j in itertools.combinations(range(num_posteriors), 2):
        posterior_a = posteriors[i].build_posterior()
        posterior_b = posteriors[j].build_posterior()

        # Draw samples from the first posterior (p_a)
        # We wrap x_o in an extra dimension to match the expected batch shape.
        try:
            samples_from_a = posterior_a.sample((num_samples,), x=x_o.reshape(1, -1))
        except Exception as e:
            print(f"Error sampling from posterior {i}: {e}. Skipping pair ({i}, {j}).")
            continue

        # Evaluate the log probabilities of these samples under both posteriors
        # log p(theta | x)
        log_prob_a = posterior_a.log_prob(samples_from_a, x=x_o.reshape(1, -1))
        log_prob_b = posterior_b.log_prob(samples_from_a, x=x_o.reshape(1, -1))

        # Approximate KL(p_a || p_b) = E_{theta ~ p_a} [log p_a(theta) - log p_b(theta)]
        # This is the average difference in log-probabilities
        # The .mean() function computes the expectation.
        kl_div = (log_prob_a - log_prob_b).mean().item()

        # Store the result
        kl_divergences[(i, j)] = kl_div

    # Optional: Print a summary
    print("\n--- Approximate KL Divergences (KL(p_i || p_j)) ---")
    for pair, kl_div in kl_divergences.items():
        print(f"KL({pair[0]} || {pair[1]}): {kl_div:.4f}")
    
    return kl_divergences

def train_single_network(inference_object,train_params,train_data):
    
    '''
    Function to call for single NPE
    
    Returns:
        1) Trained inference object. Neccessary for secquential paradigm.
        2) Posterior object.
    
    Respectively in first and second position of the return list
    '''
    
    
    _ = inference_object.append_simulations(train_params, train_data).train(force_first_round_loss=True,max_num_epochs= 300)
    posterior = inference_object.build_posterior()
    
    
    return [inference_object,posterior]

def train_single_network_parallel(inference_object_path,object_number,train_params,train_data):
    from sbi.neural_nets import posterior_nn
    '''
    Function to call for single NPE
    It saves the inference object in the temporary folder
    
    Returns:
        
        1) Posterior object.
    
   
    
    Based on your previous interactions and the UserWarning, your interpretation is correct. 
    When you reload the pickled Inference object multiple times, the training is not reset. 
    Instead, it effectively continues from where it left off.
    
    '''
    load_path = os.path.join(inference_object_path, f'model_{object_number}.pkl')
    
    
    
    
    
    if os.path.exists(load_path):
        print('path exists')
        with open(load_path, "rb") as handle:
            # Use pickle.load() to deserialize the object
            Inference_Object = pickle.load(handle)
            
    
        _ = Inference_Object.append_simulations(train_params, train_data).train(force_first_round_loss=True,max_num_epochs= 300)
        posterior = Inference_Object.build_posterior()
        
        
        
        with open(load_path, "wb") as handle:
            pickle.dump(Inference_Object, handle)
            
            
    else: # Generate and train a new one
        print('Does not exist')
    
        density_estimator_build_fun = posterior_nn(
            model="zuko_nsf", hidden_features=60, num_transforms=10
        )
   
        Inference_Object = NPE(prior=prior, density_estimator = density_estimator_build_fun)
        
        _ = Inference_Object.append_simulations(train_params, train_data).train(force_first_round_loss=True,max_num_epochs= 300)
        posterior = Inference_Object.build_posterior()
        
        
        with open(load_path, "wb") as handle:
            pickle.dump(Inference_Object, handle)
        
        
        
    
    
    return posterior
    
    
    
    
def construct_ensemble_network(n_ensembles,prior):
    
    '''
    It randomly initializes n_ensembles networks of the same Type zuko_nsf
    
    If needed the following way of initialization allows for hyperparameter tuning.
    '''
    from sbi.neural_nets import posterior_nn

    ensemble_list = []
    
    for i in range(n_ensembles):

        density_estimator_build_fun = posterior_nn(
            model="zuko_nsf", hidden_features=60, num_transforms=10
        )
        
    
        inference = NPE(prior=prior, density_estimator=density_estimator_build_fun)
         
        ensemble_list.append(inference)
        
        
    return ensemble_list


def construct_network(n_ensembles,Save_path,prior):
    
    '''
    It randomly initializes n_ensembles networks of the same Type zuko_nsf
    
    If needed the following way of initialization allows for hyperparameter tuning.
    
    Generates a temporary folder in which these networks' internal states are saved.
    '''
    from sbi.neural_nets import posterior_nn

    ensemble_list = []
    
    for i in range(n_ensembles):

        density_estimator_build_fun = posterior_nn(
            model="zuko_nsf", hidden_features=60, num_transforms=10
        )
        
    
        inference = NPE(prior=prior, density_estimator = density_estimator_build_fun)
        
        
        
        save_path = os.path.join(Save_path, f'model_{i}.pkl')
        with open(save_path, "wb") as handle:
            pickle.dump(inference, handle)

       
        
        
    return ensemble_list

    
    
def ensemble_posterior(ensemble_list,train_params,train_data,x_0):

    '''
    ensemble_list: list of NPE networks to train
    train_data: list of simulation results linked to the train_params
    train_params: parameters used to generate the data in train_data
    x_0: target observation
    
    If possible the networks training will be done in parallel to optimize the 
    training procedure.
    
    Returns:
        1) The list of trained inference objects
        2) The ensenmble-averaged posterior built upon the x_0
    
    
    NOTES:
        1) when you use joblib.Parallel with an iterable of tasks, the order of the results 
            is guaranteed to match the order of the tasks in the input iterable.
    
    '''    
    # Get the number of trained inference objects
    
    n_ensembles = len(ensemble_list)
    
    
    # # Train the ensembles 
    # Output = Parallel(n_jobs=-1)(
    #         delayed(train_single_network)(ensemble_list[i], train_params, train_data) for i in range(n_ensembles))
    
    
    Output = []
    for i in range(n_ensembles):
        
        Output.append(train_single_network(ensemble_list[i], train_params, train_data))
    
    
    Output= np.vstack(Output)
    # Extract the trained ensemble list
    trained_ensemble_list = Output[:,0]
        
        
    # Create the ensemble-averaged posterior object
    ensemble_post = EnsemblePosterior(Output[:,1])
    ensemble_post.set_default_x(x_0)
    
    
    return trained_ensemble_list, ensemble_post


def ensemble_posterior_parallel(Save_dir,n_ensemble,train_params,train_data,x_0):

    '''
    ensemble_list: list of NPE networks to train
    train_data: list of simulation results linked to the train_params
    train_params: parameters used to generate the data in train_data
    x_0: target observation
    
    If possible the networks training will be done in parallel to optimize the 
    training procedure.
    
    Returns:
        1) The list of trained inference objects
        2) The ensenmble-averaged posterior built upon the x_0
    
    
    NOTES:
        1) when you use joblib.Parallel with an iterable of tasks, the order of the results 
            is guaranteed to match the order of the tasks in the input iterable.
    
    '''    
    # Get the number of trained inference objects

    
    # # Train the ensembles 
    Output = Parallel(n_jobs=5)(
            delayed(train_single_network_parallel)(Save_dir,i,train_params,train_data) for i in range(n_ensembles))
    
    

    
   
    # Create the ensemble-averaged posterior object
    ensemble_post = EnsemblePosterior(Output)
    ensemble_post.set_default_x(x_0)
    

    
    return  ensemble_post






#%

# Define the prior
num_dims = 2
num_sims = 1000
num_rounds = 3
prior = BoxUniform(low=torch.zeros(num_dims), high=torch.ones(num_dims))
simulator = lambda theta: theta + torch.randn_like(theta) * 0.1
x_0 = torch.tensor([0.5, 0.5])

import os
import pickle
save_dir = r'C:\Users\Admin\Desktop\Leonardo\SBI'

#%%
n_ensembles = 5

# Contruct ensemble network

proposal = prior # Same for all the models
#%

for _ in range(num_rounds):
    theta = proposal.sample((num_sims,))
    x = simulator(theta)
    
    ensemble_post = ensemble_posterior_parallel(save_dir,n_ensembles,theta,x,x_0)
    
    # ensemble_list, ensemble_post = ensemble_posterior(ensemble_list,theta,x,x_o)

    accept_reject_fn = get_density_thresholder(ensemble_post, quantile=1e-4,num_samples_to_estimate_support=10000)
    proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")
    
#%%    

# Evaluate the KL divergence to be sure of intrinsic variability wihtin the ensemble.

KL_div = check_posterior_diversity(ensemble_list, x_o, num_samples=10000)

# Plot

kl_matrix = np.zeros((n_ensembles, n_ensembles))
for (i, j), value in KL_div.items():
    kl_matrix[i, j] = value
    kl_matrix[j, i] = value  # The matrix is symmetric

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(kl_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
            xticklabels=[f"Posterior {i}" for i in range(n_ensembles)],
            yticklabels=[f"Posterior {i}" for i in range(n_ensembles)])

plt.title("Kullback-Leibler Divergence Between Posteriors", fontsize=16)
plt.xlabel("Posterior Index", fontsize=12)
plt.ylabel("Posterior Index", fontsize=12)
plt.tight_layout()
plt.show()
    
