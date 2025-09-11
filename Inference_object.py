"""
Created on Thu Sep 11 10:54:10 2025

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 08:40:59 2025

@author: Admin
"""

'''

#############----------- SBI -----------#############

This code implements a more verstile Neuronal Posterior Estimator (NPE) within the SBI framework. Are implemented both an amortized 
and a non-amortized version. Indeed the foundamental idea is to initially train an amortized ensemble network. Given a target observation
initial prior distribution is obtain from it. Afterward the sequential paradigm is used. 

----------- AMORTIZED NETWORK -----------

This network is based on Neural Spline Flow architecture. An ensemble of such networks is trained through grid search in a deep manner as suggested
by Deistler et al (2022)[2]. The number of networks making up the ensemble is arbitrarly choosen. Be aware that the performance increases with the number 
of networks whitin the ensemble. In this setting the Simulation-based calibration (SBC) is used as both a diagnostic tool of the posterior estimation itself
and as way to alleviate the issue of overconfidence as done by Hermans et al. (2020) [6]







----------- SEQUENTIAL NETWORK -----------

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



#############----------- EMBEDDING NETWORK -----------#############

The embedding network is based on the ResNet architecture made up of 1DCNN blocks. The architecture features as well as its training
have been optimized to enhance its performance on the given data type. It's recommended to retrain the network on the specific target 
phenotype. The optimization paradigm might not be neccessary to repeat. 



#############----------- SIMULATOR -----------#############
Brian2 backed simulator of a coupled neruonal and astrocytic network. 





    
    
NOTES:
    1) Spline transformations are defined over the domain :math:`[-5, 5]`. Any feature outside of this domain is not transformed. 
        It is recommended to standardize features (zero mean, unit variance) before training.
        
    2) The feature-extraction and embedding network (1DCNN in my case) is trained on minibatches of a certain temporal length. 
        Even though longer/shorter batches can be used, the accuracy might be impacted. Thus use the very same simulation time.
        
    3) The emebdding vectors are generated by the emebdding network which is train upon the COSINE similarity loss function. This loss
       equals the L2 norm if the embedding vectors are NORMALIZED. SBI calculates the L2 norm on the summary statistics. Thus, 
       I shuld normalize the output vectors.
       
        
    
    
    
    
    
    
    

References:
    
    [1] Truncated proposals for scalable and hassle-free simulation-based inference
    
    [2] A Trust Crisis In Simulation-Based Inference? Your  Posterior Approximations Can Be Unfaithful
    
    [3] https://github.com/sbi-dev/sbi/blob/main/sbi/neural_nets/net_builders/flow.py
    
    [4] Neural Spline Flows
    
    [5] https://sbi-dev.github.io/sbi/latest/tutorials/16_implemented_methods/

    [6] Towards constraining warm dark matter with stellar streams through neural simulation-based inference
    
    
    
Procedure:

1) First define the the biophysical model's parameters likely to be involved in the generation of the target phenotype.
    All the remaining ones will be fixed. Be aware that this results in a huge (but neccessary for computational reasons) bias
    in the search space.

2) Perform a coarse grid search on the free parameters.    
    
3) Use the intitial guess from the amortized network as initial priors for the sequential network.
    
    
    
    
TODO:
    
    Debug Smoothing CUmulative
    
    
    
    
    
    
    
    
    
    
    
'''

import numpy as np
from sbi.neural_nets.net_builders import build_zuko_nsf
import torch
from sbi.inference import NPE
from sbi.utils import RestrictedPrior, get_density_thresholder
from sbi.utils import BoxUniform

from sbi.inference import NPE, NRE
from sbi.analysis import pairplot


import matplotlib.pyplot as plt
from brian2 import *

from brian2 import clear_cache


import os


os.chdir(r'C:\Users\Admin\Desktop\Leonardo\ASN')
import ASN_fun
# # Define the prior
# num_dims = 2
# num_sims = 3000
# num_rounds = 3
# prior = BoxUniform(low=torch.zeros(num_dims), high=torch.ones(num_dims))
# simulator = lambda theta: theta + torch.randn_like(theta) * 0.01
# x_o = torch.tensor([0.5, 0.5])


# # Generate some samples:
# theta = prior.sample((num_sims,))
# x = simulator(theta)


# prior = BoxUniform(torch.zeros(2), torch.ones(2))

# from sbi.neural_nets import posterior_nn

# density_estimator_build_fun = posterior_nn(
#     model="zuko_nsf", hidden_features=60, num_transforms=10
# )





# inference = NPE(prior=prior, density_estimator=density_estimator_build_fun)
# proposal = prior

# # Sequential approach:
    
# for _ in range(num_rounds):
#     theta = proposal.sample((num_sims,))
#     x = simulator(theta)
#     _ = inference.append_simulations(theta, x).train(force_first_round_loss=True)
#     posterior = inference.build_posterior().set_default_x(x_o)

#     accept_reject_fn = get_density_thresholder(posterior, quantile=1e-4)
#     proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")


# #%
# %matplotlib
# x_o = torch.tensor([0.5, 0.5])
# print(f"Shape of x_o: {x_o.shape}            # Must have a batch dimension")

# samples = posterior.sample((1000,), x=x_o)
# print(f"Shape of samples: {samples.shape}  # Samples are returned with a batch dimension.")

# samples = samples.squeeze(dim=1)
# print(f"Shape of samples: {samples.shape}     # Removed batch dimension.")
# fig, ax = pairplot(
#     samples, limits=[[0, 1], [0, 1]], figsize=(5, 5)
# )



def Network_wrapper(params):
    
    '''
    This wrapper simulates the neuronal and astrocytic network based on the parameters given 
    in the Params vector.
    The type of network used is a biophysical network.
    Parameters changed are the ones that we belive are involved in the phenotype we try to 
    reproduce.
    
    Params:
        
        # Neuron
        0: Sigma [mV]
        1: g_AHP  (*msiemens*cm**-2) * Neuron_area
        2: delta 
        3: I_inj [pA] # DIffernet way to change the variable because it is internally initialized
        
        # Synapse
        4: Xi_ampa [1/mmole]
        5: Xi_nmda [1/mmole]
        6: Y_T  [mmole]
        7: conn_prob
        8: Omega_d [1/second]
        9: Omega_f_sr [1/second]
        10: Omega_f_ar [1/second]
        11: U_0_sr
        12: Uar
            

    
    '''
    
    # ------------- UNPACK PARAMS -------------
    # Unpack Neuron parameters
    sigma_ = params[0]
    g_AHP_ = params[1]
    delta_ = params[2]
    I_inj_ = params[3]
    
    # Unpack Synapse parameters
    Xi_ampa_ =   params[4]
    Xi_nmda_ = params[5]
    Y_T_ = params[6]
    conn_prob_ = params[7]
    Omega_d_ = params[8]
    Omega_f_sr_ = params[9]
    Omega_f_ar_ = params[10]
    U_0_sr_ = params[11]
    Uar_ = params[12]
    
    
    
    # Clear the cache for the 'cython' code generation target
    # clear_cache('cython') 
    BrianLogger.suppress_hierarchy('brian2.devices')
    BrianLogger.suppress_hierarchy('brian2.parsing')

    start_scope()
    

    # ------------------------- PARAMETERS -------------------------

    # --------- SIMULATION -----------
    simtime = 10 * second               # simulation time must be coherent with the 1D_CNN training one.
    # transient = 3 * second              # time omitted as transient
    sed = 39                             # random number seed
    devices.device.seed(sed)            # set the seed for all the random number realisations
    Simulated_network = 'Neuronal'
    # --------- NEURON -----------
    Nn = 100
    neuron_radius = 9 #[um]
    #

   

    # --------- ELECTRODE RECORDINGS ----------- 
    pitch = 300 #[um] 
    electrode_radius = 15 #[um]

    pitch_recsites = 7.5 # [um]  
    shift = 11.25 # [um]

    electrode_dist = 300 # [um]

    c_min = 0 #[um]
    c_max = 1100 #[um]


    # ------------------------- GROUPS BUILD-UP -------------------------


    # --------- NEURON and SYNAPSE -----------
    N,S = Neuronal_Network(Nn,Connection_var = 'Random', 
                        add_delay= False,delay_mode= 'random',
                         Max_Delay = 10*ms,ics = False, Simulated_network = 'Neuronal',
                         Decay_type = 'Double_exp',synapse_type = 'neutral',sed=sed)
    
    
    
    # ------------- UPDATE PARAMETERS -------------
    N.namespace['sigma'] = sigma_*mV
    N.namespace['g_AHP'] = (g_AHP_*msiemens*cm**-2) * N.namespace['area']
    N.namespace['delta'] = delta_
    N.namespace['I_inj'] = I_inj_ * pA
    
    S.namespace['Xi_ampa'] = Xi_ampa_ / mmole
    S.namespace['Xi_nmda'] = Xi_nmda_ / mmole
    S.namespace['Y_T'] = Y_T_ * mmole
    S.namespace['conn_prob_'] = conn_prob_
    S.namespace['Omega_d'] = Omega_d_ / second
    S.namespace['Omega_f_sr'] = Omega_f_sr_ / second
    S.namespace['Omega_f_ar'] = Omega_f_ar_ / second
    S.namespace['U_0_sr'] = U_0_sr_
    S.namespace['Uar'] = Uar_
    
    
    # Update the externally injected current
    N.I = '(rand() -0.5) * I_inj'          # Make neurons heterogeneously excitable
    
        
    MonitorN = StateMonitor(N, ['V','I_cell'], record=True)
    net_ = Network(collect())  # automatically include all the stated groups
    net_.run(simtime)

    
        
    # --------------- ELECTRODE RECORDINGS ---------------
    Traces,MEA_dict = Electrode_traces(pitch,pitch_recsites,shift,N,MonitorN,electrode_dist,neuron_radius,electrode_radius)
    
    clock_dt = defaultclock.dt 
    fs = 1/(clock_dt/second)
    Raster,Raster_array = get_Raster(Traces,fs)
    
        
        
   
 
    return   MonitorN  ,N,S
    
neuron_radius = 9 #[um]
#



# --------- ELECTRODE RECORDINGS ----------- 
pitch = 300 #[um] 
electrode_radius = 15 #[um]

pitch_recsites = 7.5 # [um]  
shift = 11.25 # [um]

electrode_dist = 300 # [um]

c_min = 0 #[um]
c_max = 1100 #[um]
 
    
 





    
# Initialize a list with placeholder values.
# The number of elements corresponds to the total number of parameters.
params = [0.0] * 13
       
# Update the values at the specific indices
# Neuron
params[0] = 4.1  # Sigma [mV]
params[1] = 0.005  # g_AHP
params[2] = 0.6  # delta
params[3] = 10.0  # I_inj [pA]

# Synapse
params[4] = 0.5  # Xi_ampa [1/mmole]
params[5] = 0.3  # Xi_nmda [1/mmole]
params[6] = 500.0  # Y_T [mmole]
params[7] = 0.10  # conn_prob
params[8] = 3  # Omega_d [1/second]
params[9] = 3  # Omega_f_sr [1/second]
params[10] = 1.1428  # Omega_f_ar [1/second]
params[11] = 0.6  # U_0_sr
params[12] = 0.001 # Uar    

    

    
MonitorN  ,N,S = Network_wrapper(params)

# --------------- RUN NETWORK ---------------

#%%


    #%%


#%%

# ----- CUMULATIVE TRACES ------
smoothed_cumulative,fs_downsampled = Neuronal_traces_simulation(Raster_array,fs = fs,Visible = True,w_size=0.02,Gaussian_window=0.04)


# cumulative_stdz = Standardization(smoothed_cumulative)

# data_array =  torch.unsqueeze(torch.from_numpy(cumulative_stdz),0).float32()


    
    # # ----- EMBEDDING NETWORK ------
    # device = torch.device("cpu") # for multiprocessing 
    
    # # Load data to device 
    # data_array = data_array.to(device)
    
    # # Load the model
    # embedding_network = torch.load("C:\Users\Admin\Desktop\Leonardo\Summary Networks\Saved networks\250825\Main_Model_.pt")
    # embedding_network = embedding_network.to(device)

    # embedding_network.eval() # Always set the model to evaluation mode for inference
    # with torch.no_grad():
        
    #     # In embedding mode the network expects the input data to be (1,length)
    #     Emb = embedding_network(data_array,fs_downsampled,State='Embedding')
        
        
    # return Emb

        
    
    
    
    
    
    
    
    
    
    
    
