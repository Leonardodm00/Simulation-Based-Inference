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
issue and build more reliable posterior estimators. n_ensembles = 10
Averaging probabilities by simply adding them up is incorrect. Instead, you should average their corresponding log-probabilities. The
function performing this is 'averaged_log_prob'.

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
The main informations can be found in the main simulation code of the network. A note about the synapses.
g_ampa and g_nmda are used to rule the relative density of receptors on the post-synaptic membrane. The amplitude 
is instead modulated by the Xi_ampa and Xi_nmda terms which allows for a [Glu]-dependent amplitude of the PSCs [7]




    
    
NOTES:
    1) Spline transformations are defined over the domain :math:`[-5, 5]`. Any feature outside of this domain is not transformed. 
        It is recommended to standardize features (zero mean, unit variance) before training.
        
    2) The feature-extraction and embedding network (1DCNN in my case) is trained on minibatches of a certain temporal length. 
        Even though longer/shorter batches can be used, the accuracy might be impacted. Thus use the very same simulation time.
        
    3) The emebdding vectors are generated by the emebdding network which is train upon the COSINE similarity loss function. This loss
       equals the L2 norm if the embedding vectors are NORMALIZED. SBI calculates the L2 norm on the summary statistics. Thus, 
       I shuld normalize the output vectors.
       
    4) In some combination of parameters the simulations are very likely to give rise to Nan or errors (this happens especially when delaing 
        with the synaptic Omega rates). We can approach this hurdle in the following way: the embedding vector's entires are all set to zero.
        Indeed this condition is never met from any other simulation based emebdding. We can also implement an external network that learns to 
        predict combination of parameters that give rise to unsuccesfull (bad) simulations.
       
        
    
    
    
    
    
    
    

References:
    
    [1] Truncated proposals for scalable and hassle-free simulation-based inference
    
    [2] A Trust Crisis In Simulation-Based Inference? Your  Posterior Approximations Can Be Unfaithful
    
    [3] https://github.com/sbi-dev/sbi/blob/main/sbi/neural_nets/net_builders/flow.py
    
    [4] Neural Spline Flows
    
    [5] https://sbi-dev.github.io/sbi/latest/tutorials/16_implemented_methods/

    [6] Towards constraining warm dark matter with stellar streams through neural simulation-based inference
    
    [7] Variability of Neurotransmitter Concentration and Nonsaturation of Postsynaptic AMPA Receptors at Synapses 
        in Hippocampal Cultures and Slices
    
    
    
Procedure:

1) First define the the biophysical model's parameters likely to be involved in the generation of the target phenotype.
    All the remaining ones will be fixed. Be aware that this results in a huge (but neccessary for computational reasons) bias
    in the search space.

2) Perform a coarse grid search on the free parameters.    
    
3) Use the intitial guess from the amortized network as initial priors for the sequential network.
    
    
    
    
TODO:
    
    Debug Smoothing CUmulative
    Control ho to deal with averaged posterior probabilities.
    
    
    GENERALIZE THE SEARCHING SPACE ACROSS MORE PARAMETERS SUCH AS:
        # Neuron
        0: Sigma [mV]   [1.5-7]
        1: g_AHP  (*msiemens*cm**-2) * Neuron_area [0 - 0.01]
        2: delta [-0.5;0.5]
        # 3: I_inj [pA] # DIffernet way to change the variable because it is internally initialized NO 
        
        # Synapse
        3: Xi_ampa [1/mmole] [0.01 - 3]
        4: Xi_nmda [1/mmole] [0.01 - 3]
        5: Y_T  [mmole]   [200 - 700]
        6: conn_prob     [0.1 - 0.6]
        # 7: Omega_d [1/second] [1.5 - 4]
        # 8: Omega_f_sr [1/second] [1.5 - 4]
        # 9: Omega_f_ar [1/second] [1 - 2.5]
        7 : SYn_type: 1-2-3
        8: U_0_sr [0.005,0.5]
        9: Uar [0 - 0.005]
        
    
    
    
    
    
    
    
    
    
    
    
    
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
import itertools

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
    # Clear the cache for the 'cython' code generation target
    # clear_cache('cython') 
    BrianLogger.suppress_hierarchy('brian2.devices')
    BrianLogger.suppress_hierarchy('brian2.parsing')

    start_scope()
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
        3: conn_prob
        4: Syn_type

    
    '''
    
    # ------------- UNPACK PARAMS -------------
    # Unpack Neuron parameters
    sigma_ = params[0]
    g_AHP_ = params[1]
    delta_ = params[2]
    conn_prob_ = params[3]
    Syn_type = params[4]

    
    
    
    

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
    
    
    # ------------ DEFINE THE SYNAPSE TYPE

    if Syn_type == 1:
        synapse_type = 'depressing'
        
    elif Syn_type == 2:
        synapse_type = 'facilitating'
        
    elif Syn_type == 3:
        synapse_type = 'neutral'
        

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
                         Decay_type = 'Double_exp',synapse_type = synapse_type,sed=sed)
    
    
    
    # ------------- UPDATE PARAMETERS -------------
    N.namespace['sigma'] = sigma_*mV
    N.namespace['g_AHP'] = (g_AHP_*msiemens*cm**-2) * N.namespace['area']
    N.namespace['delta'] = delta_
    S.namespace['conn_prob_'] = conn_prob_


    
    
    # Update the externally injected current
    N.I = '(rand() -0.5) * I_inj'          # Make neurons heterogeneously excitable
    
        
    MonitorN = StateMonitor(N, ['V','I_cell'], record=True)
    Spk_montor =  SpikeMonitor(N)
    
    net_ = Network(collect())  # automatically include all the stated groups
    net_.run(simtime)

    
        
    # --------------- ELECTRODE RECORDINGS ---------------
    Traces,MEA_dict = Electrode_traces(pitch,pitch_recsites,shift,N,MonitorN,electrode_dist,neuron_radius,electrode_radius)
    
    clock_dt = defaultclock.dt 
    fs = 1/(clock_dt/second)
    Raster,Raster_array = get_Raster(Traces,fs)
    
        
        
   
 
    return   MonitorN ,Spk_montor ,N,S
    
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
params = [0.0] * 10
       
# Update the values at the specific indices
# Neuron
params[0] = 4.1  # Sigma [mV]
params[1] = 0.005  # g_AHP
params[2] = 0.6  # delta
params[3] = 0.12 # Connection_probability
params[4] = 2


    

    
MonitorN,Spk_montor  ,N,S = Network_wrapper(params)


# ------------ Visualize the spiking activity ------------
plt.figure
plt.plot(Spk_montor.t/ms, Spk_montor.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Raster Plot')
plt.show()



#%%
# !!!: FINISH TO IMPLEMENT
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
    
#%%

# -------------------- GRID SEARCH --------------------
# Set up multiprocessing-based simulations.


'''
The core idea of this pipeline is to initially perform a coarsed grained grid 
search so that, given a observation we have an approximate clue which parameter
region could be of interest, by calculateing the posterior from the amortized network. 
Then from that posterior on, sequetial NPE is performed to refine the posterior
estimation.


 NOW WE EILL FOCUS ON ONLY 4 PARAMETERS


Params:
    
    # Neuron
    0: Sigma [mV]   [1.5-7]
    1: g_AHP  (*msiemens*cm**-2) * Neuron_area [0 - 0.01]
    2: delta [-0.5;0.5]
    3: conn_prob   [0.1 - 0.6]
    4: Syn_type [1,2,3] 1: depressing, 2: facilitating, 3: neutral
    

        

'''



if __name__ == "__main__":
    os.chdir(r"C:\Users\Admin\Desktop\Leonardo\SBI\Izh prova") 
    
    
    '''
    
    Reference trace
    
    '''
    theta = np.array([0.02, 0.2, -65, 8])  # 1D tensor
        
    X_0 = simulation_wrapper(theta)
        # # nwt = Izhikevich_simulatort(theta)
        
    #     # %matplotlib
    # plt.figure()
    # plt.plot(nw['T_vec'],nw['v'])
    #     # plt.plot(nw['T_vec'],nw['v'])
        
        
#% 
    
    '''

    AMORTIZATION

    '''

    
 
    Sigma_arr = np.arange(1.5, 6, 0.5)
    g_AHP_arr = np.arange(0.001, 0.007, 0.001)
    delta_arr = np.arange(-0.5, 0.5, 0.1)
    conn_prob_arr = np.arange(0.1, 0.6, 0.01)
    Syn_type = np.arange(1, 4, 1)
    
    
    
    
    # Xi_ampa_arr = np.arange(0.2, 1, 0.1)
    # Xi_nmda_arr = np.arange(0.2, 1, 0.1)
    # Y_T_arr = np.arange(300, 700, 100)
    
    # Syn_type = np.arange(1, 3, 1)
    # U_0_sr_arr = np.arange(0.05, 0.5, 0.05)
    # Uar_arr = np.arange(0, 0.005, 0.001)
    # Tau_AHP_arr = np.arange(400, 800, 50)
    
    # Create a single list of all the arrays
    all_arrays = [
        Sigma_arr, g_AHP_arr, delta_arr, conn_prob_arr,Syn_type
    ]
    
    # Generate the combinations using itertools.product
    # This creates a generator, not a list, which is memory-efficient
    combinations_generator = itertools.product(*all_arrays)
    

     
    with concurrent.futures.ProcessPoolExecutor() as executor:  #Use all the available resources
       
        Embeddings = list(executor.map(Network_wrapper,combinations_generator))
    
# 
    # Save 
    
    np.save("Embeddings.npy", np.float16(Embeddings))
    np.save("theta_0.npy", np.float16(tuple_list0))
    









    
    
    
    
    
    
    
    
    
