
'''
This function will be used to retrieve virtual electrode traces from Raster arrays (that is to say, variables of size nx2 in which along the first 
column there are spike timings in  seconds and in the second columna are stored the idxs of firing neurons). Moreover the neuron's position msut be kept
Large simulation batch rely on seeded random positioning of cells. This allows to always retrieve how those cells are layed out in the space.

Are required tamplates of membrane potentials and overall currents registered during a spike. 

The virtual electrodes algorithm will be changed to handle in input (and in processing) the Raster arrays and the tamplates referred to above.

V_template is a list comprising n numbers of snippets which are NOT neurons - specific.
Same for I_cell_template. Thus each simulated neurons has NOT it's own templates. It can be demonstrated that there are not stark differences between neurons


V_template expressed in mV
I_cell_template expressed in mA


Output traces are in MILLIVOLT (mV)


NOTES:
    1) If other type of neurons are simulated (inh/ different type of neurons, etc...) specific templates must be extracted.


'''
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.colors as mcolors
from sklearn.metrics import mean_squared_error
import numpy as np
# import torch
import scipy.io
import pdb
import os
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
from scipy.stats import skewnorm
from scipy.stats import chi2
import seaborn as sns
import math
import matplotlib.pyplot as plt
import random
import time
from brian2 import *

# -------------------- AUSILIARY FUNCTIONS --------------------




def generate_grid_points(n_rows, n_cols, pitch,x0,y0):
    """
    Generates the 2D coordinates for a grid of electrodes.

    Args:
        n_rows (int): Number of rows in the electrode grid.
        n_cols (int): Number of columns in the electrode grid.
        pitch_micrometers (float): The distance between adjacent electrodes
                                   in micrometers.
        x0,y0 = bottom left end coordinates.
    Returns:
        numpy.ndarray: A 2D array where each row represents the (x, y)
                       coordinates of an electrode point, in micrometers.
    """
    points = []
    # Pitch is used directly in micrometers as requested
    

    for r in range(n_rows):
        for c in range(n_cols):
            x = x0 + c * pitch
            y = y0 + r * pitch
            points.append((x, y))

    return np.array(points)

def Get_12grid(pitch):
    
    '''
    Generates a set of coordinates for the 12 electrode MEA configuration
    
      x x
    x x x x
    x x x x
      x x
 
    '''
    
    shift = 100 # [um]
    x0 = 0 + shift
    y0 = 0 + shift
    pitch = 300 # [um]
    
    Grid_raw = generate_grid_points(4, 4, pitch,x0,y0)
    
    
    # The configuration is 12 electrodes. For this reason we will
    # discard the following points: 0,3,12,15
    
    Idx_remove = np.array([0,3,12,15])
    
    Grid = np.delete(Grid_raw, Idx_remove, axis=0)
    
    return Grid
    




    
def Recording_sites(pitch_recsites,shift,Grid,n_rec=3,Visible= False,electrode_radius= 15):
    MEA_dict = {}
    
    el = 0
    for point in Grid: 
        
        if n_rec == 3:
        
            # The x0 and y0 are the bottom left coordinates of the first rec site.
            # the 'point' coordinate is the center. A shift in coordinates is needed.
            # The 'point' coordinates are shifted along the diagonal about half the diameter.
            # Both x and y of the 'point' are shifted about sqrt(2)*radius
            
            # x0 = point[0]-shift
            # y0 = point[1]-shift
            x0 = point[0]-pitch_recsites
            y0 = point[1]-pitch_recsites
            rec_points = generate_grid_points(n_rec, n_rec, pitch_recsites,x0,y0)
            MEA_dict[el] = np.array(rec_points)
            
           
            el = el+1
            
        elif n_rec == 4:
        
            # The x0 and y0 are the bottom left coordinates of the first rec site.
            # the 'point' coordinate is the center. A shift in coordinates is needed.
            # The 'point' coordinates are shifted along the diagonal about half the diameter.
            # Both x and y of the 'point' are shifted about sqrt(2)*radius
            
            x0 = point[0]-shift
            y0 = point[1]-shift

            rec_points = generate_grid_points(n_rec, n_rec, pitch_recsites,x0,y0)
            MEA_dict[el] = np.array(rec_points)
            
           
            el = el+1
        
    if Visible == True:
        # 2. Setup the plot
        fig, ax = plt.subplots()

        # Ensure the plot scales correctly to show the circles
        ax.set_aspect('equal', adjustable='box') 
        ax.autoscale_view()

        radius = electrode_radius
        # 3. Plot the circles
        for cn in range(len(Grid[:,0])):
            # Create a Circle patch
            circle = plt.Circle(
                (Grid[cn,0], Grid[cn,1]),  # Center (x, y)
                radius,                # Radius
                color='blue',          # Color of the circle
                alpha=0.3,             # Transparency (makes overlapping easier to see)
                fill=True,             # Fill the circle
                edgecolor='black',     # Color of the outline
                linewidth=1            # Thickness of the outline
            )
            
            # Add the circle to the Axes
            ax.add_patch(circle)
            
            # Optional: Plot the center point as a dot
            ax.plot(Grid[cn,0], Grid[cn,1], 'ro', markersize=5)
            
            
           
        for j in range(len(Grid[:,0])):
            
            rec_s = MEA_dict[j]
            
            for cn in range(len(rec_s[:,0])):
                
                
                # Optional: Plot the center point as a dot
                ax.plot(rec_s[cn,0], rec_s[cn,1], 'ro', markersize=10) 
                
        # 4. Set plot limits (important for seeing all circles)
        # Find the min/max coordinates and add a buffer equal to the radius
        min_x = np.min(Grid[:,0]) - radius * 1.5
        max_x = np.max(Grid[:,0]) + radius * 1.5
        min_y = np.min(Grid[:,1]) - radius * 1.5
        max_y = np.max(Grid[:,1]) + radius * 1.5

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        # 5. Add labels and title
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(f"Circles with Radius = {radius}")
        ax.grid(True)

        # 6. Show the plot
        plt.show()
    
        
        
    return MEA_dict


def Plot_CultureDevice(Grid,Cell_positions):
    
    tertiary_color_palette = [
    # Warm Tones
    (1.0, 0.647, 0.0),    # Orange (RGB 255, 165, 0)
    (1.0, 0.498, 0.314),  # Coral (RGB 255, 127, 80)
    (0.8, 0.0, 0.0),      # Dark Red / Maroon-ish (RGB 204, 0, 0) - Not pure Red (1,0,0)
    (0.627, 0.322, 0.176),# Sienna (RGB 160, 82, 45) - Earthy Brown
    (1.0, 0.753, 0.796),  # Pink (RGB 255, 192, 203)

    # Cool Tones
    (0.294, 0.0, 0.510),  # Indigo (RGB 75, 0, 130) - Deep Blue-Purple
    (0.502, 0.0, 0.502),  # Purple (RGB 128, 0, 128) - More vibrant Purple
    (0.251, 0.878, 0.816),# Turquoise (RGB 64, 224, 208) - Blue-Green
    (0.0, 0.502, 0.502),  # Teal (RGB 0, 128, 128)

    # Earthy/Muted Tones
    (0.502, 0.502, 0.0),  # Olive (RGB 128, 128, 0) - Muted Yellow-Green
    (0.439, 0.502, 0.565),# Slate Gray (RGB 112, 128, 144) - Muted Blue-Gray
    (0.753, 0.753, 0.0)   # Chartreuse (RGB 192, 192, 0) - Muted Yellow-Green
]
    
    Nn = len(Cell_positions[:,1])
    fig, ax = plt.subplots() # This creates both a figure and an axes for you
    # %matplotlib
    deep_pink = (1.0000, 0.0784, 0.5765)
    normalized_gold_rgb = (1.0, 215 / 255, 0.0)
    radius_el = 15 #[um]
    radius_cell = 9 #[um]
    
    
    col = 0
    for point in Grid:
        # point will be an array like [x, y]
        circle = plt.Circle((point[0], point[1]), radius_el, color=tertiary_color_palette[col], fill=True)
        
        ax.add_patch(circle) # Add each circle patch to the axes
        col = col+1
        
        
        
    x = Cell_positions[:,0]
    y = Cell_positions[:,1]
    
    for neu in range(Nn):
        
        circle = plt.Circle((x[neu], y[neu]), radius_cell, color='k', fill=True)
        ax.add_patch(circle) # Add each circle patch to the axes
        
    ax.set_aspect('equal', adjustable='box')
    min_x = np.min(Grid[:, 0]) - radius_el * 1.2 # Add some buffer
    max_x = np.max(Grid[:, 0]) + radius_el * 1.2
    min_y = np.min(Grid[:, 1]) - radius_el * 1.2
    max_y = np.max(Grid[:, 1]) + radius_el * 1.2

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plt.xlabel("[um]")
    plt.ylabel("[um]")
    plt.show()  


def Electrode_traces(pitch,pitch_recsites,shift,Raster_array,V_template,I_cell_template,Cell_positions,electrode_dist,
                     neuron_radius,electrode_radius,T_sim = None,fs_snippet = False,fs_raster = False, Visible = False):
    
    '''
    This version of the function retrieves the virtual electrode traces given the Raster array, templates of V_mem and 
    I_Cell of spiking neurons and neuron's positions
    
    
    MEA_dict = dictionary with electrodes info: position and recordin sites
    Traces = List of Lists that contain the recordings.
    
    Raster_array = 1st column = timing [s], 2nd column = idx.
    V_template = list of lists. Each list is the memebrane voltage time course in samples. The relative snippet's sampling
                 frequency is stored in fs_snippet. [mV]
    I_cell = as V_template. [mA]
    Cell_positions = nx2. Positions MUST be provided in micrometers.
    
    fs_raster = sampling frequency of the raster array and the one used to construct the final traces.
    
    fs_raster/fs_snippet are both defined in Hz. 
    
    T_sim = SImulation time [s]
    
    ASSUMPTIONS:
        - For now we will assume that fs_raster and fs_snippet are the same
    
    '''
    
    fs_raster = fs_raster/second
    
    Grid = Get_12grid(pitch)
    
    MEA_dict = Recording_sites(pitch_recsites,shift,Grid,n_rec = 3)
    
    # --- Plot Device + Neurons
    
    if Visible:
        
        Plot_CultureDevice(Grid,Cell_positions)
        
        
    # ----------- RESAMPLE TEMPLATES according to fs_raster -----------
    #!!! TODO
    
    Traces = Electrode_recording(MEA_dict,Raster_array,V_template,I_cell_template,Cell_positions,electrode_dist,
                         neuron_radius,electrode_radius,T_sim = T_sim,fs_raster = fs_raster)
    
    if Visible:
        t_vec = np.linspace(0,len(Traces[0]),len(Traces[0]))
        
        plt.figure()
        tertiary_color_palette = [
            # Warm Tones
            (1.0, 0.647, 0.0),    # Orange (RGB 255, 165, 0)
            (1.0, 0.498, 0.314),  # Coral (RGB 255, 127, 80)
            (0.8, 0.0, 0.0),      # Dark Red / Maroon-ish (RGB 204, 0, 0) - Not pure Red (1,0,0)
            (0.627, 0.322, 0.176),# Sienna (RGB 160, 82, 45) - Earthy Brown
            (1.0, 0.753, 0.796),  # Pink (RGB 255, 192, 203)
        
            # Cool Tones
            (0.294, 0.0, 0.510),  # Indigo (RGB 75, 0, 130) - Deep Blue-Purple
            (0.502, 0.0, 0.502),  # Purple (RGB 128, 0, 128) - More vibrant Purple
            (0.251, 0.878, 0.816),# Turquoise (RGB 64, 224, 208) - Blue-Green
            (0.0, 0.502, 0.502),  # Teal (RGB 0, 128, 128)
        
            # Earthy/Muted Tones
            (0.502, 0.502, 0.0),  # Olive (RGB 128, 128, 0) - Muted Yellow-Green
            (0.439, 0.502, 0.565),# Slate Gray (RGB 112, 128, 144) - Muted Blue-Gray
            (0.753, 0.753, 0.0)   # Chartreuse (RGB 192, 192, 0) - Muted Yellow-Green
        ]
    
        col = 0
        for ch in range(12):
            
            # if ch == 9:
            #     plt.plot(t_vec,Traces[ch]*0.1-np.mean(Traces[el])+ch*0.1,color = tertiary_color_palette[col])
            #     col = col+1
                
            # else:
                plt.plot(t_vec,Traces[ch]-np.mean(Traces[ch])+ch*0.1,color = tertiary_color_palette[col])
                col = col+1
        plt.show()
        
    
    return Traces,MEA_dict


def Electrode_recording(MEA_dict,Raster_array,V_template,I_cell_template,Cell_positions,electrode_dist,
                     neuron_radius,electrode_radius,T_sim = None,fs_raster = False):
    
    '''
    In the pre-processing we should already have casted the templates in fs_raster sampling frequency
    
    
    '''

    # Generate the KDTree form the neuronal position data
    
    pos = Cell_positions

    # Generate the KDTree
    Neuron_positions = KDTree(pos)
    
    
    Electrode_recordings = {}
    
    for key in MEA_dict.keys():
        
        rec_sites = MEA_dict[key]
        
        Electrode_rec = Electrode_trace(rec_sites,Raster_array,V_template,I_cell_template,Neuron_positions,electrode_dist,
                             neuron_radius,electrode_radius,T_sim = T_sim,fs_raster = fs_raster) 

        Electrode_recordings[key] = Electrode_rec
        
        
    return Electrode_recordings
    





def Electrode_trace(rec_sites,Raster_array,V_template,I_cell_template,Neuron_positions,electrode_dist,
                     neuron_radius,electrode_radius,T_sim = None,fs_raster = False):

    '''
    For neurons below d_lim a simplyfied EEI model is used
    For neurons below d_lim a Dipole approximation is implemented 
    
    Two contributes for background noise: 1) distant neurons, 2) White noise
    
    White noise: gaussian distribution of mean 0 and std 1uV --> 1e-3 mV
    
    For each recording site the sum of all the contributing neurons is taken and 
    for the whole electrode the mean across the recording sites.
    
    electrode_dist = max senstivity distance of the electrode
    Neuron_psoitions = sklearn.neighbors.NearestNeighbors object
    neuron_radius = radiu of the neurons expressed in micrometers.
            it is used to because the distance is calculated between centers
    electrode_radius = radius of the electrode  
    Raster_array = 1st column = timing [s], 2nd column = idx.
    V_template = list of lists. Each list is the memebrane voltage time course in samples. The relative snippet's sampling
                 frequency is stored in fs_snippet.
                 
    I_cell = as V_template. 

    fs_raster = sampling frequency of the raster array and the one used to construct the final traces.
    
    fs_raster/fs_snippet are both defined in Hz.
    
    T_sim = SImulation time [s]
    
    # --- Papers:
        
        1) Multi-program approach for simulating recorded extracellular signals
          generated by neurons coupled to microelectrode arrays.
          
        2) A Detailed and Fast Model of Extracellular Recordings.
    
    '''
    
    # For each recording site extract the recorded neurons
    d_lim = 50 # [um]
    White_noise = 1e-3 # [mV]
    Rho_s = 0.7 * 1e6 #[ Ohm * um ] Saline bath resistivity 
    Site_voltages = {}
    s = 0
    
    
    snippet_size = len(V_template[1])
    for site in rec_sites:
        # For each recording site extract the recorded neurons
        NN_idx,NN_dist = Neuron_positions.query_radius(site.reshape(1, -1), r=electrode_dist, return_distance=True)
        NN_idx = NN_idx[0]
        NN_dist = NN_dist[0]
        
        
        Voltages = []
        for neu in range(len(NN_idx)):
            
            # Retrieve the spike timings of the selected neuron
            spike_timings = Raster_array[np.where(Raster_array[:, 1] == NN_idx[neu])[0], 0] * second
            
            # Covert the timings in samples
            spike_timings = spike_timings * fs_raster
            
            # From the neuron's pool of templates choose one randomly
            N_templates = len(V_template)
           
           
        # First evaluate which model to use
        
            if NN_dist[neu] >= d_lim+neuron_radius+electrode_radius:
                
     
    
                
                if snippet_size % 2 == 0: # even num of samples in the snippet
                
                
                   V_temp = np.zeros(int(T_sim * fs_raster))
                
                   for spk_temp in spike_timings:
                       # Add spike snippets
                       Rand_template = V_template[np.random.randint(0, N_templates)]
                       V_temp[int(spk_temp) - int(snippet_size/2) : int(spk_temp) + int(snippet_size/2)] = Rand_template * 1/(NN_dist[neu]**2)
                   
                   Voltages.append(V_temp)
                   
                else:
                    
                    V_temp = np.zeros(int(T_sim * fs_raster))
                     
                    # Add spike snippets
                    for spk_temp in spike_timings:
                        Rand_template = V_template[np.random.randint(0, N_templates)]
                        V_temp[int(spk_temp) - int(snippet_size/2) - 1 : int(spk_temp) + int(snippet_size/2)] = Rand_template * 1/(NN_dist[neu]**2)
                    
                    Voltages.append(V_temp)
                    
       
                
            else:
                
                

                if snippet_size % 2 == 0: # even num of samples in the snippet
                
                   V_temp = np.zeros(int(T_sim * fs_raster))
                
                   # Add spike snippets
                   for spk_temp in spike_timings:
                       Rand_template = I_cell_template[np.random.randint(0, N_templates)]
                       V_temp[int(spk_temp) - int(snippet_size/2) : int(spk_temp) + int(snippet_size/2)] = (Rho_s*Rand_template)/(4*np.pi*NN_dist[neu])      
                   
                   Voltages.append(V_temp)
                   
                else:
                    
                    V_temp = np.zeros(int(T_sim * fs_raster))
                     
                    # Add spike snippets
                    for spk_temp in spike_timings:
                        Rand_template = I_cell_template[np.random.randint(0, N_templates)]
                        V_temp[int(spk_temp) - int(snippet_size/2) - 1 : int(spk_temp) + int(snippet_size/2)] = (Rho_s*Rand_template)/(4*np.pi*NN_dist[neu])      
                    
                    Voltages.append(V_temp)
                
            
                
                
                
                
                
                
            
        # Sum column-wise
        
        Voltages_sum = np.sum(np.array(Voltages),axis=0)
        
        # Save
        Site_voltages[s] = Voltages_sum
        s = s+1
                                          

    # --- MEAN ---
    
    # Extract traces
    rec_list = [Site_voltages[key] for key in Site_voltages.keys()]   
    rec_list_ = np.vstack(rec_list)
    
    # Take the mean
    Electrode_trace_ = np.mean(rec_list_, axis=0)
    
    # Add white noise
    white_noise_vector = np.random.normal(loc=0, scale=White_noise, size=len(Electrode_trace_))

    Electrode_trace_ = Electrode_trace_ + white_noise_vector
    return Electrode_trace_
def get_Raster(Traces,fs,low_f=200,high_f=2000,Visible=True):
    from scipy import signal

    from scipy.signal import find_peaks
    '''
    Alternatively an elliptic filter can be used.
    Elliptic filters offer the steepest possible rolloff between the passband and stopband for a given filter order.
    This makes them highly efficient for applications that require a sharp frequency cutoff. 
    However, this superior performance comes at the cost of ripples in both the passband and the stopband.
    
    
    Spike timings are defined in seconds
    
    Raster_array: 1st column channel 2nd column spk timings in SAMPLES
    
    '''
    
        # set up a filter to filter the voltage signal


    Wn = [2*low_f/fs, 2*high_f/fs]

    b, a = signal.butter(2, Wn, btype='bandpass')
    
    APs_time = []
    APs_unit = []
    # voltagetraces = zeros((len(Traces),len(Traces[0])))
    Raster = zeros((len(Traces),len(Traces[0])))
    Voltagefilt_array = []
    for k in range(len(Traces)):
        Trace_temp = Traces[k]
        # Subtract the mean
        Trace_temp = Trace_temp - np.mean(Trace_temp)
        Voltagefilt = signal.filtfilt(b, a, Trace_temp)  # filter
        Voltagefilt_array.append(Voltagefilt)
        threshold = np.mean(Voltagefilt) +  4 * np.std(Voltagefilt)      #threshold to detect APs
        APstemp, _ = find_peaks(abs(Voltagefilt), height=threshold)
        for j in range(len(APstemp)):
            
            
            
            APs_time = np.append(APs_time, APstemp[j])
            APs_unit = np.append(APs_unit,k)
        # voltagetraces[k, :] = Voltagefilt

       
        
        Raster[k,APstemp] = 1
        
        
    if Visible:
        t_vec = np.linspace(0,len(Traces[0]),len(Traces[0]))
        
        plt.figure()
        tertiary_color_palette = [
            # Warm Tones
            (1.0, 0.647, 0.0),    # Orange (RGB 255, 165, 0)
            (1.0, 0.498, 0.314),  # Coral (RGB 255, 127, 80)
            (0.8, 0.0, 0.0),      # Dark Red / Maroon-ish (RGB 204, 0, 0) - Not pure Red (1,0,0)
            (0.627, 0.322, 0.176),# Sienna (RGB 160, 82, 45) - Earthy Brown
            (1.0, 0.753, 0.796),  # Pink (RGB 255, 192, 203)
        
            # Cool Tones
            (0.294, 0.0, 0.510),  # Indigo (RGB 75, 0, 130) - Deep Blue-Purple
            (0.502, 0.0, 0.502),  # Purple (RGB 128, 0, 128) - More vibrant Purple
            (0.251, 0.878, 0.816),# Turquoise (RGB 64, 224, 208) - Blue-Green
            (0.0, 0.502, 0.502),  # Teal (RGB 0, 128, 128)
        
            # Earthy/Muted Tones
            (0.502, 0.502, 0.0),  # Olive (RGB 128, 128, 0) - Muted Yellow-Green
            (0.439, 0.502, 0.565),# Slate Gray (RGB 112, 128, 144) - Muted Blue-Gray
            (0.753, 0.753, 0.0)   # Chartreuse (RGB 192, 192, 0) - Muted Yellow-Green
        ]
    
        col = 0
        for ch in range(12):
            
            # if ch == 9:
            #     plt.plot(t_vec,Traces[ch]*0.1-np.mean(Traces[el])+ch*0.1,color = tertiary_color_palette[col])
            #     col = col+1
                
            # else:
            plt.plot(t_vec/fs,Voltagefilt_array[ch]-np.mean(Voltagefilt_array[ch])+ch,color = tertiary_color_palette[col])
            col = col+1

            
            indices = [i for i, x in enumerate(APs_unit) if x == ch]
        
            # Plot the unit indices (y-axis) against the spike times (x-axis)
            plt.scatter(APs_time[indices]/fs, APs_unit[indices]+0.05, s=5, marker='|',color = tertiary_color_palette[ch])
            
        # Customize the plot
        plt.title('Spiking Activity on filtered signal(Raster Plot)')
        plt.xlabel('Time (s)')
        plt.ylabel('Channel')
        
        plt.grid(True)
        plt.show()
        
    
    # Use zip to pair the elements from the two lists
    combined_list = list(zip(APs_unit, APs_time))
    
    
    # Convert the list of tuples to a numpy array
    Raster_array = np.array(combined_list)
            
                
    
    return Raster,Raster_array

spikes_dir = r'C:\Users\Admin\Desktop\Leonardo\ASN\Output Temp\Virtual electrodes reconstruction'
file_path_spk = os.path.join(spikes_dir, f"Spikes.npz")
file_path_VI = os.path.join(spikes_dir, f"VI_data.npz")
file_path_coord= os.path.join(spikes_dir, f"Coord.npy")

Coord = np.load(file_path_coord)
data = np.load(file_path_VI)

# Access the arrays by the names you gave them
v_loaded_array = data['v_data']
i_loaded_array = data['i_data']



#%

spks =  np.load(file_path_spk)
spks = spks['arr_0']
# --------- ELECTRODE RECORDINGS ----------- 
pitch = 300 #[um] 
electrode_radius = 15 #[um]

pitch_recsites = 10 # [um]  
shift = 11.25 # [um]

electrode_dist = 300 # [um]

c_min = 0 #[um]
c_max = 1100 #[um]






Grid = Get_12grid(pitch)

MEA_dict = Recording_sites(pitch_recsites,shift,Grid,n_rec = 3,Visible= False)


start_time = time.time()



Traces,MEA_dict = Electrode_traces(pitch,pitch_recsites,shift,spks,v_loaded_array,i_loaded_array,Coord,electrode_dist,
                     neuron_radius,electrode_radius,T_sim = 100*second,fs_snippet = (1/(0.05*ms)),fs_raster = (1/(0.05*ms)), Visible = False)




#%%
get_Raster(Traces, (1/(0.05*ms)),low_f=200,high_f=2000,Visible=True)
end_time = time.time()
print(f'Enalpsed time original fun: {end_time - start_time}')
