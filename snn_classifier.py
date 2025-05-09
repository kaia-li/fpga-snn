#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nest
import nest.topology as topo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
import time
from tqdm import tqdm
import pickle
import os
from scipy import ndimage

class MNIST_SNN_Classifier:
    def __init__(self, learning_rule='stdp', save_dir='./results'):
        """
        init SNN classifier

        params:
            learning_rule:  'stdp', 'r-stdp', 'bp-stdp'
        """
        self.learning_rule = learning_rule
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # nn params
        self.input_size = 28 * 28  # MNIST image size
        self.n_classes = 10        
        self.n_hidden = 100        
        
        # topo params
        self.input_rows = 28
        self.input_cols = 28
        self.receptive_field_radius = 5.0  # receptive field r
        self.center_radius = 1.0           # center region of center-surround
        self.surround_radius = 3.0         # surround region of center-surround
        self.n_orientations = 4            # number of orientation selectivity filters (0°, 45°, 90°, 135°)
        self.n_retina_cells = 16           # retinal ganglion cells per type (ON/OFF)
        self.n_lgn_cells = 16              # LGN cells per channel (P/M)
        self.n_v1_cells = 16               # V1 simple cells per orientation
        self.n_v2_cells = 64               # V2 complex cells
        self.n_v4_cells = 32               # V4 form recognition cells
        
        # learning params
        self.learning_rate = 0.01
        self.epochs = 5            
        self.batch_size = 100     
        self.sim_time = 100.0      
        self.reward_factor = 1.0   # R-STDP reward factor
        
        # neuron params
        self.neuron_params = {
            "V_th": -55.0,         
            "V_reset": -70.0,      
            "V_m": -70.0,          # Initial membrane potential
            "E_L": -70.0,          # Leak potential
            "t_ref": 2.0,          # Refractory period
            "tau_m": 10.0,         # Membrane time constant
            "tau_syn_ex": 2.0,     # Excitatory synaptic time constant
            "tau_syn_in": 2.0      # Inhibitory synaptic time constant
        }
        
        # retina neuron params - faster response
        self.retina_params = self.neuron_params.copy()
        self.retina_params.update({
            "tau_m": 5.0,          # Faster membrane time constant
            "tau_syn_ex": 1.0      # Faster synaptic time constant
        })
        
        # LGN neuron params - P and M channels
        self.lgn_p_params = self.neuron_params.copy()  # P-channel (color, detail)
        self.lgn_p_params.update({"tau_m": 8.0})
        
        self.lgn_m_params = self.neuron_params.copy()  # M-channel (motion, low spatial freq)
        self.lgn_m_params.update({"tau_m": 4.0})       # M-cells respond faster
        
        # STDP params
        self.stdp_params = {
            "lambda": 0.01,        # Learning rate
            "alpha": 2.0,          # Time window ratio
            "mu_plus": 1.0,        # LTP coefficient
            "mu_minus": 1.0,       # LTD coefficient
            "tau_plus": 20.0,      # LTP time constant
            "Wmax": 1.0,           # Max weight
            "weight": 0.1          # Initial weight
        }
        
        self.scaler = StandardScaler()
        
        # Create visual filters 
        self._create_visual_filters()
        
        # Initialize network
        self._initialize_network()
        
    def _create_visual_filters(self):
        """Create biologically-inspired visual filters"""
        # Create center-surround filters (ON and OFF cells)
        self.on_center_filter = self._create_center_surround_filter(
            center_sign=1, surround_sign=-1, 
            center_sigma=self.center_radius, 
            surround_sigma=self.surround_radius
        )
        
        self.off_center_filter = self._create_center_surround_filter(
            center_sign=-1, surround_sign=1, 
            center_sigma=self.center_radius, 
            surround_sigma=self.surround_radius
        )
        
        # Create orientation selective filters (V1 simple cells)
        self.orientation_filters = []
        for i in range(self.n_orientations):
            angle = i * (180.0 / self.n_orientations)
            orientation_filter = self._create_gabor_filter(
                angle=angle, 
                sigma=2.0, 
                wavelength=6.0, 
                size=7
            )
            self.orientation_filters.append(orientation_filter)
            
    def _create_center_surround_filter(self, center_sign, surround_sign, 
                                       center_sigma, surround_sigma, size=7):
        """Create center-surround receptive field filter"""
        # Create meshgrid for 2D Gaussian
        x, y = np.meshgrid(
            np.linspace(-size//2, size//2, size),
            np.linspace(-size//2, size//2, size)
        )
        
        # Center Gaussian
        center = np.exp(-(x**2 + y**2) / (2.0 * center_sigma**2))
        center = center / np.sum(center)
        
        # Surround Gaussian
        surround = np.exp(-(x**2 + y**2) / (2.0 * surround_sigma**2))
        surround = surround / np.sum(surround)
        
        # Combine center and surround
        cs_filter = center_sign * center + surround_sign * surround
        
        # Normalize to zero mean
        cs_filter = cs_filter - np.mean(cs_filter)
        
        return cs_filter
    
    def _create_gabor_filter(self, angle, sigma, wavelength, size=7):
        """Create Gabor filter for orientation selectivity"""
        # Convert angle to radians
        angle_rad = angle * np.pi / 180.0
        
        # Create meshgrid
        x, y = np.meshgrid(
            np.linspace(-size//2, size//2, size),
            np.linspace(-size//2, size//2, size)
        )
        
        # Rotate coordinates
        x_rot = x * np.cos(angle_rad) + y * np.sin(angle_rad)
        y_rot = -x * np.sin(angle_rad) + y * np.cos(angle_rad)
        
        # Create Gabor filter
        gabor = np.exp(-(x_rot**2 + y_rot**2) / (2.0 * sigma**2)) * np.cos(2.0 * np.pi * x_rot / wavelength)
        
        # Normalize
        gabor = gabor - np.mean(gabor)
        gabor = gabor / np.sqrt(np.sum(gabor**2))
        
        return gabor
        
    def _initialize_network(self):
        """Initialize SNN """
        # Reset NEST kernel
        nest.ResetKernel()
        nest.SetKernelStatus({"resolution": 0.1})
        
        # Input Layer (equivalent to photoreceptors)
        input_layer_params = {
            "extent": [self.input_rows, self.input_cols],  # Grid size
            "rows": self.input_rows,
            "columns": self.input_cols,
            "elements": "poisson_generator",
            "edge_wrap": False
        }
        self.input_layer = topo.CreateLayer(input_layer_params)
        
        # Retinal Ganglion Cells (RGC) - ON and OFF center cells
        retina_on_params = {
            "extent": [6.0, 6.0],  # Smaller grid for retinal cells
            "rows": 6,
            "columns": 6,
            "elements": "iaf_psc_alpha",
            "params": self.retina_params,
            "edge_wrap": False
        }
        self.retina_on_layer = topo.CreateLayer(retina_on_params)
        
        retina_off_params = retina_on_params.copy()
        self.retina_off_layer = topo.CreateLayer(retina_off_params)
        
        # Lateral Geniculate Nucleus (LGN) - P and M channels
        lgn_p_params = {  # Parvocellular pathway (high resolution, color)
            "extent": [5.0, 5.0],
            "rows": 5,
            "columns": 5,
            "elements": "iaf_psc_alpha",
            "params": self.lgn_p_params,
            "edge_wrap": False
        }
        self.lgn_p_layer = topo.CreateLayer(lgn_p_params)
        
        lgn_m_params = {  # Magnocellular pathway (motion, low resolution)
            "extent": [4.0, 4.0],
            "rows": 4,
            "columns": 4,
            "elements": "iaf_psc_alpha",
            "params": self.lgn_m_params,
            "edge_wrap": False
        }
        self.lgn_m_layer = topo.CreateLayer(lgn_m_params)
        
        # Primary Visual Cortex (V1) - orientation selectivity
        # Create one layer for each orientation
        self.v1_layers = []
        for i in range(self.n_orientations):
            v1_params = {
                "extent": [4.0, 4.0],
                "rows": 4,
                "columns": 4,
                "elements": "iaf_psc_alpha",
                "params": self.neuron_params,
                "edge_wrap": False
            }
            v1_layer = topo.CreateLayer(v1_params)
            self.v1_layers.append(v1_layer)
            
        # Secondary Visual Area (V2) - complex features
        v2_params = {
            "extent": [4.0, 4.0],
            "rows": 4,
            "columns": 4,
            "elements": "iaf_psc_alpha",
            "params": self.neuron_params,
            "edge_wrap": False
        }
        self.v2_layer = topo.CreateLayer(v2_params)
        
        # Visual Area V4 - shape recognition
        v4_params = {
            "extent": [4.0, 4.0],
            "rows": 4,
            "columns": 4,
            "elements": "iaf_psc_alpha",
            "params": self.neuron_params,
            "edge_wrap": False
        }
        self.v4_layer = topo.CreateLayer(v4_params)
        
        # output - classification
        output_layer_params = {
            "extent": [5.0, 2.0],  # 5x2 layout
            "rows": 5,
            "columns": 2,
            "elements": "iaf_psc_alpha",
            "params": self.neuron_params,
            "edge_wrap": False
        }
        self.output_layer = topo.CreateLayer(output_layer_params)
        
        # === Create connections between layers ===
        
        # Choose synapse model based on learning rule
        if self.learning_rule == 'stdp':
            syn_model = "stdp_synapse"
            syn_params = self.stdp_params
        elif self.learning_rule == 'r-stdp':
            # R-STDP
            nest.CopyModel("stdp_synapse", "r_stdp_synapse", 
                          {"A_plus": self.stdp_params["mu_plus"] * self.stdp_params["lambda"],
                           "A_minus": self.stdp_params["mu_minus"] * self.stdp_params["lambda"],
                           "Wmax": self.stdp_params["Wmax"],
                           "tau_plus": self.stdp_params["tau_plus"]})
            syn_model = "r_stdp_synapse"
            syn_params = {}
        elif self.learning_rule == 'bp-stdp':
            # BP-STDP with temporal dynamics
            bp_stdp_params = {
                "A_plus": self.stdp_params["mu_plus"] * self.stdp_params["lambda"],
                "A_minus": self.stdp_params["mu_minus"] * self.stdp_params["lambda"],
                "Wmax": self.stdp_params["Wmax"],
                "tau_plus": self.stdp_params["tau_plus"],
                "tau_minus": self.stdp_params["tau_plus"] * self.stdp_params["alpha"],
                "nearest_spike": True  # Consider nearest spike for pairing
            }
            nest.CopyModel("stdp_synapse", "bp_stdp_synapse", bp_stdp_params)
            syn_model = "bp_stdp_synapse"
            syn_params = {}
            
        # Input - Retinal Ganglion Cell connections
        # ON/OFF-center
        # ON-center RGC
        on_conn_params = {
            "connection_type": "divergent",
            "mask": {"circular": {"radius": self.receptive_field_radius}},
            "kernel": {"gaussian": {"p_center": 1.0, "sigma": 1.0}},
            "weights": {"uniform": {"min": 0.2, "max": 0.8}},
            "delays": 1.0,
            "synapse_model": "static_synapse"  # No plasticity at retinal level
        }
        topo.ConnectLayers(self.input_layer, self.retina_on_layer, on_conn_params)
        
        # OFF-center RGC
        off_conn_params = on_conn_params.copy()
        topo.ConnectLayers(self.input_layer, self.retina_off_layer, off_conn_params)
        
        # Retinal Ganglion Cells - LGN connections
        # RGC ON - LGN P (Parvocellular pathway for detail)
        rgc_on_to_lgn_p_params = {
            "connection_type": "divergent",
            "mask": {"circular": {"radius": 2.0}},
            "kernel": {"gaussian": {"p_center": 1.0, "sigma": 0.5}},
            "weights": {"uniform": {"min": 0.5, "max": 1.0}},
            "delays": 1.0,
            "synapse_model": "static_synapse"
        }
        topo.ConnectLayers(self.retina_on_layer, self.lgn_p_layer, rgc_on_to_lgn_p_params)
        
        # RGC OFF - LGN P
        rgc_off_to_lgn_p_params = rgc_on_to_lgn_p_params.copy()
        topo.ConnectLayers(self.retina_off_layer, self.lgn_p_layer, rgc_off_to_lgn_p_params)
        
        # RGC ON - LGN M (Magnocellular pathway for motion)
        rgc_on_to_lgn_m_params = {
            "connection_type": "divergent",
            "mask": {"circular": {"radius": 3.0}},  # Larger receptive field
            "kernel": {"gaussian": {"p_center": 1.0, "sigma": 1.0}},
            "weights": {"uniform": {"min": 0.5, "max": 1.0}},
            "delays": 1.0,
            "synapse_model": "static_synapse"
        }
        topo.ConnectLayers(self.retina_on_layer, self.lgn_m_layer, rgc_on_to_lgn_m_params)
        
        # RGC OFF - LGN M
        rgc_off_to_lgn_m_params = rgc_on_to_lgn_m_params.copy()
        topo.ConnectLayers(self.retina_off_layer, self.lgn_m_layer, rgc_off_to_lgn_m_params)
        
        # LGN - V1 connections (orientation selectivity)
        # Create connections to each V1 orientation-selective layer
        for i, v1_layer in enumerate(self.v1_layers):
            # LGN P - V1
            lgn_p_to_v1_params = {
                "connection_type": "divergent",
                "mask": {"circular": {"radius": 2.0}},
                "kernel": {"gaussian": {"p_center": 1.0, "sigma": 0.5}},
                "weights": {"uniform": {"min": 0.1, "max": 0.5}},
                "delays": 1.0,
                "synapse_model": syn_model
            }
            if syn_params:
                lgn_p_to_v1_params["synapse_parameters"] = syn_params
                
            topo.ConnectLayers(self.lgn_p_layer, v1_layer, lgn_p_to_v1_params)
            
            # LGN M - V1
            lgn_m_to_v1_params = lgn_p_to_v1_params.copy()
            topo.ConnectLayers(self.lgn_m_layer, v1_layer, lgn_m_to_v1_params)
        
        # V1 - V2 connections (feature integration)
        for v1_layer in self.v1_layers:
            v1_to_v2_params = {
                "connection_type": "divergent",
                "mask": {"circular": {"radius": 2.0}},
                "kernel": {"gaussian": {"p_center": 1.0, "sigma": 0.5}},
                "weights": {"uniform": {"min": 0.1, "max": 0.5}},
                "delays": 1.0,
                "synapse_model": syn_model
            }
            if syn_params:
                v1_to_v2_params["synapse_parameters"] = syn_params
                
            topo.ConnectLayers(v1_layer, self.v2_layer, v1_to_v2_params)
        
        # V2 - V4 connections (object recognition)
        v2_to_v4_params = {
            "connection_type": "divergent",
            "mask": {"circular": {"radius": 2.0}},
            "kernel": {"gaussian": {"p_center": 1.0, "sigma": 0.5}},
            "weights": {"uniform": {"min": 0.1, "max": 0.5}},
            "delays": 1.0,
            "synapse_model": syn_model
        }
        if syn_params:
            v2_to_v4_params["synapse_parameters"] = syn_params
            
        topo.ConnectLayers(self.v2_layer, self.v4_layer, v2_to_v4_params)
        
        # V4 - Output connections (classification)
        v4_to_output_params = {
            "connection_type": "divergent",
            "mask": {"rectangular": {"lower_left": [-2.5, -1.0], "upper_right": [2.5, 1.0]}},
            "weights": {"uniform": {"min": 0.1, "max": 0.5}},
            "delays": 1.0,
            "synapse_model": syn_model
        }
        if syn_params:
            v4_to_output_params["synapse_parameters"] = syn_params
            
        topo.ConnectLayers(self.v4_layer, self.output_layer, v4_to_output_params)
        
        # Create recording devices
        self.spike_detector = nest.Create("spike_detector", 
                                         params={"withgid": True, "withtime": True})
        
        # Connect recording devices to output layer
        output_nodes = nest.GetNodes(self.output_layer)[0]
        nest.Connect(output_nodes, self.spike_detector)
        
        # BP-STDP recording devices
        if self.learning_rule == 'bp-stdp':
            # Create spike detectors for each layer to track neural activity
            self.retina_on_spike_detector = nest.Create("spike_detector", 
                                                     params={"withgid": True, "withtime": True})
            self.retina_off_spike_detector = nest.Create("spike_detector", 
                                                      params={"withgid": True, "withtime": True})
            self.lgn_p_spike_detector = nest.Create("spike_detector", 
                                                  params={"withgid": True, "withtime": True})
            self.lgn_m_spike_detector = nest.Create("spike_detector", 
                                                  params={"withgid": True, "withtime": True})
            self.v1_spike_detectors = []
            for _ in range(self.n_orientations):
                self.v1_spike_detectors.append(nest.Create("spike_detector", 
                                                         params={"withgid": True, "withtime": True}))
            self.v2_spike_detector = nest.Create("spike_detector", 
                                               params={"withgid": True, "withtime": True})
            self.v4_spike_detector = nest.Create("spike_detector", 
                                               params={"withgid": True, "withtime": True})
            
            # Connect detectors to respective layers
            retina_on_nodes = nest.GetNodes(self.retina_on_layer)[0]
            retina_off_nodes = nest.GetNodes(self.retina_off_layer)[0]
            lgn_p_nodes = nest.GetNodes(self.lgn_p_layer)[0]
            lgn_m_nodes = nest.GetNodes(self.lgn_m_layer)[0]
            v2_nodes = nest.GetNodes(self.v2_layer)[0]
            v4_nodes = nest.GetNodes(self.v4_layer)[0]
            
            nest.Connect(retina_on_nodes, self.retina_on_spike_detector)
            nest.Connect(retina_off_nodes, self.retina_off_spike_detector)
            nest.Connect(lgn_p_nodes, self.lgn_p_spike_detector)
            nest.Connect(lgn_m_nodes, self.lgn_m_spike_detector)
            nest.Connect(v2_nodes, self.v2_spike_detector)
            nest.Connect(v4_nodes, self.v4_spike_detector)
            
            for i, v1_layer in enumerate(self.v1_layers):
                v1_nodes = nest.GetNodes(v1_layer)[0]
                nest.Connect(v1_nodes, self.v1_spike_detectors[i])
            
            # Store all connections for BP-STDP weight updates
            self.all_connections = {}
            
            # LGN to V1 connections
            for i, v1_layer in enumerate(self.v1_layers):
                v1_nodes = nest.GetNodes(v1_layer)[0]
                self.all_connections[f'lgn_p_to_v1_{i}'] = nest.GetConnections(
                    source=lgn_p_nodes, target=v1_nodes)
                self.all_connections[f'lgn_m_to_v1_{i}'] = nest.GetConnections(
                    source=lgn_m_nodes, target=v1_nodes)
            
            # V1 to V2 connections
            for i, v1_layer in enumerate(self.v1_layers):
                v1_nodes = nest.GetNodes(v1_layer)[0]
                self.all_connections[f'v1_{i}_to_v2'] = nest.GetConnections(
                    source=v1_nodes, target=v2_nodes)
            
            # V2 to V4 connections
            self.all_connections['v2_to_v4'] = nest.GetConnections(
                source=v2_nodes, target=v4_nodes)
            
            # V4 to output connections
            self.all_connections['v4_to_output'] = nest.GetConnections(
                source=v4_nodes, target=output_nodes)
    
    def load_mnist_data(self):

        print("Loading MNIST...")# CIFAR10...
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def _preprocess_image(self, img_array):

        img = img_array.reshape(28, 28)
        
        # Apply contrast enhancement (similar to retinal adaptation)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-9)
        
        # Apply center-surround filtering (similar to retinal ganglion cells)
        on_response = ndimage.convolve(img, self.on_center_filter, mode='constant')
        off_response = ndimage.convolve(img, self.off_center_filter, mode='constant')
        
        # Rectify responses (neurons can't have negative firing rates)
        on_response = np.maximum(0, on_response)
        off_response = np.maximum(0, off_response)
        
        # Apply orientation selective filtering (similar to V1 simple cells)
        orientation_responses = []
        for orientation_filter in self.orientation_filters:
            orientation_response = ndimage.convolve(img, orientation_filter, mode='constant')
            orientation_response = np.maximum(0, orientation_response)  # Rectify
            orientation_responses.append(orientation_response)
        
        return img, on_response, off_response, orientation_responses
    
    def _set_input_rates(self, processed_image_data):
        """Set firing rates for input layer based on processed image data"""
        original_img, on_response, off_response, orientation_responses = processed_image_data
        
        # Get nodes for input layer
        input_nodes = nest.GetNodes(self.input_layer)[0]
        
        # Convert original image to firing rates (0-200 Hz)
        original_rates = (original_img.flatten() * 200.0).tolist()
        
        # Set firing rates for input (photoreceptor) layer
        for i, rate in enumerate(original_rates):
            nest.SetStatus([input_nodes[i]], {"rate": rate})
        
    
    def _apply_reward(self, connections, reward):
        """Apply reward signal to R-STDP"""
        if self.learning_rule != 'r-stdp':
            return
            
        for conn in connections:
            w = nest.GetStatus([conn], 'weight')[0]
            if reward > 0:  
                new_w = w * (1.0 + self.learning_rate * reward)
            else:  
                new_w = w * (1.0 - self.learning_rate * abs(reward))
                
            new_w = max(0.0, min(new_w, self.stdp_params["Wmax"]))
            nest.SetStatus([conn], {'weight': new_w})
    
    def _apply_backprop(self, target, actual):
        """Apply BP-STDP backpropagation to the visual pathway"""
        if self.learning_rule != 'bp-stdp':
            return
            
        # Get current simulation time
        current_time = nest.GetKernelStatus('time')
        
        # Get spike events from all layers
        output_events = nest.GetStatus(self.spike_detector, 'events')[0]
        v4_events = nest.GetStatus(self.v4_spike_detector, 'events')[0]
        v2_events = nest.GetStatus(self.v2_spike_detector, 'events')[0]
        v1_events = [nest.GetStatus(detector, 'events')[0] 
                    for detector in self.v1_spike_detectors]
        lgn_p_events = nest.GetStatus(self.lgn_p_spike_detector, 'events')[0]
        lgn_m_events = nest.GetStatus(self.lgn_m_spike_detector, 'events')[0]
        
        # Reset spike detectors
        nest.SetStatus(self.spike_detector, {'n_events': 0})
        nest.SetStatus(self.v4_spike_detector, {'n_events': 0})
        nest.SetStatus(self.v2_spike_detector, {'n_events': 0})
        for detector in self.v1_spike_detectors:
            nest.SetStatus(detector, {'n_events': 0})
        nest.SetStatus(self.lgn_p_spike_detector, {'n_events': 0})
        nest.SetStatus(self.lgn_m_spike_detector, {'n_events': 0})
        
        # Calculate output error
        output_error = target - actual
        
        # Get nodes from each layer
        output_nodes = nest.GetNodes(self.output_layer)[0]
        v4_nodes = nest.GetNodes(self.v4_layer)[0]
        v2_nodes = nest.GetNodes(self.v2_layer)[0]
        v1_nodes = [nest.GetNodes(layer)[0] for layer in self.v1_layers]
        lgn_p_nodes = nest.GetNodes(self.lgn_p_layer)[0]
        lgn_m_nodes = nest.GetNodes(self.lgn_m_layer)[0]
        
        # Create error gradients for each output neuron
        output_gradients = {}
        for i, node_id in enumerate(output_nodes):
            if i < len(output_error):
                output_gradients[node_id] = output_error[i]
            else:
                output_gradients[node_id] = 0.0
        
        # Backward pass: propagate error from output to V4
        v4_gradients = self._backprop_gradients(
            v4_nodes, output_nodes, 
            self.all_connections['v4_to_output'],
            output_gradients
        )
        
        # V4 to V2
        v2_gradients = self._backprop_gradients(
            v2_nodes, v4_nodes,
            self.all_connections['v2_to_v4'],
            v4_gradients
        )
        
        # V2 to V1 (for each orientation channel)
        v1_gradients = []
        for i, nodes in enumerate(v1_nodes):
            v1_grad = self._backprop_gradients(
                nodes, v2_nodes,
                self.all_connections[f'v1_{i}_to_v2'],
                v2_gradients
            )
            v1_gradients.append(v1_grad)
        
        # Update weights with temporal dynamics (V4 to output)
        self._update_weights_with_timing(
            self.all_connections['v4_to_output'],
            v4_events, output_events,
            v4_gradients, output_gradients
        )
        
        # Update V2 to V4 weights
        self._update_weights_with_timing(
            self.all_connections['v2_to_v4'],
            v2_events, v4_events,
            v2_gradients, v4_gradients
        )
        
        # Update V1 to V2 weights for each orientation
        for i, v1_grad in enumerate(v1_gradients):
            self._update_weights_with_timing(
                self.all_connections[f'v1_{i}_to_v2'],
                v1_events[i], v2_events,
                v1_grad, v2_gradients
            )
    
    def _backprop_gradients(self, source_nodes, target_nodes, connections, target_gradients):
        """Backpropagate gradients from target to source layer"""
        source_gradients = {node_id: 0.0 for node_id in source_nodes}
        
        # Get connection information
        conn_info = {}
        for conn in connections:
            status = nest.GetStatus([conn])[0]
            source = status['source']
            target = status['target']
            weight = status['weight']
            
            if target in target_gradients:
                if source not in conn_info:
                    conn_info[source] = []
                conn_info[source].append((target, weight))
        
        # Backpropagate error
        for source in conn_info:
            for target, weight in conn_info[source]:
                source_gradients[source] += target_gradients[target] * weight
        
        # Apply sigmoid derivative (approximating neural transfer function)
        for node in source_gradients:
            # Get membrane potential as activation approximation
            v_m = nest.GetStatus([node], 'V_m')[0]
            v_th = self.neuron_params['V_th']
            v_reset = self.neuron_params['V_reset']
            
            # Normalize membrane potential
            v_norm = (v_m - v_reset) / (v_th - v_reset)
            v_norm = max(0.0, min(1.0, v_norm))
            
            # Apply sigmoid derivative: f'(x) = f(x) * (1 - f(x))
            sigmoid_derivative = v_norm * (1.0 - v_norm)
            source_gradients[node] *= sigmoid_derivative
        
        return source_gradients
    
    def _update_weights_with_timing(self, connections, pre_events, post_events, pre_gradients, post_gradients):
        """Update connection weights based on gradients and spike timing"""
        # Extract spike times and senders
        pre_times = pre_events['times']
        pre_senders = pre_events['senders']
        post_times = post_events['times']
        post_senders = post_events['senders']
        
        # Create lookup tables for spike times
        pre_spikes = {}
        for t, sender in zip(pre_times, pre_senders):
            if sender not in pre_spikes:
                pre_spikes[sender] = []
            pre_spikes[sender].append(t)
        
        post_spikes = {}
        for t, sender in zip(post_times, post_senders):
            if sender not in post_spikes:
                post_spikes[sender] = []
            post_spikes[sender].append(t)
        
        # Get STDP time constants
        tau_plus = self.stdp_params['tau_plus']
        tau_minus = tau_plus * self.stdp_params['alpha']
        
        # Update each connection
        for conn in connections:
            status = nest.GetStatus([conn])[0]
            source = status['source']
            target = status['target']
            weight = status['weight']
            
            # Skip neurons without gradients
            if source not in pre_gradients or target not in post_gradients:
                continue
            
            # Get gradient values
            pre_grad = pre_gradients[source]
            post_grad = post_gradients[target]
            
            # Calculate weight update
            delta_w = 0.0
            
            # Consider spike timing
            if source in pre_spikes and target in post_spikes:
                for t_pre in pre_spikes[source]:
                    for t_post in post_spikes[target]:
                        # Calculate spike time difference
                        delta_t = t_post - t_pre
                        
                        # Apply STDP window
                        if delta_t > 0:  # LTP: post fires after pre
                            stdp_factor = np.exp(-delta_t / tau_plus)
                            delta_w += self.learning_rate * post_grad * stdp_factor
                        else:  # LTD: pre fires after post
                            stdp_factor = np.exp(delta_t / tau_minus)
                            delta_w -= self.learning_rate * pre_grad * stdp_factor
            else:
                # If no spikes, use gradient only
                gradient_factor = post_grad * pre_grad
                if gradient_factor > 0:
                    delta_w = self.learning_rate * gradient_factor
                else:
                    delta_w = 0.1 * self.learning_rate * gradient_factor
            
            # Update weight
            new_w = weight + delta_w
            new_w = max(0.0, min(new_w, self.stdp_params["Wmax"]))
            nest.SetStatus([conn], {'weight': new_w})
    
    def _get_predicted_class(self):
        """Get predicted class based on output layer activity"""
        events = nest.GetStatus(self.spike_detector, 'events')[0]
        senders = events['senders']
        
        # Get output neurons
        output_nodes = nest.GetNodes(self.output_layer)[0]
        
        # Count spikes for each output neuron
        spike_counts = np.zeros(len(output_nodes))
        for sender in senders:
            try:
                idx = output_nodes.index(sender)
                spike_counts[idx] += 1
            except ValueError:
                pass
        
        # Return neuron with most spikes as predicted class
        if np.sum(spike_counts) == 0:
            return -1
        
        return np.argmax(spike_counts)
    
    def train(self, X_train, y_train):
        """Train SNN network using visual pathways"""
        print(f"Training with {self.learning_rule} learning rule...")
        
        accuracy_history = []
        n_samples = len(X_train)
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            correct = 0
            
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            
            for i in tqdm(range(0, n_samples, self.batch_size)):
                batch_indices = indices[i:min(i+self.batch_size, n_samples)]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]
                
                for j, (x, y) in enumerate(zip(batch_X, batch_y)):
                    # Reset spike detector
                    nest.SetStatus(self.spike_detector, {'n_events': 0})
                    
                    # Process image similar to human visual system
                    processed_data = self._preprocess_image(x)
                    
                    # Set input rates (equivalent to photoreceptor activity)
                    self._set_input_rates(processed_data)
                    

                    nest.Simulate(self.sim_time)
                    
                    # Get predicted class
                    predicted = self._get_predicted_class()
                    target = int(y)
                    
                    if predicted == target:
                        correct += 1
                    
                    # Apply learning based on selected rule
                    if self.learning_rule == 'r-stdp':
                        # Reward signal (1 for correct, -1 for incorrect)
                        reward = 1.0 if predicted == target else -1.0
                        
                        # Apply reward to plastic connections
                        # V4 to output connections
                        self._apply_reward(self.all_connections['v4_to_output'], reward)
                        
                        # V2 to V4 connections
                        self._apply_reward(self.all_connections['v2_to_v4'], reward * 0.8)
                        
                        # V1 to V2 connections
                        for i in range(self.n_orientations):
                            self._apply_reward(
                                self.all_connections[f'v1_{i}_to_v2'], 
                                reward * 0.6
                            )
                            
                    elif self.learning_rule == 'bp-stdp':
                        # Create one-hot target vector
                        target_vector = np.zeros(self.n_classes)
                        if target >= 0 and target < self.n_classes:
                            target_vector[target] = 1.0
                        
                        # Create actual output vector
                        actual_vector = np.zeros(self.n_classes)
                        if predicted >= 0 and predicted < self.n_classes:
                            actual_vector[predicted] = 1.0
                        
                        # Apply backpropagation
                        self._apply_backprop(target_vector, actual_vector)
                
                # Calculate batch accuracy
                batch_accuracy = correct / len(batch_indices) * 100
                print(f"Batch accuracy: {batch_accuracy:.2f}%")
                correct = 0
            
            # Evaluate on training data
            accuracy = self.evaluate(X_train[:1000], y_train[:1000])  # Use subset for speed
            accuracy_history.append(accuracy)
            print(f"Epoch {epoch+1} training accuracy: {accuracy:.2f}%")
        
        self._save_results(accuracy_history)
        return accuracy_history
    
    def evaluate(self, X_test, y_test):
        """Evaluate SNN performance"""
        print("Evaluating network performance...")
        
        correct = 0
        n_samples = len(X_test)
        
        for i in tqdm(range(n_samples)):
            # Reset spike detector
            nest.SetStatus(self.spike_detector, {'n_events': 0})
            
            # Process image through visual pathways
            processed_data = self._preprocess_image(X_test[i])
            self._set_input_rates(processed_data)
            
            # Run simulation
            nest.Simulate(self.sim_time)
            
            # Get prediction
            predicted = self._get_predicted_class()
            target = int(y_test[i])
            
            if predicted == target:
                correct += 1
        
        accuracy = correct / n_samples * 100
        print(f"Test accuracy: {accuracy:.2f}%")
        return accuracy
    
    def _save_results(self, accuracy_history):
        """Save training results"""
        results = {
            'learning_rule': self.learning_rule,
            'accuracy_history': accuracy_history,
            'final_accuracy': accuracy_history[-1] if accuracy_history else 0.0,
            'params': {
                'input_size': self.input_size,
                'n_classes': self.n_classes,
                'retina': {
                    'n_on_cells': self.n_retina_cells,
                    'n_off_cells': self.n_retina_cells
                },
                'lgn': {
                    'n_p_cells': self.n_lgn_cells,
                    'n_m_cells': self.n_lgn_cells
                },
                'v1': {
                    'n_orientations': self.n_orientations,
                    'cells_per_orientation': self.n_v1_cells
                },
                'v2': {
                    'n_cells': self.n_v2_cells
                },
                'v4': {
                    'n_cells': self.n_v4_cells
                }
            }
        }
        
        # Save results
        filename = f"{self.save_dir}/{self.learning_rule}_hvs_results.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {filename}")
        
        # Plot accuracy history
        plt.figure(figsize=(10, 6))
        plt.plot(accuracy_history)
        plt.title(f'HVS-SNN Classifier Accuracy ({self.learning_rule})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/{self.learning_rule}_hvs_accuracy.png")
        plt.close()

def visualize_receptive_fields(model, save_dir='./results'):
    """Visualize receptive fields of the network"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Visualize center-surround filters
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(model.on_center_filter, cmap='RdBu_r')
    plt.title('ON-Center Filter')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(model.off_center_filter, cmap='RdBu_r')
    plt.title('OFF-Center Filter')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/center_surround_filters.png")
    plt.close()
    
    # Visualize orientation filters
    plt.figure(figsize=(15, 4))
    for i, orientation_filter in enumerate(model.orientation_filters):
        plt.subplot(1, model.n_orientations, i+1)
        plt.imshow(orientation_filter, cmap='RdBu_r')
        angle = i * (180.0 / model.n_orientations)
        plt.title(f'Orientation {angle}°')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/orientation_filters.png")
    plt.close()
    
def visualize_visual_processing(model, image, save_dir='./results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    original, on_response, off_response, orientation_responses = model._preprocess_image(image)
    
    # Plot original image
    plt.figure(figsize=(6, 6))
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.colorbar()
    plt.savefig(f"{save_dir}/original_image.png")
    plt.close()
    
    # Plot retinal responses
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(on_response, cmap='hot')
    plt.title('ON-Center Response')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(off_response, cmap='hot')
    plt.title('OFF-Center Response')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/retinal_responses.png")
    plt.close()
    
    # Plot orientation responses
    plt.figure(figsize=(15, 4))
    for i, orientation_response in enumerate(orientation_responses):
        plt.subplot(1, model.n_orientations, i+1)
        plt.imshow(orientation_response, cmap='hot')
        angle = i * (180.0 / model.n_orientations)
        plt.title(f'Orientation {angle}°')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/orientation_responses.png")
    plt.close()

def visualize_network_activity(model, spike_data, save_dir='./results'):
    """Visualize spike activity in different visual areas"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Plot raster plot of spike times
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    labels = ['Retina ON', 'Retina OFF', 'LGN P', 'LGN M', 'V1', 'V2', 'V4', 'Output']
    
    for i, (spikes, label, color) in enumerate(zip(spike_data, labels, colors)):
        if 'times' in spikes and len(spikes['times']) > 0:
            plt.scatter(spikes['times'], 
                       [i] * len(spikes['times']), 
                       c=color, label=label, s=10, alpha=0.7)
    
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Time (ms)')
    plt.title('Spike Activity in Visual Pathways')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/spike_activity.png")
    plt.close()

def visualize_connectivity(model, save_dir='./results'):
    """Visualize connectivity patterns between visual areas"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create a schematic of the network architecture
    plt.figure(figsize=(12, 8))
    
    # Define node positions
    positions = {
        'input': (0, 0),
        'retina_on': (1, 0.5),
        'retina_off': (1, -0.5),
        'lgn_p': (2, 0.5),
        'lgn_m': (2, -0.5),
        'v1_0': (3, 0.75),
        'v1_1': (3, 0.25),
        'v1_2': (3, -0.25),
        'v1_3': (3, -0.75),
        'v2': (4, 0),
        'v4': (5, 0),
        'output': (6, 0)
    }
    
    # Node sizes based on neuron counts
    sizes = {
        'input': 5000,
        'retina_on': 2000,
        'retina_off': 2000,
        'lgn_p': 1500,
        'lgn_m': 1500,
        'v1_0': 1000,
        'v1_1': 1000,
        'v1_2': 1000,
        'v1_3': 1000,
        'v2': 3000,
        'v4': 2000,
        'output': 1000
    }
    
    # Node colors
    colors = {
        'input': 'lightblue',
        'retina_on': 'lightgreen',
        'retina_off': 'lightgreen',
        'lgn_p': 'pink',
        'lgn_m': 'pink',
        'v1_0': 'orange',
        'v1_1': 'orange',
        'v1_2': 'orange',
        'v1_3': 'orange',
        'v2': 'purple',
        'v4': 'red',
        'output': 'yellow'
    }
    
    # Node labels
    labels = {
        'input': 'Input',
        'retina_on': 'Retina ON',
        'retina_off': 'Retina OFF',
        'lgn_p': 'LGN P',
        'lgn_m': 'LGN M',
        'v1_0': 'V1 (0°)',
        'v1_1': 'V1 (45°)',
        'v1_2': 'V1 (90°)',
        'v1_3': 'V1 (135°)',
        'v2': 'V2',
        'v4': 'V4',
        'output': 'Output'
    }
    
    # Draw nodes
    for node, pos in positions.items():
        plt.scatter(pos[0], pos[1], s=sizes[node], c=colors[node], 
                   alpha=0.7, edgecolors='black', zorder=5)
        plt.text(pos[0], pos[1], labels[node], 
                ha='center', va='center', fontsize=10)
    
    # Define connections
    connections = [
        ('input', 'retina_on'),
        ('input', 'retina_off'),
        ('retina_on', 'lgn_p'),
        ('retina_off', 'lgn_p'),
        ('retina_on', 'lgn_m'),
        ('retina_off', 'lgn_m'),
        ('lgn_p', 'v1_0'),
        ('lgn_p', 'v1_1'),
        ('lgn_p', 'v1_2'),
        ('lgn_p', 'v1_3'),
        ('lgn_m', 'v1_0'),
        ('lgn_m', 'v1_1'),
        ('lgn_m', 'v1_2'),
        ('lgn_m', 'v1_3'),
        ('v1_0', 'v2'),
        ('v1_1', 'v2'),
        ('v1_2', 'v2'),
        ('v1_3', 'v2'),
        ('v2', 'v4'),
        ('v4', 'output')
    ]
    
    # Draw connections
    for source, target in connections:
        plt.plot([positions[source][0], positions[target][0]],
                [positions[source][1], positions[target][1]],
                'k-', alpha=0.5, zorder=1)
    
    plt.xlim(-0.5, 6.5)
    plt.ylim(-1.5, 1.5)
    plt.title('Human Visual System Network Architecture')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/network_architecture.png")
    plt.close()

def main():
    """Main function to run HVS-based SNN classifier"""
    # Create save directory
    save_dir = './hvs_snn_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load data once
    X_train, X_test, y_train, y_test = None, None, None, None
    
    # Train models with different learning rules
    learning_rules = ['stdp', 'r-stdp', 'bp-stdp']
    results = {}
    
    for rule in learning_rules:
        print(f"\n===== Training {rule} model =====")
        model = MNIST_SNN_Classifier(learning_rule=rule, save_dir=save_dir)
        
        # Load data only once
        if X_train is None:
            X_train, X_test, y_train, y_test = model.load_mnist_data()
        
        # Visualize receptive fields and network architecture
        visualize_receptive_fields(model, save_dir)
        visualize_connectivity(model, save_dir)
        
        # Visualize visual processing for a sample image
        sample_idx = np.random.randint(0, len(X_test))
        visualize_visual_processing(model, X_test[sample_idx], save_dir)
        
        # Train the model
        # Use smaller subset for faster training
        history = model.train(X_train[:3000], y_train[:3000])
        
        # Evaluate model
        accuracy = model.evaluate(X_test[:1000], y_test[:1000])
        
        results[rule] = {
            'history': history,
            'final_accuracy': accuracy
        }
    
    # Compare performance of different learning rules
    plt.figure(figsize=(12, 8))
    for rule, result in results.items():
        plt.plot(result['history'], label=rule)
    
    plt.title('HVS-SNN Classifier Performance Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/hvs_learning_comparison.png")
    plt.close()
    
    # Print final results
    print("\n===== Final Results =====")
    for rule, result in results.items():
        print(f"{rule}: {result['final_accuracy']:.2f}%")
    
    # Save comparison results
    with open(f"{save_dir}/comparison_results.pkl", 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
