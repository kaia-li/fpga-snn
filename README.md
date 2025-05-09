# fpga-snn  
SNN-FPGA implementation based on NEST pre-trained  
*NEST is designed specifically for simulating biological neural systems and can accurately simulate the dynamics of spiking neurons. Using biologically inspired learning rules such as STDP instead of back propagation, which is more in line with the learning characteristics of real SNNs and has higher biological interpretability.* 
*In snn_classifier.py, bp-stdp tried.*
## SNN pre-trained  
Using NEST to build 2-layer SNN: input-hidden-output  
RF topology  
Weight saved  
## Weight transfer  
Transfer NEST-weight to FPGA format  
Optimizing the weight  
##  FPGA implementation  
Using FPGA for computing core: spike encoding, potential calculation, weight update  
*spike_encoder.v - Convert pixel values ​​to time-dependent spike trains*  
*stdp.v - Synaptic model that implements the STDP learning rule*  
*synapse_matrix.v - Manages connections and weight updates between neurons*  
*snn_compute.v - Integrates computational modules*  
### RF Topological structure implementation - sparse connections in the synapse_matrix module
Closer to the principles of biological neural networks, intuitive expression of spatial relationships  
Complex connections, high resource consumption, limited parallelism  
### RF Convolution implementation
Reduce weight storage requirements  
Increase computational parallelism  
Use DSP resources to accelerate convolution operations
