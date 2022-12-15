# MLP_GNN_ensemble

Harnessing Shannon entropy of molecular symbols in deep neural networks to enhance prediction accuracy
------------------------------------------------------------------------------------------------------
This repository holds the codes for Fig. 3 of 'Harnessing Shannon entropy of molecular symbols in deep neural networks to enhance prediction accuracy'. 

Description
-----------
Hybrid of MLP and GNN models are compared in different architectural variations: (i) only MLP, (ii) 2D GNN, (iii) 3D GNN, (iv) MLP+2D GNN and (v) MLP+3D GNN. The specific purpose of the codes are described in the Notes section below. The basic dataset has been provided in the repository in the form of a .csv file.

Usage
-----
1. Download or make a clone of the repository
2. Make a new conda environment using the environment file 'mlp_gnn.yml'
3. Run the GNN or MLP-GNN hybrid python files directly using a python IDE or from command line

Example: python 3D_GNN_hybrid_mod_logp_with_MW_with_partial_shannon_in_MLP_-2_-4_layers.py

Notes
-----
1. The function files are: (i) mlp_cnn_gnn.py, (ii) sd_to_graph_2D_GNN.py, (iii) sd_to_graph_3D_GNN.py and (iv) sd_to_graph_3D_GNN_with_shannon_partial_shannon.py. Therefore, directly run the other python files apart from these. 
2. (i) For 2D GNN model, run: 2D_GNN_logp_with_shannon_partial_shannon.py
(ii) For 3D GNN model, run: 3D_GNN_logp_with_shannon_partial_shannon.py
(iii) For MLP-2D GNN hybrid model, run: 2D_GNN_hybrid_mod_logp_with_MW_with_partial_shannon_in_MLP_-2_-4_layers.py
(iv) For MLP-3D GNN hybrid model, run: 3D_GNN_hybrid_mod_logp_with_MW_with_partial_shannon_in_MLP_-2_-4_layers.py
3. Note that the folders Ki_sdf_to_graph and Ki_sdf_to_graph_2D would be populated while running 3D_GNN_hybrid_mod_logp_with_MW_with_partial_shannon_in_MLP_-2_-4_layers.py and 2D_GNN_hybrid_mod_logp_with_MW_with_partial_shannon_in_MLP_-2_-4_layers.py, respectively.
     
