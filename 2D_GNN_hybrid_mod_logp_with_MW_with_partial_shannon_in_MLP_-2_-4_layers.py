### This script could run hybrid MLP-GNN models sweeping over layer connections : (-2,-2), (-2,-3) and (-2,-4). The current setting is for (-2,-4) layer connections.

# Importing necessary packages
import time
start_time = time.time()


# Importing necessary conformer generation & graph building packages
from openbabel import pybel
import networkx as nx

import pandas as pd
import matplotlib.pyplot as plt

# Importing necessary rdkit packages
from rdkit import Chem

# Importing graph generation functions
from sd_to_graph_2D_GNN import obabel_to_networkx3d_graph, shannon_entropy_smiles

import random
import shutil
import os
import numpy as np

# Importing tensorflow related packages, models and functions
from mlp_cnn_gnn import mlp_cnn_gnn

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# Importing fingerprinting package if MHFP fingerprint is used
from mhfp.encoder import MHFPEncoder

# Stellargraph related packages
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph import StellarGraph

import scipy
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error


####------------------------------------------Saving the target as a csv file-----------------------------------------------------------------------------------------------------
## Getting the list of molecules from 
Ki_data_list = pd.read_csv('Ki_target_CHEMBL5023_only_equal_less_than_10K_dot_smiles_removed.csv', encoding='cp1252') 
Ki_list = Ki_data_list['Smiles'].values

new_df_mol = Ki_data_list
# Extracting the SMILES corresponding to the index values
df_SMILES = new_df_mol['Smiles'].values

# length of df_SMILES strings
len_df_SMILES = []

for i in range(len(df_SMILES)):
    
    len_df_SMILES.append(len(df_SMILES[i]))

## Plotting the length of len_df_SMILES histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(len_df_SMILES, bins = 10)
plt.title('SMILES string length Distribution')
plt.xlabel('length of SMILES')
plt.show() 

# Need to filter out SMILES strings beyond 500 characters length for comparing it to 3D case
df_SMILES_mod = []
for i in range(len(df_SMILES)):
    
    if len_df_SMILES[i] <= 500:
        df_SMILES_mod.append(df_SMILES[i])
        
# Reading the SMILES strings using pybel
read_mols_all_mod = [pybel.readstring("smi", x) for x in df_SMILES_mod]

## Processing the targets which are logP values
target_logP_mod = []
for i in range(0,len(df_SMILES_mod)):
    read_mols_all_mod[i].addh()
    
    ## The calcdesc() method of a Molecule returns a dictionary containing descriptor values for LogP, Polar Surface Area (“TPSA”) and Molar Refractivity (“MR”)
    descvalues = read_mols_all_mod[i].calcdesc()
    target_logP_mod.append(descvalues["logP"])

y_val_mod = pd.DataFrame(target_logP_mod, columns = ["logP"])    

### Saving the featuredataframe as a csv
y_val_mod.to_csv('Ki_data_list_logP_mod.csv', index=False)       


# This is the part to save ALL the target values in a csv file - keep it commented as we will be using only smiles strings <=500 characters
###-------------------------------------------------------------------------------------------------------------------------------------------
# ## Reading the SMILES strings using pybel
# read_mols_all = [pybel.readstring("smi", x) for x in df_SMILES]

# ## Processing the targets
# target_logP = []
# for i in range(0,len(df_SMILES)):
#     read_mols_all[i].addh()
    
#     ## The calcdesc() method of a Molecule returns a dictionary containing descriptor values for LogP, Polar Surface Area (“TPSA”) and Molar Refractivity (“MR”)
#     descvalues = read_mols_all[i].calcdesc()
#     target_logP.append(descvalues["logP"])

# y_val = pd.DataFrame(target_logP, columns = ["logP"])    
# # # ## Saving the featuredataframe as a csv
# y_val.to_csv('IC50_data_list_logP.csv', index=False)

###-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

for i in range(0,len(df_SMILES_mod)):
    
    smiles = df_SMILES_mod[i]
     
    mol = Chem.MolFromSmiles(df_SMILES[i])
    mol = Chem.AddHs(mol)


    # Adding H's here while reading the smiles
    input_mol = pybel.readstring("smi", smiles)
    input_mol.OBMol.AddHydrogens()
    
    # nx graph with H's: input_mol corresponds to the molecule in pybel and mol corresponds to the molecule in rdkit (H atoms are added in both cases)
    graph = obabel_to_networkx3d_graph(input_mol, mol)
    
    
    # saving graph created above in gml format, note: THIS TAKES LARGE SPACE TO SAVE ON DISK, USER MAY CONSIDER GRAPH GENERATION AND TRAINING ON-THE-FLY
    nx.write_gml(graph, "graph_min_en_conf_2D{}.gml".format(i))
    shutil.move("graph_min_en_conf_2D{}.gml".format(i), "Ki_sdf_to_graph_2D/mol{}.gml".format(i))
    
    
    # #Constructing a stellargraph with all the above info and checking the nodes, edges and adjacency
    
    # g = graph
    # ## getting the feature attribute of graph g
    # g_feature_attr = StellarGraph.from_networkx(g, node_features="feat")
    # ## g_feature_attr = StellarGraph.from_networkx(g)
    # ##print(g_feature_attr.info())
    
    # ## Getting the new stellar graph as g_stellar
    # g_stellar = g_feature_attr
    
    # ## Getting node features in a matrix
    # # print(g_stellar.node_features())
    
    # ## Getting edge features in a matrix
    # # print(g_stellar.edge_features())
    
    # ## Getting the adjacency matrix with weights
    # # print(g_stellar.to_adjacency_matrix(weighted = True))
    
    # ## The output graph list
    # graph_data_list.append(g_stellar)    

# ###----------------------------------------------------------------Constructing Stellar graphs from saved nx graphs : PLEASE COMMENT THE ABOVE NX GRAPH CONSTRUCTION SECTION-----------------------------------------------------------------------------

stellar_graph_list = []
for i in range(0,len(df_SMILES_mod)):
    
    g_nx = nx.read_gml("Ki_sdf_to_graph_2D/mol{}.gml".format(i))
    g_sg = StellarGraph.from_networkx(g_nx, node_features="feat")
    stellar_graph_list.append(g_sg)
    
###----------------------------------Total Process Time-------------------------------------------------------------------------------------------------------
end_time = time.time()
print("The duration of run (in s) so far: ", end_time - start_time)

###-----------------------------------------------------------------------------------------------------------------------------------------------------------
    
# #-------------------------------------------------------------------------------------------------------------------------------------------------------------------

graphs = stellar_graph_list

graphs_summary = pd.DataFrame( [ ( g.number_of_nodes(), g.number_of_edges() ) for g in graphs] , columns = ["nodes", "edges"]  )
    
print(graphs_summary.describe().round(1))

# display the graph_labesl
graph_labels = y_val_mod
print(" Original labels: ", graph_labels)

# # Converting the labels into 0 or 1
# graph_labels = pd.get_dummies(graph_labels, drop_first = True)
# print(" After conversion: ", graph_labels)

# #-------------------------------------------------------------------------------------------------------------------------------------------------------------------

### In GNN, # of nodes can differ per graph, but node features should have the same dimension. Else, will get the error: "graphs: expected node features for all graph to have same dimensions". The PaddedGraphGenerator
### helps to address this problem
generator = PaddedGraphGenerator(graphs = graphs)

# #-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ### Uncomment if atomic number and atomic charge are to be used as descriptors
# # Constructing the padded array of atomic numbers per molecule
# def atomic_num_padding(atomic_num_list, max_len_smiles):
    
#     len_atomic_num_list = len(atomic_num_list)
    
#     len_forward_padding = int((max_len_smiles - len_atomic_num_list)/2)
#     len_back_padding = max_len_smiles - len_forward_padding - len_atomic_num_list
    
#     atomic_num_padded = list(np.zeros(len_forward_padding))  + list(atomic_num_list) + list(np.zeros(len_back_padding))
    
#     return atomic_num_padded 

# # Constructing the padded array of Gasteiger-Marseli atomic partial charges per molecule
# def atomic_charge_padding(atomic_charge_list, max_len_smiles):
    
#     len_atomic_charge_list = len(atomic_charge_list)
    
#     len_forward_padding = int((max_len_smiles - len_atomic_charge_list)/2)
#     len_back_padding = max_len_smiles - len_forward_padding - len_atomic_charge_list
    
#     atomic_charge_padded = list(np.zeros(len_forward_padding))  + list(atomic_charge_list) + list(np.zeros(len_back_padding))
    
#     return atomic_charge_padded

# Constructing the padded array of fractional or partial Shannon entropy (SMILES) per molecule
def ps_padding(ps, max_len_smiles):
    
    len_ps = len(ps)
    
    len_forward_padding = int((max_len_smiles - len_ps)/2)
    len_back_padding = max_len_smiles - len_forward_padding - len_ps
    
    ps_padded = list(np.zeros(len_forward_padding))  + list(ps) + list(np.zeros(len_back_padding))
    
    return ps_padded 


# generating a dictionary of atom occurrence frequencies
def freq_atom_list(atom_list_input_mol):
    
    atom_list = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I'] 
    dict_freq = {}
    
    # adding keys
    for i in range(len(atom_list)):
        dict_freq[atom_list[i]] = 0  ### The values are all set 0 initially

    
    # update the value by 1 when a key in encountered in the string
    for i in range(len(atom_list_input_mol)):
        dict_freq[ atom_list_input_mol[i] ] = dict_freq[ atom_list_input_mol[i] ] + 1
    
    # The dictionary values as frequency array
    freq_atom_list =  list(dict_freq.values())/ (  sum(  np.asarray (list(dict_freq.values()))  )    )
    
    # Getting the final frequency dictionary
    # adding values to keys
    for i in range(len(atom_list)):
        dict_freq[atom_list[i]] = freq_atom_list[i]  
        
    freq_atom_list = dict_freq
        
    return freq_atom_list
  
  
# Maximum length of SMILES strings in the database
len_smiles = []
for j in range(0,len(df_SMILES)):
    
    mol = Chem.MolFromSmiles(df_SMILES[j])
    mol = Chem.AddHs(mol)
    
    k=0
    for atom in mol.GetAtoms():
        k = k +1 
    len_smiles.append(k)

    
max_len_smiles = max(len_smiles)

# Getting the Shannon framework & MW to use as descriptor array
shannon_arr = []
fp_combined = []
MW_list = []

for i in range(0,len(df_SMILES)):  

    
  mol_smiles = df_SMILES[i]
  
  # ## Uncomment if MHFP fingerprint needs to be added: SECFP (SMILES Extended Connectifity Fingerprint)
  # ## SECFP (SMILES Extended Connectifity Fingerprint)
  # fp = MHFPEncoder.secfp_from_smiles(in_smiles = mol_smiles, length=2048, radius=3, rings=True, kekulize=True, sanitize=False)
  
  mol = Chem.MolFromSmiles(df_SMILES[i])
  mol = Chem.AddHs(mol)
  
  MW_list.append( Chem.rdMolDescriptors.CalcExactMolWt(mol) )
  
  # estimating the fractional or partial Shannon entropy (SMILES) for an atom type => the current node
  total_shannon = shannon_entropy_smiles(df_SMILES[i])
  shannon_arr.append( total_shannon )
  
  # The atom list as per rdkit in string form
  atom_list_input_mol = []
  for atom_rdkit in mol.GetAtoms():
     atom_list_input_mol.append(str(atom_rdkit.GetSymbol()))     
        
     
  freq_list_input_mol = freq_atom_list(atom_list_input_mol)
  
  ps = []
  for atom_rdkit in mol.GetAtoms():
      atom_symbol = atom_rdkit.GetSymbol()
      atom_type = atom_symbol ### atom symbol in atom type
      
      partial_shannon = freq_list_input_mol[atom_type] * total_shannon
      ps.append(  partial_shannon )

  # padded fractional or partial Shannon entropy (SMILES)
  ps_arr = ps_padding(ps, max_len_smiles)     
  fp_combined.append(ps_arr)


# partial shannon_entropy as feature
fp_mol = pd.DataFrame(fp_combined)

### MW as feature
MW_mol= pd.DataFrame(MW_list)

# ### converting the shannon_arr list to dataframe
shannon_list = pd.DataFrame(shannon_arr)

# concatenating MW_mol, fp_mol and shannon_list to feed into MLP
fp_mol = pd.concat([ MW_mol, shannon_list, fp_mol], axis = 1)


# # concatenating MW_mol and fp_mol to feed into MLP: No total shannon in MLP
# fp_mol = pd.concat([ MW_mol, fp_mol], axis = 1)


# #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Defining model parameters for GNN training
epochs = 500
batch_size = 10

# defining early stopping
es = EarlyStopping(
    
                    monitor = "val_loss", min_delta = 0, patience = 500, restore_best_weights = True
                   
                  )
#----------------------------------------------------------------------GNN: training/ testing split------------------------------------------------------------------------------------------------------------------------------

# Prepare the dataset for training the GNN
def get_generators(train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
    )

    return train_gen, test_gen


# Getting the train, test & indices using train_test_split
indices = np.arange(len(graphs))

### Splitting both graphs & images along with targets/ labels: ## training & testing of CNN set should be as per GNN indices: train_index, test_index
X_train, X_test, train_index, test_index, XtrainTotalData, XtestTotalData,XtrainLabels, XtestLabels = train_test_split(graphs, indices, fp_mol, y_val_mod, test_size=0.15,random_state=10)


train_gen, test_gen = get_generators(train_index, test_index, graph_labels, batch_size=10)


###------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# grab the maximum/ minimum price in the target columm
maxPrice = y_val_mod.iloc[:,-1].max() # grab the maximum price in the training set's last column
minPrice = y_val_mod.iloc[:,-1].min() # grab the minimum price in the training set's last column
print(maxPrice,minPrice)


# Data corresponds to MLP input
XtrainData = XtrainTotalData 
XtestData = XtestTotalData

# perform min-max scaling each continuous feature column to the range [0 1]
cs = MinMaxScaler()
trainContinuous = cs.fit_transform(XtrainData)
testContinuous = cs.transform(XtestData)

print("[INFO] processing input data after normalization....")
XtrainData, XtestData = trainContinuous,testContinuous


# print("Xtrain feature data shape",XtrainData.shape)
# print("Xtest feature data shape",XtestData.shape)
# print("Xtrain label shape",XtrainLabels.shape)
# print("Xtest label data shape",XtestLabels.shape)
# print("Shape of the logP labels array", y_val_mod.shape)


trainY = XtrainLabels
testY = XtestLabels

# Converting the target column in np array
trainY = np.asarray(trainY)
testY = np.asarray(testY)


###--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#####----------------------------------------------------------Constructing the GNN model-------------------------------------------------------------------------------------------------------

# GNN model input: graphs data to train/ test
graphs_train = []
for k in range(0, len(train_index)):
    graphs_train.append(graphs[train_index[k]])
    
graphs_test = []
for k in range(0, len(test_index)):
    graphs_test.append(graphs[test_index[k]])

# GNN model input    
testing_data_gnn = generator.flow(graphs_test)
training_data_gnn = generator.flow(graphs_train)    
    
# Defining the model instance
gnn_model =mlp_cnn_gnn.create_gnn(generator)

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt  = Adam(lr = 0.00015, decay = 0)
gnn_model.compile(loss = "mean_absolute_percentage_error", optimizer = opt)

epoch_number = 500
BS = 10;

## Model fitting across epochs
## shuffle = False to reduce randomness and increase reproducibility
H = gnn_model.fit(train_gen, epochs = epochs, validation_data = test_gen, verbose = 2, shuffle = False, callbacks = [es])


# to get the penultimate layer from gnn_model
layer_name_gnn = []
for layer in gnn_model.layers:
    print(layer.name)
    layer_name_gnn.append(layer.name)
      
# Selecting the penultimate layer
layer_name = layer_name_gnn[-2]
penultimate_layer_model_gnn = Model(inputs=gnn_model.input,
                                  outputs=gnn_model.get_layer(layer_name).output)

# Getting the output tensors of the penultimate gnn layer
penultimate_output_gnn_train = penultimate_layer_model_gnn.predict(training_data_gnn)
penultimate_output_gnn_test = penultimate_layer_model_gnn.predict(testing_data_gnn)

## Check the dtype & shape
print("gnn_model penultimate layer data type: ", penultimate_output_gnn_train.dtype)
print("gnn_model penultimate layer shape: ", penultimate_output_gnn_train.shape)


# Selecting the previous layer to the penultimate layer
layer_name = layer_name_gnn[-3]
penultimate_layer_model_gnn_prev = Model(inputs=gnn_model.input,
                                  outputs=gnn_model.get_layer(layer_name).output)

# Getting the input tensors to the penultimate gnn layer which are output from the previous gnn layer
penultimate_output_gnn_prev_train = penultimate_layer_model_gnn_prev.predict(training_data_gnn)   
penultimate_output_gnn_prev_test = penultimate_layer_model_gnn_prev.predict(testing_data_gnn)

# Check the dtype & shape
print("gnn_model penultimate layer data type: ", penultimate_output_gnn_prev_train.dtype)
print("gnn_model penultimate layer shape: ", penultimate_output_gnn_prev_train.shape)

##----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#####----------------------------------------------------------Constructing the MLP model-------------------------------------------------------------------------------------------------------

# create the hybrid MLP model
mlp = mlp_cnn_gnn.create_mlp(XtrainData.shape[1], regress = False) # the input dimension to mlp would be shape[1] of the matrix i.e. column features
print("shape of mlp", mlp.output.shape)


# The final input to the last layer will be output from MLP lyers. Just define them as combinedInput
combinedInput = mlp.output
print("shape of combinedInput",combinedInput.shape)

# Final FC (Dense) layers 
x = Dense(100, activation = "relu") (combinedInput)
x = Dense(16, activation = "relu") (combinedInput)  ### This (none, 16) dense dimension for concatenating it to GNN
x = Dense(1, activation = "linear") (x)
print("shape of x", x.shape)


# final MLP model definition
model = Model(inputs = mlp.input, outputs = x)


# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = SGD(lr= 1.02e-6, decay = 0)
model.compile(loss = "mean_absolute_percentage_error", optimizer = opt)

#train the network
print("[INFO] training network...")

epoch_number = 500
BS = 10;

# Model fitting across epochs
# Defining the early stop to monitor the validation loss to avoid overfitting.
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=epoch_number, verbose=1, mode='auto')

## shuffle = False to reduce randomness and increase reproducibility
# H1 = model.fit( x = [XtrainData , trainX], y = trainY, validation_data = ( [XtestData, testX], testY), batch_size = BS, epochs = epoch_number, verbose=1, shuffle=False, callbacks = [early_stop]) 
H1 = model.fit( x = XtrainData , y = trainY, validation_data = ( XtestData, testY), batch_size = BS, epochs = epoch_number, verbose=1, shuffle=False, callbacks = [early_stop]) 


#### Getting the penultimate layer output from CNN

## MLP-CNN model inputs: training/ testing
# testing_data = [ XtestData , XImagetest ]
# training_data = [ XtrainData , XImagetrain]

testing_data =  XtestData  
training_data = XtrainData  


# to get the penultimate layer of mlp_cnn model (which is acually an MLP-based model)
layer_name_mlp_cnn = []
for layer in model.layers:
    print(layer.name)
    layer_name_mlp_cnn.append(layer.name)
      
##Selecting the penultimate layer
layer_name = layer_name_mlp_cnn[-2]
penultimate_layer_model_mlp_cnn = Model(inputs=model.input,
                                  outputs=model.get_layer(layer_name).output)

penultimate_output_mlp_cnn_train= penultimate_layer_model_mlp_cnn.predict(training_data)
penultimate_output_mlp_cnn_test = penultimate_layer_model_mlp_cnn.predict(testing_data)


# Selecting the previous layers (L = -2 or L = -3 or L = -4) to the penultimate layer: Here L = -4 is chosen
L = -4
layer_name = layer_name_mlp_cnn[L]
penultimate_layer_model_mlp_cnn_prev = Model(inputs=model.input,
                                  outputs=model.get_layer(layer_name).output)

# Getting the input tensors to the penultimate mlp_cnn layer
penultimate_output_mlp_cnn_prev_train = penultimate_layer_model_mlp_cnn_prev.predict(training_data)
penultimate_output_mlp_cnn_prev_test = penultimate_layer_model_mlp_cnn_prev.predict(testing_data)


##------------------------------------------------------------------------------Training the hybrid MLP-GNN model---------------------------------------------------------------------------------------------------------------

# The final input to our last layer will be concatenated output from both MLP and GNN lyers
combinedInput_final = concatenate([penultimate_layer_model_mlp_cnn.output, penultimate_layer_model_gnn.output])
print("shape of combinedInput",combinedInput_final.shape)

# Final FC (Dense) layers 
x = Dense(100, activation = "relu")(combinedInput_final)
x = Dropout(0.0001)(x) 
x = Dense(1, activation = "linear") (x)
print("shape of x", x.shape)

# final model instance: the inputs are the outputs of the previous layer of the penultimate layers of both MLP & GNN-based models
model_final = Model(inputs = [penultimate_layer_model_mlp_cnn_prev.output, penultimate_layer_model_gnn_prev.output], outputs = x)

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr= 1.61e-3, decay = 0)
model_final.compile(loss = "mean_absolute_percentage_error", optimizer = opt)

#train the network
print("[INFO] training network...")
epoch_number = 1000
BS = 10;

### Fitting the final model 
H2 = model_final.fit( x = [penultimate_output_mlp_cnn_prev_train , penultimate_output_gnn_prev_train], y = trainY, validation_data = ( [penultimate_output_mlp_cnn_prev_test , penultimate_output_gnn_prev_test], testY), batch_size = BS, epochs = epoch_number, verbose=1, shuffle=False) 

 
print("[INFO] evaluating network...")
## Evaluating the testing dataset
preds = model_final.predict([penultimate_output_mlp_cnn_prev_test , penultimate_output_gnn_prev_test],batch_size=BS)

#####----------------------------------------------------------Evaluating & plotting the model results--------------------------------------------------------------------------------------------------------------------------

# compute the difference between the predicted and actual values and then compute the % difference and absolute % difference
diff = preds.flatten() - testY.flatten() 
PercentDiff = (diff/testY.flatten())*100
absPercentDiff = (np.abs(PercentDiff))

# compute the mean and standard deviation of absolute percentage difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print("[INF0] mean {: .2f},  std {: .2f}".format(mean, std) )

# plotting predicted logP vs Actual logP
N = len(testY)
colors = np.random.rand(N)
x = testY.flatten()
y =  preds.flatten() 
plt.scatter(x, y, c=colors)
plt.plot( [0,15],[0,15] )
plt.xlabel('Actual logP', fontsize=18)
plt.ylabel('Predicted logP', fontsize=18)
plt.savefig('logP_pred_using_hybrid_MLP_GNN_2D_with_partial_shannon.png')
plt.show()

####-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


####-------------------------------------------------------------------------------------------------------Evaluating standard statistics of model performance--------------------------------------------------------------------
    
# ### MAE as a function
# def mae(y_true, predictions):
#     y_true, predictions = np.array(y_true), np.array(predictions)
#     return np.mean(np.abs(y_true - predictions))

# ### The MAE estimated
# print("The mean absolute error estimated: {}".format( mae(x, y) )) 

### The MAPE
print("The mean absolute percentage error: {}".format( mean_absolute_percentage_error(x, y) ) )   

### The MAE
print("The mean absolute error: {}".format( mean_absolute_error(x, y) ))    
    
### The MSE
print("The mean squared error: {}".format( mean_squared_error(x, y) ))  

### The RMSE
print("The root mean squared error: {}".format( mean_squared_error(x, y, squared= False) ) )    

### General stats
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
print("The R^2 value between actual and predicted target:", r_value**2)


# plot the training loss and validation loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch_number ), H2.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch_number ), H2.history["val_loss"], label="val_loss")
plt.title("Training loss and Validation loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss@epochs_with_partial_shannon_hybrid_MLP_GNN_2D')

####------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------