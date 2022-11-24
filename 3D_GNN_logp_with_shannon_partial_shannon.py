### This script models and runs a 3D GNN-based network architecture to predict logp values of molecules as per the dataset

# Importing necessary packages
import time
start_time = time.time()


from openbabel import pybel
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Importing necessary rdkit packages
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.Chem.Draw import MolToImage, MolDrawOptions
from rdkit.ML.Cluster import Butina


# Importing graph generation functions
from sd_to_graph_3D_GNN import obabel_to_networkx3d_graph, shannon_entropy_smiles

import shutil
import numpy as np


## Importing models and functions
from mlp_cnn_gnn import mlp_cnn_gnn

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam


# Importing MHFP fingerprinting package
from mhfp.encoder import MHFPEncoder

# Importing Stellargraph related packages
from stellargraph.mapper import PaddedGraphGenerator


import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error



####------------------------------------------saving the target (logp) as a csv file--------------------------------------------------------------------------------------------
# Getting the list of molecules from csv file
Ki_data_list = pd.read_csv('Ki_target_CHEMBL5023_only_equal_less_than_10K_dot_smiles_removed.csv', encoding='cp1252') 
Ki_list = Ki_data_list['Smiles'].values

new_df_mol = Ki_data_list
# Extracting the SMILES corresponding to the index values
df_SMILES = new_df_mol['Smiles'].values

## length of df_SMILES strings
len_df_SMILES = []

for i in range(len(df_SMILES)):
    
    len_df_SMILES.append(len(df_SMILES[i]))

## Plotting the length of len_df_SMILES histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(len_df_SMILES, bins = 10)
plt.title('SMILES string length Distribution')
plt.xlabel('length of SMILES')
plt.show() 

# Need to filter out SMILES strings beyond 500 characters length for the sake of computational time
df_SMILES_mod = []
for i in range(len(df_SMILES)):
    
    if len_df_SMILES[i] <= 500:
        df_SMILES_mod.append(df_SMILES[i])
        
# Reading the SMILES strings using pybel
read_mols_all_mod = [pybel.readstring("smi", x) for x in df_SMILES_mod]

# processing the targets which are logP values
target_logP_mod = []
for i in range(0,len(df_SMILES_mod)):
    read_mols_all_mod[i].addh()
    
    ## The calcdesc() method of a Molecule returns a dictionary containing descriptor values for LogP, Polar Surface Area (“TPSA”) and Molar Refractivity (“MR”)
    descvalues = read_mols_all_mod[i].calcdesc()
    target_logP_mod.append(descvalues["logP"])

y_val_mod = pd.DataFrame(target_logP_mod, columns = ["logP"])    

# Saving the feature dataframe as a csv
y_val_mod.to_csv('Ki_data_list_logP_mod.csv', index=False)        


# This is the part to save ALL the target values in a csv file - keep it commented as we will be using only smiles strings <=500 characters
###-------------------------------------------------------------------------------------------------------------------------------------------
# #Reading the SMILES strings using pybel
# read_mols_all = [pybel.readstring("smi", x) for x in df_SMILES]

# # Processing the targets
# target_logP = []
# for i in range(0,len(df_SMILES)):
#     read_mols_all[i].addh()
    
#     ## The calcdesc() method of a Molecule returns a dictionary containing descriptor values for LogP, Polar Surface Area (“TPSA”) and Molar Refractivity (“MR”)
#     descvalues = read_mols_all[i].calcdesc()
#     target_logP.append(descvalues["logP"])

# y_val = pd.DataFrame(target_logP, columns = ["logP"])    
# # # ## Saving the featuredataframe as a csv
# y_val.to_csv('Ki_data_list_logP_mod.csv', index=False)
###-------------------------------------------------------------------------------------------------------------------------------------------

# ###-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


####------------------------------------------conformer & graph generation: saving graph of the lowest energy conformer (note: keep this section commented after graph generation & saving)-----------------------------------------------------------------------------------------------------
def calc_energy(mol, conformerId, minimizeIts):
 	ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conformerId)
 	ff.Initialize()
 	ff.CalcEnergy()
 	results = {}
 	if minimizeIts > 0:
         results["converged"] = ff.Minimize(maxIts=minimizeIts)
 	results["energy_abs"] = ff.CalcEnergy()
 	return results   

def cluster_conformers(mol, mode="RMSD", threshold=2.0):
 	if mode == "TFD": 
          dmat = TorsionFingerprints.GetTFDMatrix(mol)
 	else: 
          dmat = AllChem.GetConformerRMSMatrix(mol, prealigned=False)
 	rms_clusters = Butina.ClusterData(dmat, mol.GetNumConformers(), threshold, isDistData=True, reordering=True)
 	return rms_clusters
 
def confgen(input, output, prunermsthresh, numconf):
    
    mol = Chem.AddHs(Chem.MolFromSmiles(input), addCoords=True)
    
    # For larger smiles or long molecules, use conditions like useRandomCoords=True, maxAttempts = 5000
    cids = AllChem.EmbedMultipleConfs(mol, numConfs = numconf, numThreads=0,useRandomCoords=True,
                            maxAttempts = 5000, pruneRmsThresh = prunermsthresh ) #maxAttempts = 10000 for longer SMILES string
        

    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
    
    # relaxation with 'MMFF94s' force field
    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant='MMFF94s')
    
    w = Chem.SDWriter(output)
    
    res = []
 
    for cid in cids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
        e = ff.CalcEnergy()
        res.append((cid, e))
    sorted_res = sorted(res, key=lambda x:x[1])
    
    rdMolAlign.AlignMolConformers(mol)
    
    i=0
    for cid, e in sorted_res:
        
        # print("The conformer energy: ", e)
        mol.SetProp('CID', str(cid))
        mol.SetProp('Energy', str(e))
        
        if i == 0: ### saving only the lowest energy conformer
            w.write(mol, confId=cid)
            w.close()
        i = i + 1    

for i in range(0,len(df_SMILES_mod)):  
    
    smiles = df_SMILES_mod[i]
     
    mol_rdkit = Chem.MolFromSmiles(df_SMILES[i])
    mol_rdkit = Chem.AddHs(mol_rdkit)    

    n_conf = 10
    confgen( smiles,"min_en_conf{}.sdf".format(i), 0.1, n_conf) ### we create 10 conformer as dataset to pick the lowest energy one
   
    # moving the generated conformer to Ki_sdf folder to save it as a sdf file
    shutil.move("min_en_conf{}.sdf".format(i), "Ki_sdf/mol{}.sdf".format(i))
    
    # defining the iterable as the .sdf file 
    iterable = "Ki_sdf/mol{}.sdf".format(i)

    # reading a molecule in .sdf file format with coordinates and neighbor coordinates
    for mol in pybel.readfile("sdf",iterable):
        mol_pybel = mol
        # print(mol)
        for atom in mol:
            coords = atom.coords
            # print(coords)

    
    # constructing the networkx graph or nx graph: mol_pybel corresponds to the lowest energy 3D sdf structure calculated and mol_rdkit corresponds to the rdkit mol from smiles (H atoms are added in both cases)
    graph = obabel_to_networkx3d_graph(mol_pybel, mol_rdkit)
    
    
    # saving graph created above in gml format, note: THIS TAKES LARGE SPACE TO SAVE ON DISK, USER MAY CONSIDER GRAPH GENERATION AND TRAINING ON-THE-FLY
    nx.write_gml(graph, "graph_min_en_conf{}.gml".format(i))
    shutil.move("graph_min_en_conf{}.gml".format(i), "Ki_sdf_to_graph/mol{}.gml".format(i))
    
    
    # Constructing a stellargraph with all the above info and checking the nodes, edges and adjacency
    # from stellargraph import StellarGraph
    # g = graph
    # ## getting the feature attribute of graph g
    # g_feature_attr = StellarGraph.from_networkx(g, node_features="feat")
    # ## g_feature_attr = StellarGraph.from_networkx(g)
    # ##print(g_feature_attr.info())
    
    # ## Getting the new stellar graph as g_stellar
    # g_stellar = g_feature_attr
    
    # # Getting node features in a matrix
    # print(g_stellar.node_features())
    
    # # # Getting edge features in a matrix
    # # print(g_stellar.edge_features())
    
    # # Getting the adjacency matrix with weights
    # print(g_stellar.to_adjacency_matrix(weighted = True))
    
####-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

####----------------------------------------------------------------Constructing Stellar graphs from saved nx graphs : PLEASE COMMENT THE ABOVE NX GRAPH CONSTRUCTION SECTION-----------------------------------------------------------------------------
from stellargraph import StellarGraph
stellar_graph_list = []
for i in range(0,len(df_SMILES_mod)):
    
    g_nx = nx.read_gml("Ki_sdf_to_graph/mol{}.gml".format(i))
    g_sg = StellarGraph.from_networkx(g_nx, node_features="feat")
    stellar_graph_list.append(g_sg)
####-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    
####----------------------------------Total Process Time-------------------------------------------------------------------------------------------------------------------
end_time = time.time()
print("The duration of run (in s) so far till graph construction: ", end_time - start_time)
###-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    
###--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Summary stats of the graph data
graphs = stellar_graph_list

graphs_summary = pd.DataFrame( [ ( g.number_of_nodes(), g.number_of_edges() ) for g in graphs] , columns = ["nodes", "edges"]  )
    
print(graphs_summary.describe().round(1))

# display the graph_labels
graph_labels = y_val_mod
print(" Original labels: ", graph_labels)

# # converting the labels into 0 or 1
# graph_labels = pd.get_dummies(graph_labels, drop_first = True)
# print(" After conversion: ", graph_labels)
###--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

###------------------------------------------------------------------------Basic graph & network processing conditions-------------------------------------------------------

### In GNN, # of nodes can differ per graph, but node features should have the same dimension. Else, will get the error: "graphs: expected node features for all graph to have same dimensions". The PaddedGraphGenerator
### helps to address this problem
generator = PaddedGraphGenerator(graphs = graphs)

# Defining model parameters for GNN training
epoch_number = 500
BS = 10

# epoch_number = 1000
# BS = 10

# defining early stopping
es = EarlyStopping(
    
                    monitor = "val_loss", min_delta = 0, patience = 500, restore_best_weights = True
                   
                  )
###---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

###----------------------------------------------------------------------GNN: generation of training/ testing set------------------------------------------------------------------------------------------------------------------------------

# Prepare the dataset for training the GNN
def get_generators(train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
    )

    return train_gen, test_gen


# Getting the train & test indices using train_test_split
indices = np.arange(len(graphs))

# Splitting both graphs & indices along with targets/ labels
X_train, X_test, train_index, test_index, XtrainLabels, XtestLabels = train_test_split(graphs, indices, y_val_mod, test_size=0.15,random_state=10)


train_gen, test_gen = get_generators(train_index, test_index, graph_labels, batch_size = BS)

###----------------------------------------------------------------------GNN: normalizing the target column (optional) and converting it as np array----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# grab the maximum/ minimum price in the target columm
maxPrice = y_val_mod.iloc[:,-1].max() # grab the maximum in the training set's last column
minPrice = y_val_mod.iloc[:,-1].min() # grab the minimum in the training set's last column
# print(maxPrice,minPrice)

### Normalizing the test or Labels: turned off for training hybrid models. Using it is optional here.
# XtrainLabels = XtrainLabels/(maxPrice)
# XtestLabels = XtestLabels/(maxPrice)  

print("Xtrain label shape",XtrainLabels.shape)
print("Xtest label data shape",XtestLabels.shape)
print("Shape of the logP labels array", y_val_mod.shape)

trainY = XtrainLabels
testY = XtestLabels

## Converting the target column in np array
trainY = np.asarray(trainY)
testY = np.asarray(testY)


###------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#####----------------------------------------------------------Constructing & training the GNN model------------------------------------------------------------------------------------------------------

# GNN model input: graphs data to train/ test
graphs_train = []
for k in range(0, len(train_index)):
    graphs_train.append(graphs[train_index[k]])
    
graphs_test = []
for k in range(0, len(test_index)):
    graphs_test.append(graphs[test_index[k]])

# GNN model input to generator as per train/test set   
testing_data_gnn = generator.flow(graphs_test)
training_data_gnn = generator.flow(graphs_train)    
    
# Defining the model instance
gnn_model = mlp_cnn_gnn.create_gnn(generator)

# Final Dense/FC layers 
x = Dense(10, activation = "relu") (gnn_model.output)
x = Dense(1, activation = "linear") (x)
print("shape of x", x.shape)

# The final model as gnn_model
gnn_model = Model(inputs = gnn_model.input , outputs = x)

# initialize the optimizer & compile the model
print("[INFO] compiling model...")
opt  = Adam(lr = 0.00015, decay =  0.00015/200)
gnn_model.compile(loss = "mean_absolute_percentage_error", optimizer = opt)

epoch_number = 500
BS = 10

epochs = epoch_number

# Model fitting across epochs
# shuffle = False to reduce randomness and increase reproducibility
H = gnn_model.fit(train_gen, epochs = epochs, validation_data = test_gen, verbose = 2, shuffle = False, callbacks = [es])

####-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("[INFO] evaluating network...")
# Evaluating the testing dataset
preds = gnn_model.predict(test_gen,batch_size=BS)

#####----------------------------------------------------------Evaluating & plotting the model results--------------------------------------------------------------------------------------------------------------------------

# compute the difference between the predicted and actual values, then compute the % difference and absolute difference
diff = preds.flatten() - testY.flatten() 
PercentDiff = (diff/testY.flatten())*100
absPercentDiff = (np.abs(PercentDiff))

# compute the mean and standard deviation of absolute percentage difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print("[INF0] mean {: .2f},  std {: .2f}".format(mean, std) )

## Plotting Predicted logP vs Actual logP
N = len(testY)
colors = np.random.rand(N)
x = testY.flatten()
y =  preds.flatten() 
plt.scatter(x, y, c=colors)
plt.plot( [0,10],[0,10] )
plt.xlabel('Actual logP', fontsize=18)
plt.ylabel('Predicted logP', fontsize=18)
plt.savefig('logP_pred_using_GNN_3D_with_partial_shannon.png')
plt.show()

####-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


####-----------------------------------------------------------------------Evaluating standard statistics of model performance---------------------------------------------------------------------------------------------------  
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
plt.plot(np.arange(0, epoch_number ), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch_number ), H.history["val_loss"], label="val_loss")
plt.title("Training loss and Validation loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss@epochs_with_shannon_3D_GNN_with_partial_shannon')

####-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
