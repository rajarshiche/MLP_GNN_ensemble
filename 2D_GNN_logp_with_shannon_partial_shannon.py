
# Importing necessary packages
import time
start_time = time.time()

from openbabel import pybel
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Importing Chem from rdkit package
from rdkit import Chem


# Importing graph generation functions
from sd_to_graph_2D_GNN import obabel_to_networkx3d_graph, shannon_entropy_smiles


import shutil
import numpy as np
import pandas as pd

from mlp_cnn_gnn import mlp_cnn_gnn

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard


# Importing fingerprinting package: MHFP encoder
from mhfp.encoder import MHFPEncoder

## Stellargraph related packages
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

# Need to filter out SMILES strings beyond 500 to compare the training & testing with "3D_GNN_logp_with_shannon_partial_shannon.py"
df_SMILES_mod = []
for i in range(len(df_SMILES)):
    
    if len_df_SMILES[i] <= 500:
        df_SMILES_mod.append(df_SMILES[i])
        
# Reading the SMILES strings using pybel
read_mols_all_mod = [pybel.readstring("smi", x) for x in df_SMILES_mod]

# Processing the targets which are logP values
target_logP_mod = []
for i in range(0,len(df_SMILES_mod)):
    read_mols_all_mod[i].addh()
    
    # The calcdesc() method of a Molecule returns a dictionary containing descriptor values for LogP, Polar Surface Area (“TPSA”) and Molar Refractivity (“MR”)
    descvalues = read_mols_all_mod[i].calcdesc()
    target_logP_mod.append(descvalues["logP"])

y_val_mod = pd.DataFrame(target_logP_mod, columns = ["logP"])   
 
# Saving the feature dataframe as a csv
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

###-----------------------------------------------------------------------------------------------------------------------------------------



###-----------------------------------------------------------------------------------------------------------------------------------------
for i in range(0,len(df_SMILES_mod)):

    smiles = df_SMILES_mod[i]
     
    mol = Chem.MolFromSmiles(df_SMILES[i])
    mol = Chem.AddHs(mol)
    

    ### Adding H's here while reading the smiles
    input_mol = pybel.readstring("smi", smiles)
    input_mol.OBMol.AddHydrogens()
    
    ### nx graph with H's
    graph = obabel_to_networkx3d_graph(input_mol, mol)
    

    
    # saving graph created above in gml format. NOTE: THIS TAKES LARGE SPACE TO SAVE ON DISK
    nx.write_gml(graph, "graph_min_en_conf_2D{}.gml".format(i))
    shutil.move("graph_min_en_conf_2D{}.gml".format(i), "Ki_sdf_to_graph_2D/mol{}.gml".format(i))
    
    
    
    # Constructing a stellargraph with all the above info and checking the nodes, edges and adjacency
    ### Constructing a stellargraph with all the above info
    # from stellargraph import StellarGraph
    
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
    
###-----------------------------------------------------------------------------------------------------------------------------------------


###----------------------------------------------------------------Constructing Stellar graphs from saved nx graphs ----------------------------------------------------------------------------

stellar_graph_list = []
for i in range(0,len(df_SMILES_mod)):
    
    g_nx = nx.read_gml("Ki_sdf_to_graph_2D/mol{}.gml".format(i))
    g_sg = StellarGraph.from_networkx(g_nx, node_features="feat")
    stellar_graph_list.append(g_sg)
###---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    
###----------------------------------Total Process Time for graph construction and ready for training-------------------------------------------------------------------------------------------
end_time = time.time()
print("The duration of run (in s) so far: ", end_time - start_time)

###-----------------------------------------------------------------------------------------------------------------------------------------------------------
    
###-----------------------------------------------------------------------------------------------------------------------------------------------------------
### Summary stats of the graph data
graphs = stellar_graph_list

graphs_summary = pd.DataFrame( [ ( g.number_of_nodes(), g.number_of_edges() ) for g in graphs] , columns = ["nodes", "edges"]  )
    
print(graphs_summary.describe().round(1))

# display the graph_labesl
graph_labels = y_val_mod
print(" Original labels: ", graph_labels)

# # Converting the labels into 0 or 1
# graph_labels = pd.get_dummies(graph_labels, drop_first = True)
# print(" After conversion: ", graph_labels)

###-----------------------------------------------------------------------------------------------------------------------------------------------------------
###-------------------------------------------------------Basic graph & network processing conditions---------------------------------------------------------

#### In GNN, # of nodes can differ per graph, but node features should have the same dimension. Else, will get the error: "graphs: expected node features for all graph to have same dimensions"
generator = PaddedGraphGenerator(graphs = graphs)

# Defining model parameters for GNN training
epoch_number = 500
BS = 10

# defining early stopping
es = EarlyStopping(
    
                    monitor = "val_loss", min_delta = 0, patience = 500, restore_best_weights = True
                   
                  )
###-----------------------------------------------------------------------------------------------------------------------------------------------------------


###----------------------------------------------------------------------GNN: generation of training/ testing data----------------------------------------------

# Prepare the dataset for training the GNN
def get_generators(train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
    )

    return train_gen, test_gen


# Getting the indices
indices = np.arange(len(graphs))

# Splitting both graphs & indices along with targets/ labels
X_train, X_test, train_index, test_index, XtrainLabels, XtestLabels = train_test_split(graphs, indices, y_val_mod, test_size=0.15,random_state=10)

train_gen, test_gen = get_generators(train_index, test_index, graph_labels, batch_size = BS)

###-----------------------------------------------------------------------------------------------------------------------------------------------------------



###--------------------------------------------------------------------------------GNN: normalizing the target column (optional) and converting it as np array---------------------------------------------------------------

# grab the maximum/ minimum price in the target columm
maxPrice = y_val_mod.iloc[:,-1].max() # grab the maximum price in the training set's last column
minPrice = y_val_mod.iloc[:,-1].min() # grab the minimum price in the training set's last column
print(maxPrice,minPrice)

### Normalizing the test or Labels: turned off during CNN-GNN hybrid models
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


###--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

###----------------------------------------------------------Constructing & training the GNN model------------------------------------------------------------------------------------------------------

# GNN model input: graphs data to train/ test
graphs_train = []
for k in range(0, len(train_index)):
    graphs_train.append(graphs[train_index[k]])
    
graphs_test = []
for k in range(0, len(test_index)):
    graphs_test.append(graphs[test_index[k]])

### GNN model input    
testing_data_gnn = generator.flow(graphs_test)
training_data_gnn = generator.flow(graphs_train)    
    
### Overall model input
gnn_model = mlp_cnn_gnn.create_gnn(generator)


# Final FC (Dense) layers 
x = Dense(10, activation = "relu") (gnn_model.output)
x = Dense(1, activation = "linear") (x)
print("shape of x", x.shape)

# The final model as gnn_model
gnn_model = Model(inputs = gnn_model.input , outputs = x)

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt  = Adam(lr = 0.00015, decay =  0.00015/200)
gnn_model.compile(loss = "mean_absolute_percentage_error", optimizer = opt)

epoch_number = 500
BS = 10

epochs = epoch_number


# Model fitting across epochs
# shuffle = False to reduce randomness and increase reproducibility
H = gnn_model.fit(train_gen, epochs = epochs, validation_data = test_gen, verbose = 2, shuffle = False, callbacks = [es])

###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("[INFO] evaluating network...")
## Evaluating the testing dataset
preds = gnn_model.predict(test_gen,batch_size=BS)

###----------------------------------------------------------Evaluating & plotting the model results--------------------------------------------------------------------------------------------------------------------------

# compute the difference between the predicted and actual values, then compute the % difference and absolute % difference
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
plt.savefig('logP_pred_using_GNN_2D_with_partial_shannon.png')
plt.show()

###-----------------------------------------------------------------------------------------Evaluating standard statistics of model performance-----------------------------------------------------------------------------
    
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
plt.savefig('loss@epochs_with_shannon_2D_GNN_with_partial_shannon')

###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------