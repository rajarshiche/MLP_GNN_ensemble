# This script can be used with Shannon or w/o Shannon entropy framework: (i) only MW or (ii) MW + partial Shannon(SMILES) as descriptors or (iii) MW + Shannon(SMILES) + partial Shannon(SMILES) as descriptors


# Importing necessary packages

from openbabel import pybel
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
import re
import math
from mlp_cnn_gnn import mlp_cnn_gnn
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error
from rdkit import Chem
# k-mer tokenizer
from SmilesPE.pretokenizer import kmer_tokenizer


# Getting the list of molecules from csv file containing smiles strings
df = pd.read_csv('Ki_target_CHEMBL5023_only_equal_less_than_10K_dot_smiles_removed.csv', encoding='cp1252') 
df_smiles = df['Smiles'].values



# Processing the targets
# Reading the SMILES strings using pybel
read_mols_all_mod = [pybel.readstring("smi", x) for x in df_smiles]
target_logP_mod = []
for i in range(0,len(df_smiles)):
    read_mols_all_mod[i].addh()
    # The calcdesc() method of a Molecule returns a dictionary containing descriptor values for LogP, Polar Surface Area (“TPSA”) and Molar Refractivity (“MR”)
    descvalues = read_mols_all_mod[i].calcdesc()
    target_logP_mod.append(descvalues["logP"])

y_val_mod = pd.DataFrame(target_logP_mod, columns = ["logP"])   
 
# Saving the target dataframe as a csv
y_val_mod.to_csv('Ki_data_list_logP_mod.csv', index=False)       


# Getting the MWs as the sole descriptor array
fp_matrix = df['Molecular Weight'].values
  
# converting fp_matrix to dataframe for input into MLP
fp_mol = pd.DataFrame(fp_matrix)

  


# Estimation of Shannon entropy (SMILES)
SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
regex = re.compile(SMI_REGEX_PATTERN)

def shannon_entropy_smiles(mol_smiles):
    
    molecule = mol_smiles 
    tokens = regex.findall(molecule)
    
    ## uncomment the below line if kmer_tokenizer is to be used
    # tokens = kmer_tokenizer(molecule, ngram=1)
    
    ### Frequency of each token generated
    L = len(tokens)
    L_copy = L
    tokens_copy = tokens
    
    num_token = []
    
    
    for i in range(0,L_copy):
        
        token_search = tokens_copy[0]
        num_token_search = 0
        
        if len(tokens_copy) > 0:
            for j in range(0,L_copy):
                if token_search == tokens_copy[j]:
                    num_token_search += 1    
                    
            num_token.append(num_token_search)   
                
            while token_search in tokens_copy:
                    
                tokens_copy.remove(token_search)
                    
            L_copy = L_copy - num_token_search
            
            if L_copy == 0:
                break
        else:
            pass
    
    # Calculation of Shannon entropy
    total_tokens = sum(num_token)
    
    shannon = 0
    
    for k in range(0,len(num_token)):
        
        pi = num_token[k]/total_tokens
        
        shannon = shannon - pi * math.log2(pi)
        
    return shannon   


# generating a dictionary of atom occurrence frequencies given atom_list
def freq_atom_list(atom_list_input_mol):
    
    atom_list = ['H', 'C', 'N', 'O', 'S', 'P','F', 'Cl', 'Br', 'I'] 
    dict_freq = {}
    
    # adding keys
    for i in range(len(atom_list)):
        dict_freq[atom_list[i]] = 0  ### The values are all set 0 initially

    
    # update the value by 1 when a key in encountered in the string
    for i in range(len(atom_list_input_mol)):
        dict_freq[ atom_list_input_mol[i] ] = dict_freq[ atom_list_input_mol[i] ] + 1
    
    # convert the dictionary values as frequency array
    freq_atom_list =  list(dict_freq.values())/ (  sum(  np.asarray (list(dict_freq.values()))  )    )
    
    # getting the final frequency dictionary
    # adding values to keys
    for i in range(len(atom_list)):
        dict_freq[atom_list[i]] = freq_atom_list[i]  
        
    freq_atom_list = dict_freq
         
    return freq_atom_list


# Maximum length of SMILES strings in the database
len_smiles = []
for j in range(0,len(df['Smiles'])):
    
    mol = Chem.MolFromSmiles(df['Smiles'][j])
    # No H considered
    # mol = Chem.AddHs(mol)
    k=0
    for atom in mol.GetAtoms():
        k = k +1 
    len_smiles.append(k)

max_len_smiles = max(len_smiles)


# Constructing the padded array of partia or fractional Shannon per molecule
def ps_padding(ps, max_len_smiles):
    
    len_ps = len(ps)
    
    len_forward_padding = int((max_len_smiles - len_ps)/2)
    len_back_padding = max_len_smiles - len_forward_padding - len_ps
    
    ps_padded = list(np.zeros(len_forward_padding))  + list(ps) + list(np.zeros(len_back_padding))
    
    return ps_padded 


# estimate the Shannon/ partial Shannon and construct the descriptor table
fp_combined = []
shannon_arr = []
for i in range(0,len(df['Smiles'])):  
    
  mol = Chem.MolFromSmiles(df['Smiles'][i])
  
  total_shannon = shannon_entropy_smiles(df['Smiles'][i])
  shannon_arr.append( total_shannon )
  
  # The atom list as per rdkit in string form
  atom_list_input_mol = []
  for atom_rdkit in mol.GetAtoms():
     atom_list_input_mol.append(str(atom_rdkit.GetSymbol()))     
         
  freq_list_input_mol = freq_atom_list(atom_list_input_mol)

  ps = []
  for atom_rdkit in mol.GetAtoms():
      atom_symbol = atom_rdkit.GetSymbol()
      atom_type = atom_symbol # atom symbol in atom type
      
      partial_shannon = freq_list_input_mol[atom_type] * total_shannon
      ps.append( partial_shannon )
      # ps.append( freq_list_input_mol[atom_type] )
  

  ps_arr = ps_padding(ps, max_len_smiles)     
  fp_combined.append(ps_arr)

# Shannon_entropy as descriptor
shannon_smiles = pd.DataFrame(shannon_arr, columns= ['shannon_smiles'])

# fractional/ partial shannon_entropy as descriptor
fp_mol = pd.DataFrame(fp_combined)

# MW as descriptor to df_new table
df_new = pd.concat([ df['Molecular Weight'] ], axis = 1)

# Uncomment below:  MW + Shannon(SMILES) + partial Shannon(SMILES) as descriptors
# df_new = pd.concat([ df['Molecular Weight'], shannon_smiles , fp_mol], axis = 1)

# Uncomment below:  MW + partial Shannon(SMILES) as descriptors
# df_new = pd.concat([ df['Molecular Weight'], fp_mol], axis = 1)


# Normalizing the target
maxPrice = y_val_mod.iloc[:,-1].max() # grab the maximum price in the training set's last column
minPrice = y_val_mod.iloc[:,-1].min() # grab the minimum price in the training set's last column
print(maxPrice,minPrice)



print("[INFO] constructing training/ testing split")
split = train_test_split(df_new, y_val_mod, test_size = 0.15, random_state = 10) 

# Distribute values between train & test 
(XtrainTotalData, XtestTotalData, XtrainLabels, XtestLabels) = split  # split format always is in ( data_train, data_test, label_train, label_test)
 
# Normalizing the test or Labels
XtrainLabels = XtrainLabels/(maxPrice)
XtestLabels = XtestLabels/(maxPrice)  

XtrainData = XtrainTotalData
XtestData = XtestTotalData

# # Checking the shapes of input data
# print("Xtrain feature data shape",XtrainData.shape)
# print("Xtest feature data shape",XtestData.shape)
# print("Xtrain label shape",XtrainLabels.shape)
# print("Xtest label data shape",XtestLabels.shape)


# perform min-max scaling each continuous feature column to the range [0 1]
cs = MinMaxScaler()
trainContinuous = cs.fit_transform(XtrainData)
testContinuous = cs.transform(XtestData)

print("[INFO] processing input data after normalization....")
XtrainData, XtestData = trainContinuous,testContinuous


# create the MLP model
mlp =  mlp_cnn_gnn.create_mlp(XtrainData.shape[1], regress = False) # the input dimension to mlp would be shape[1] of the matrix i.e. column features

# # checking the model output shape
# print("shape of mlp", mlp.output.shape)

# The final input to our last layer will be output from the MLP lyers: we just defined it as "combinedInput" so that other output layers could be added later for hybrid models
combinedInput = mlp.output
# print("shape of combinedInput",combinedInput.shape)

# Final FC or Dense layers 
x = Dense(100, activation = "relu") (combinedInput)
x = Dense(16, activation = "relu") (combinedInput)  # This (none, 16) dense dimension is to maintain similar architecture as in GNN or in MLP+GNN models for comparison
x = Dense(1, activation = "linear") (x)
print("shape of x", x.shape)


# FINAL MODEL 
model = Model(inputs = mlp.input, outputs = x)
# print("shape of mlp input", mlp.input.shape)


# initialize the optimizer, compile the model
print("[INFO] compiling model...")
opt = SGD(lr= 1.02e-6, decay = 0)
model.compile(loss = "mean_absolute_percentage_error", optimizer = opt)

#train the network
print("[INFO] training network...")
trainY = XtrainLabels
testY = XtestLabels
epoch_number = 500
BS = 10;
# print("The batch size", BS)

# Defining the early stop to monitor the validation loss to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=epoch_number, verbose=1, mode='auto')

H = model.fit( x = XtrainData , y = trainY, validation_data = ( XtestData, testY), batch_size = BS, epochs = epoch_number, verbose=1, shuffle=False, callbacks = [early_stop]) # shuffle = False to reduce randomness and increase reproducibility


# evaluate the network
print("[INFO] evaluating network...")
preds = model.predict(XtestData,batch_size=BS)

# compute the difference between the predicted and actual values, then compute the % difference 
diff = preds.flatten() - testY.values.flatten()
PercentDiff = (diff/testY.values.flatten())*100
absPercentDiff = (np.abs(PercentDiff))

# compute the mean and standard deviation of absolute percentage difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print("[INF0] prediction mean {: .2f},  std {: .2f}".format(mean, std) )

# Plotting Predicted vs Actual 
N = len(testY)
colors = np.random.rand(N)
x = testY.values.flatten()*maxPrice 
y = preds.flatten()* maxPrice
plt.scatter(x, y, c=colors)
plt.plot( [0,maxPrice],[0,maxPrice] )
plt.xlabel('Actual logP', fontsize=18)
plt.ylabel('Predicted logP', fontsize=18)
plt.savefig('logP_pred_MLP_onlyn.png')
plt.show()


# # Calculating MAE from a function
# def mae(y_true, predictions):
#     y_true, predictions = np.array(y_true), np.array(predictions)
#     return np.mean(np.abs(y_true - predictions))

# #The MAE estimated
# print("The mean absolute error estimated: {}".format( mae(x, y) )) 

# Evaluation of prediction statistics / performance

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