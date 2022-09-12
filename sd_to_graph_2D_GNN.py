
# Importing necessary packages
import openbabel as ob
from openbabel import pybel
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# in case MHFP fingerprint is used
from mhfp.encoder import MHFPEncoder

import re
import math

# in case k-mer tokenization is used
from SmilesPE.pretokenizer import kmer_tokenizer


##-------------------------------------------------------------------------------------------------------------

def atom_one_hot(atom_type):
    
    atom_list = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I']
    atom_type_encoded = np.zeros(len(atom_list))
    
    for i in range(0, len(atom_list)):
        
        if atom_type == atom_list[i]:
            atom_type_encoded[i] = 1
            break
    # print(atom_type_encoded)    
            
    return atom_type_encoded.astype('float')

 
##-------------------------------------------------------------------------------------------------------------

##-------------------------------------------------------------------------------------------------------------

## Calculating the Shannon entropy (SMILES) for each smiles string

SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
regex = re.compile(SMI_REGEX_PATTERN)


def shannon_entropy_smiles(mol_smiles):
    
    molecule = mol_smiles 
    tokens = regex.findall(molecule)
    # tokens = kmer_tokenizer(molecule, ngram=1)
    # print(tokens)
    
    # Frequency of each token generated
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
        
    
    # Calculation of Shannon entropy (SMILES)
    total_tokens = sum(num_token)
    
    shannon = 0
    
    for k in range(0,len(num_token)):
        
        pi = num_token[k]/total_tokens
        
        shannon = shannon - pi * math.log2(pi)
    
        
    return shannon    

# generating a dictionary of atom occurrence frequencies
def freq_atom_list(atom_list_input_mol):
    
    atom_list = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I'] 
    dict_freq = {}
    
    ### adding keys
    for i in range(len(atom_list)):
        dict_freq[atom_list[i]] = 0  ### The values are all set 0 initially
    
    ### update the value by 1 when a key in encountered in the string
    for i in range(len(atom_list_input_mol)):
        dict_freq[ atom_list_input_mol[i] ] = dict_freq[ atom_list_input_mol[i] ] + 1
    
    ### The dictionary values as frequency array
    freq_atom_list =  list(dict_freq.values())/ (  sum(  np.asarray (list(dict_freq.values()))  )    )
    
    # Getting the final frequency dictionary
    # adding values to keys
    for i in range(len(atom_list)):
        dict_freq[atom_list[i]] = freq_atom_list[i]  
        
    freq_atom_list = dict_freq    
        
    return freq_atom_list

# generating a dictionary of bond occurrence frequencies (note: not called or used in the current graph architecture)
def freq_bond_list(bond_list_input_mol):
    
    bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'QUADRUPLE', 'AROMATIC', 'HYDROGEN', 'IONIC'] 
    dict_freq = {}
    
    ### adding keys
    for i in range(len(bond_list)):
        dict_freq[bond_list[i]] = 0  ### The values are all set 0 initially

    
    ### update the value by 1 when a key in encountered in the string
    for i in range(len(bond_list_input_mol)):
        dict_freq[ bond_list_input_mol[i] ] = dict_freq[ bond_list_input_mol[i] ] + 1
    
    ### The dictionary values as frequency array
    freq_bond_list =  list(dict_freq.values())/ (  sum(  np.asarray (list(dict_freq.values()))  )    )
    
    
    # Getting the final frequency dictionary
    # adding values to keys
    for i in range(len(bond_list)):
        dict_freq[bond_list[i]] = freq_bond_list[i]  
        
    freq_bond_list = dict_freq
          
    return freq_bond_list

##-------------------------------------------------------------------------------------------------------------
def obabel_to_networkx3d_graph(input_mol, mol, **kwargs):

    """
    Takes a pybel or rdkit molecule object and converts it into a networkx graph with node and edge features.
 
    """
 
    graph = nx.Graph()
    graph.graph['info'] = str(input_mol).strip()
    
        
    ### The atom list as per rdkit in string form
    atom_list_input_mol = []
    for atom_rdkit in mol.GetAtoms():
        atom_list_input_mol.append(str(atom_rdkit.GetSymbol()))     

    
    freq_list_input_mol = freq_atom_list(atom_list_input_mol)
 
    # print("The atom frequency dictionary: ", freq_list_input_mol)  
   
    for atom, atom_rdkit in zip(input_mol, mol.GetAtoms()):
        
        # atomic_no = atom.atomicnum ### atomic_no could be used in place of atomic_mass
        atomic_mass = atom.atomicmass
        node_id = atom.idx - 1
        graph.add_node(node_id)
        
        atom_symbol = atom_rdkit.GetSymbol()
        
        atom_type = atom_symbol ### atom symbol in atom type

        
        ## getting the smiles string from input_mol
        mol_smiles = input_mol.write("smi")
                
        node_feature = []
        # node_feature.append(atomic_no) 
        node_feature.append(atomic_mass)
           
        
        # adding total Shannon entropy (SMILES) to node feature 
        node_feature.append(shannon_entropy_smiles(mol_smiles))  
        
        # SECFP (SMILES Extended Connectifity Fingerprint)
        # fp = MHFPEncoder.secfp_from_smiles(in_smiles = mol_smiles, length=2048, radius=3, rings=True, kekulize=True, sanitize=False)
        # for k in range(0,len(fp)):
        #     node_feature.append(fp[k].astype(float))
            
        # estimating the partial or fractional Shannon entropy (SMILES) for an atom type => the current node
        total_shannon = shannon_entropy_smiles(mol_smiles) 
        partial_shannon = freq_list_input_mol[atom_type] * total_shannon
        node_feature.append(  partial_shannon )
        
        
        # Defining the node feature with node id and under general graph features 'feat' 
        graph.nodes[node_id]['feat'] = node_feature
        
        # # To check the node features
        # print("node feature", node_feature)

       
    for bond in ob.OBMolBondIter(input_mol.OBMol):
        label = bond.GetBondOrder() ## Alternatively:OBBond.GetBondOrder(bond)
        # print("Bond label: ", label)
        fraction_label = label/6   ### Normalizing bond order wrt total order = 1+2+3 = 6
        graph.add_edge(bond.GetBeginAtomIdx() - 1,
                        bond.GetEndAtomIdx() - 1,
                        weight = fraction_label) ## The weight attributes added 
    
    return graph

##-------------------------------------------------------------------------------------------------------------

# testing any SMILES  
if __name__ == '__main__': 
    
    from rdkit import Chem
    
    smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    # smiles = 'COc1cc(C(F)(F)F)ccc1CN1C(=O)C(OC(C)C)=C(C(=O)c2ccccc2)C1c1ccc(Br)cc1'
    
    ### Adding H's here while reading the smiles
    input_mol = pybel.readstring("smi", smiles)
    input_mol.OBMol.AddHydrogens()
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
        

    ### nx graph with H's
    g = obabel_to_networkx3d_graph(input_mol, mol)
    nx.draw(g, with_labels = True)
    plt.show()
    
    ### Getting the feature matrix 
    X = np.array([v for k, v in nx.get_node_attributes(g, 'feat').items()])    ### for key, value in kwargs.items()
    print(X)
    
    # getting the adjacency matrix
    A = nx.adjacency_matrix(g) ## Alternatively: nx.to_numpy_array(g, nodelist = list(g.nodes))
    print(A.todense())
    
    # getting the attribute dictionary: 
    X_dict = nx.get_node_attributes(g, 'feat')  
    
    # constructing a stellargraph with all the above info
    from stellargraph import StellarGraph
    
    # getting the feature attribute of graph g
    g_feature_attr = StellarGraph.from_networkx(g, node_features= 'feat')
    print(g_feature_attr.info())
    
    # Getting the new stellar graph as g_stellar
    g_stellar = g_feature_attr
    print(g_stellar.node_features())
    
    # getting the new stellar graph directly from nx graph 'g'
    g_stellar = StellarGraph.from_networkx(g)
    print(g_stellar.info())
    
##--------------------------------------------------------------------------------------------------------------------


   