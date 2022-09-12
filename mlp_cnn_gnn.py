# import necessary packages 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from stellargraph.layer import GCNSupervisedGraphClassification



class mlp_cnn_gnn:

    def create_gnn(generator):    
        
        ####--------------------------------------------creating the stellargraph GCN Model------------------------------------------------------------------------------------------------------------------
        gc_model = GCNSupervisedGraphClassification(
                layer_sizes=[440, 64, 32],
                activations=["relu", "relu", "relu"],
                generator=generator,
                dropout=0.1)
            
        x_inp, x_out = gc_model.in_out_tensors()
        predictions = Dense(units=32, activation="relu")(x_out)
        predictions = Dense(units=16, activation="relu")(predictions)

        # define th GNN model
        model_GNN = Model(inputs=x_inp, outputs=predictions)
        
        # return the GNN model
        model_GNN.summary()
        return model_GNN

        ####-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        
        ####-----------------------------------------MLP & CNN models (CNN model building function was not actually used, only the MLP function was used)----------------------------------------------------
    # a tuple of progressively larger filters used so that our network learn more discriminate features
    # regress: A boolean indicating whether or not a fully-connected linear activation will be added to the CNN for regression purpose    
    def create_cnn(width, height, depth, filters = (2,4,8), regress = False):    

        # initialize the input shape and channel dimension assuming Tensorflow/ channels-last ordering (=> depth in the end)
        inputShape = (height, width, depth)
        chanDim = -1    

        # define the model input
        inputs = Input(shape = inputShape)    

        # We loop over the filters to create CONV => RELU => BN => POOL layers. Each iteration of the loop appends these layers
        for (i,f) in enumerate(filters): # f is the number of filters in each iteration

            # for the 1st CONV layer we set the input as x
            if i == 0:
                x = inputs

            # for rest iterations we progressively change x through the layers: CONV => RELU => BN => POOL
            x = Conv2D(f , (2,2), padding = "same")(x) # f is the number of filters in that particular layer
            x = Activation("relu")(x)
            x = BatchNormalization(axis = chanDim)(x)
            x = MaxPooling2D(pool_size = (2,2))(x)


        # Flatten the volume and then apply FC => RELU => BN => Dropout
        # x = Flatten() (x)
        x = GlobalAveragePooling2D() (x)  # GlobalAveragePooling2D(), compared to Flatten(), gave better accuracy values, and significantly reduced over-fitting and the no. of parameters
        x = Dense (10000) (x)
        
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.4)(x)  # Dropout helps in reducing validation loss

        # Apply another FC layer (Dense) to match the number of nodes coming out from MLP
        x = Dense(500)(x)
        x = Activation("relu")(x)

        # check to see if the regression node should be added
        if regress:
            x = Dense(1, activation ="linear")(x)

        # construct the CNN model
        model = Model(inputs, x)    

        # return the CNN model
        model.summary()
        
        return model    


    def create_mlp(dim, regress = False):

	# define our MLP Network: architecture dim 1000 -> 500

        model = Sequential()
        model.add(Dense(1000, input_dim = dim, activation = "relu"))
        
        # Make sure the below dimension within Dense is the same as the output dimension of create_cnn if this is used
        model.add(Dense(500, activation = "relu"))

    	# check to see if the regression node is to be added
        if regress: # if we are performing regression directly, we add a Dense layer containing a single neuron with a linear activation function
            model.add(Dense(1, activation = "linear"))

        
        # return the MLP model
        model.summary()
        return model	        
        
    ####-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------