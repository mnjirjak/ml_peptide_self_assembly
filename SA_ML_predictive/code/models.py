from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Masking, Concatenate
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D
from tensorflow.keras.regularizers import l2

def _create_seq_model(input_shape, conv1_filters = 5, conv2_filters = 5, conv_kernel_size = 6, num_cells = 64, dropout = 0.1, mask_value = 2):
    input_layer = Input(shape = input_shape)
    mask = Masking(mask_value = mask_value)(input_layer)
    
    if conv1_filters > 0:
        final_output_layer = Conv1D(conv1_filters, conv_kernel_size, padding = 'same', kernel_initializer = 'he_normal')(mask)
        
        if conv2_filters > 0:
            final_output_layer = Conv1D(conv2_filters, conv_kernel_size, padding = 'same', kernel_initializer = 'he_normal')(final_output_layer)
        final_output_layer = Bidirectional(LSTM(num_cells))(final_output_layer)
    else:
        final_output_layer = Bidirectional(LSTM(num_cells))(mask)

    if dropout > 0:
        final_output_layer = Dropout(dropout)(final_output_layer) 
     
    return input_layer, final_output_layer

def _one_prop_model(lstm1 = 5, lstm2 = 5, dense = 15, dropout = 0.2, lambda2 = 0.0, mask_value = 2):
    # LSTM model which processes dipeptide AP scores.

    input_layer = Input((None, 1))
    mask = Masking(mask_value = mask_value)(input_layer)
    lstm_layer_1 = Bidirectional(LSTM(lstm1, return_sequences = True))(mask)
    lstm_layer_2 = LSTM(lstm2)(lstm_layer_1)

    dense_layer1 = Dense(dense, activation = 'selu', kernel_regularizer = l2(l = lambda2))(lstm_layer_2)
    final_output_layer = Dropout(dropout)(dense_layer1)

    return input_layer, final_output_layer 

def create_seq_model(input_shape, conv1_filters = 5, conv2_filters = 5, conv_kernel_size = 6, num_cells = 64, dropout = 0.1, mask_value = 2):
    model_input = Input(shape = input_shape, name = "input_1")
    mask = Masking(mask_value = mask_value)(model_input)
    
    if conv1_filters > 0:
        final_output_layer = Conv1D(conv1_filters, conv_kernel_size, padding = 'same', kernel_initializer = 'he_normal', name = "conv1d_1")(mask)
        
        if conv2_filters > 0:
            final_output_layer = Conv1D(conv2_filters, conv_kernel_size, padding = 'same', kernel_initializer = 'he_normal', name = "conv1d_2")(final_output_layer)
        final_output_layer = Bidirectional(LSTM(num_cells, name = "bi_lstm"))(final_output_layer)
    else:
        final_output_layer = Bidirectional(LSTM(num_cells, name = "bi_lstm"))(mask)

    if dropout > 0:
        final_output_layer = Dropout(dropout, name = "dropout")(final_output_layer)

    final_output_layer = Dense(1, activation = 'sigmoid', name = "output_dense")(final_output_layer)
    
    model = Model(inputs = model_input, outputs = final_output_layer)
    return model

def only_amino_di_tri_model(lstm1 = 5, lstm2 = 5, dense = 15, dropout = 0.2, lambda2 = 0.0, mask_value = 2):
       # Instantiate separate submodels.
       inputs = []
       outputs = []
       for i in range(3):
           input1, output1 = _one_prop_model(lstm1, lstm2, dense, dropout, lambda2, mask_value) 
           inputs.append(input1)
           outputs.append(output1) 

       # Merge the submodels.
       merge_layer = Concatenate()(
           outputs
       )

       final_output_layer = Dense(1, activation = 'sigmoid')(merge_layer)

       # The final model accepts three lists of AP scores as input, and returns a number in range [0, 1] which
       # indicates how probable it is the input sequence has self assembly.
       model = Model(inputs = inputs, outputs = final_output_layer)
       return model
 
def amino_di_tri_model(input_shape, conv = 64, kernel_size = 6, numcells = 128, lstm1 = 5, lstm2 = 5, dense = 15, dropout = 0.2, lambda2 = 0.0, mask_value = 2):
    # Instantiate separate submodels.
    inputs = []
    outputs = []
    for i in range(3):
        input1, output1 = _one_prop_model(lstm1, lstm2, dense, dropout, lambda2, mask_value) 
        inputs.append(input1)
        outputs.append(output1)
        
    input1, output1 = _create_seq_model(input_shape = input_shape, conv1_filters = conv, conv2_filters = conv, conv_kernel_size = kernel_size, num_cells = numcells, dropout = dropout, mask_value = mask_value)
    inputs.append(input1)
    outputs.append(output1)

    # Merge the submodels.
    merge_layer = Concatenate()(
        outputs
    )

    final_output_layer = Dense(1, activation = 'sigmoid')(merge_layer)

    # The final model accepts three lists of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    model = Model(inputs = inputs, outputs = final_output_layer)
    return model