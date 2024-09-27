import tensorflow as tf 
import models
from utils import reshape_for_model, data_and_labels_from_indices, MODELS_PATH, TMP_MODELS_PATH
import numpy as np
import os

BATCH_SIZE = 600
LEARNING_RATE_SET = 0.01
DROPOUT = 0.5
LSTM = 5
CONV = 5
LAMBDA = 0.0

hyperparameter_numcells = [32, 48, 64]
hyperparameter_kernel_size = [4, 6, 8]

# This function keeps the initial learning rate for the first ten epochs.
# The learning rate decreases exponentially after the first ten epochs.
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

def return_callbacks(model_file, metric):
    callbacks = [
        # Save the best model (the one with the lowest value for the specified metric).
        tf.keras.callbacks.ModelCheckpoint(
            model_file, save_best_only = True, monitor = metric, mode = 'min'
        ), 
        tf.keras.callbacks.LearningRateScheduler(scheduler)
    ]
    return callbacks
 
def hyperparameter_tuning(model_name, test_number, train_and_validation_data, train_and_validation_labels, kfold_second, epochs, factor_NSA, mask_value = 2):
    params_nr = 0 
    min_val_loss = 1000
     
    best_numcells = 0
    best_kernel = 0  
    
    if not os.path.isdir(TMP_MODELS_PATH + model_name):
      os.makedirs(TMP_MODELS_PATH + model_name)

    indices = []
    for train_data_indices, validation_data_indices in kfold_second.split(train_and_validation_data, train_and_validation_labels): 
        indices.append([train_data_indices, validation_data_indices])
     
    for numcells in hyperparameter_numcells: 
      for kernel in hyperparameter_kernel_size:   
        params_nr += 1
        fold_nr = 0 
        history_val_loss = []  
        
        for pair in indices:   
            
            train_data_indices = pair[0]
            
            validation_data_indices = pair[1]
            
            fold_nr += 1 

            # Convert train indices to train data and train labels.
            train_data, train_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, train_data_indices)
            train_data, train_labels = reshape_for_model(model_name, train_data, train_labels)

            # Convert validation indices to validation data and validation labels.
            val_data, val_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, validation_data_indices)
            val_data, val_labels = reshape_for_model(model_name, val_data, val_labels)
              
            # Choose correct model and instantiate model.
            if "AP" in model_name and "SP" in model_name:
              model = models.amino_di_tri_model(input_shape = np.shape(train_data[3][0]), conv = CONV, numcells = numcells, kernel_size = kernel, lstm1 = LSTM, lstm2 = LSTM, dense =2 * numcells, dropout = DROPOUT, lambda2 = LAMBDA, mask_value = mask_value)

            if "AP" in model_name and "SP" not in model_name:
              model = models.only_amino_di_tri_model(lstm1 = LSTM, lstm2 = LSTM, dense =2 * numcells, dropout = DROPOUT, lambda2 = LAMBDA, mask_value = mask_value)

            if "AP" not in model_name and "SP" in model_name:
              model = models.create_seq_model(input_shape = np.shape(train_data[0]), conv1_filters = CONV, conv2_filters = CONV, conv_kernel_size = kernel, num_cells = numcells, dropout = DROPOUT, mask_value = mask_value)

            # Print model summary.
            model.summary()

            callbacks = return_callbacks(TMP_MODELS_PATH + model_name + "/" + model_name + "_test_" + str(test_number) + "_fold_" + str(fold_nr) + "_params_" + str(params_nr) + "_num_cells_" + str(numcells) + "_kernel_size_" + str(kernel) + ".h5", "val_loss") 

            optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE_SET)

            model.compile(
                optimizer = optimizer,
                loss = 'binary_crossentropy',
                metrics = ['accuracy']
            ) 
              
            # Train the model.
            # After model training, the `history` variable will contain important parameters for each epoch, such as
            # train loss, train accuracy, learning rate, and so on.
            history = model.fit(
                    train_data, 
                    train_labels, 
                    validation_data = [val_data, val_labels],
                    class_weight = {0: factor_NSA, 1: 1.0}, 
                    epochs = epochs,
                    batch_size = BATCH_SIZE,
                    callbacks = callbacks,
                    verbose = 1
                ) 

            history_val_loss += history.history['val_loss']
            
        avg_val_loss = np.mean(history_val_loss)  
        
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss 
            best_numcells = numcells
            best_kernel = kernel  
            
    print("%s Test %d Best params: num_cells: %d kernel_size: %d" % (model_name, test_number, best_numcells, best_kernel))
           
    train_and_validation_data, train_and_validation_labels = reshape_for_model(model_name, train_and_validation_data, train_and_validation_labels)
      
    # Choose correct model and instantiate model. 
    if "AP" in model_name and "SP" in model_name:
      model = models.amino_di_tri_model(input_shape = np.shape(train_and_validation_data[3][0]), conv = CONV, numcells = best_numcells, kernel_size = best_kernel, lstm1 = LSTM, lstm2 = LSTM, dense =2 * best_numcells, dropout = DROPOUT, lambda2 = LAMBDA, mask_value = mask_value)

    if "AP" in model_name and "SP" not in model_name:
      model = models.only_amino_di_tri_model(lstm1 = LSTM, lstm2 = LSTM, dense =2 * best_numcells, dropout = DROPOUT, lambda2 = LAMBDA, mask_value = mask_value)

    if "AP" not in model_name and "SP" in model_name:
      model = models.create_seq_model(input_shape = np.shape(train_and_validation_data[0]), conv1_filters = CONV, conv2_filters = CONV, conv_kernel_size = best_kernel, num_cells = best_numcells, dropout = DROPOUT, mask_value = mask_value)
 
    # Print model summary.
    model.summary()
    
    callbacks = return_callbacks(TMP_MODELS_PATH + model_name + "/" + model_name + "_test_" + str(test_number) +  "_num_cells_" + str(best_numcells) + "_kernel_size_" + str(best_kernel) + ".h5", "loss") 

    optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE_SET)

    model.compile(
        optimizer = optimizer,
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    ) 
      
    # Train the model.
    # After model training, the `history` variable will contain important parameters for each epoch, such as
    # train loss, train accuracy, learning rate, and so on.
    history = model.fit(
            train_and_validation_data, 
            train_and_validation_labels,  
            class_weight = {0: factor_NSA, 1: 1.0}, 
            epochs = epochs,
            batch_size = BATCH_SIZE,
            callbacks = callbacks,
            verbose = 1
        )
    
def model_training(model_name, data, labels, best_numcells, best_kernel, epochs, factor_NSA, mask_value = 2):
      
    if not os.path.isdir(MODELS_PATH):
      os.makedirs(MODELS_PATH) 
            
    print("%s Params: num_cells: %d kernel_size: %d" % (model_name, best_numcells, best_kernel))
           
    data, labels = reshape_for_model(model_name, data, labels)
      
    # Choose correct model and instantiate model. 
    if "AP" in model_name and "SP" in model_name:
      model = models.amino_di_tri_model(input_shape = np.shape(data[3][0]), conv = CONV, numcells = best_numcells, kernel_size = best_kernel, lstm1 = LSTM, lstm2 = LSTM, dense =2 * best_numcells, dropout = DROPOUT, lambda2 = LAMBDA, mask_value = mask_value)

    if "AP" in model_name and "SP" not in model_name:
      model = models.only_amino_di_tri_model(lstm1 = LSTM, lstm2 = LSTM, dense =2 * best_numcells, dropout = DROPOUT, lambda2 = LAMBDA, mask_value = mask_value)

    if "AP" not in model_name and "SP" in model_name:
      model = models.create_seq_model(input_shape = np.shape(data[0]), conv1_filters = CONV, conv2_filters = CONV, conv_kernel_size = best_kernel, num_cells = best_numcells, dropout = DROPOUT, mask_value = mask_value)
 
    # Print model summary.
    model.summary()
    
    callbacks = return_callbacks(MODELS_PATH + model_name + ".h5", "loss") 

    optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE_SET)

    model.compile(
        optimizer = optimizer,
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    ) 
      
    # Train the model.
    # After model training, the `history` variable will contain important parameters for each epoch, such as
    # train loss, train accuracy, learning rate, and so on.
    history = model.fit(
            data, 
            labels,  
            class_weight = {0: factor_NSA, 1: 1.0}, 
            epochs = epochs,
            batch_size = BATCH_SIZE,
            callbacks = callbacks,
            verbose = 1
        )