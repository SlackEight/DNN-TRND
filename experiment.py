import utils.polar_pla as pla
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import models
import progressbar
import time
import torch.optim as optim

# ---------- DATA PREPROCESSING ---------- #
output_file = 'results.txt'
# filenames of the datasets
file_names = ['CTtemp.csv','snp500.csv', 'hpc.csv']#,'hpc.csv', 'sin.csv']
max_errors = [6000, 10, 5000]
filter_size = [5,10,40]
epochs_per_set = [2000,200,400]#[2000,100,200]
data_sets = []
# now we need pre-process the data
for i in range(len(file_names)):
    name = file_names[i]
    # read in the time series
    f = open("DataSets/"+name, 'r')
    time_series = []
    for line in f:
        time_series.append(float(line))

    # apply median filter
    time_series = pla.median_filter(time_series, filter_size[i])

    # apply sliding window piecewise linear segmentation
    x, _ = pla.sliding_window_pla(time_series, max_errors[i])
    data_sets.append(x)

if torch.cuda.is_available():  
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")
#dev = torch.device("cpu")
print(f"Using {dev}")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# now we need to define two different set creation methods for CNNs and RNNs
train_proportion = 0.7
def sliding_window_MLP(data, seq_length):
    inputs = []
    outputs = []
    for i in range(0, len(data)-seq_length*2, 2):
        inputs.append(data[i:(i+seq_length*2)]) # the next n are the input
        outputs.append(data[i+seq_length*2:i+seq_length*2+1]) # and the one after that is the output
    return Variable(torch.cuda.FloatTensor(np.array(inputs)).to(dev)), Variable(torch.cuda.FloatTensor(np.array(outputs)).to(dev))

def sliding_window_CNN(data, seq_length):
    inputs = []
    outputs = []
    for i in range(0, len(data)-seq_length*2, 2):
        temp = data[i:(i+seq_length*2)]
        new = []
        for x in range(0,len(temp),2):
            new.append([temp[x],temp[x+1]])
        inputs.append(new)
        outputs.append(data[i+seq_length*2:i+seq_length*2+1]) # and the one after that is the output
    return Variable(torch.cuda.FloatTensor(np.array(inputs)).to(dev)), Variable(torch.cuda.FloatTensor(np.array(outputs)).to(dev))

def sliding_window_RNN(data, seq_length):
    seq_length *= 2
    inputs = []
    outputs = []
    for i in range(0, len(data)-seq_length, 2):
        inputs.append(np.array(data[i:(i+seq_length)]).reshape(int(seq_length/2),2))
        outputs.append(np.array(data[i+seq_length:i+seq_length+1]))
    return Variable(torch.cuda.FloatTensor(inputs).to(dev)), Variable(torch.cuda.FloatTensor(outputs).to(dev))

def dataload(window_func ,batch_size, data, seq_len):
    # convert data to tensor, and apply dataloader
    total_data_input, total_data_output = window_func(data, seq_len)
    train_size = int(len(total_data_input)*train_proportion)

    training_data_input = torch.narrow(total_data_input, 0, 0, train_size)
    training_data_output = torch.narrow(total_data_output, 0, 0, train_size)

    validation_index = int((len(total_data_input) - train_size)*0.5) #Calculates how many data points in the validation set
    testing_index = len(total_data_input) - train_size - validation_index

    validation_data_input = torch.narrow(total_data_input, 0, train_size, validation_index).to(dev)
    validation_data_output = torch.narrow(total_data_output, 0, train_size, validation_index).to(dev)

    testing_data_input = torch.narrow(total_data_input, 0, train_size+validation_index, testing_index).to(dev)
    testing_data_output = torch.narrow(total_data_output, 0, train_size+validation_index, testing_index).to(dev)

    train = torch.utils.data.TensorDataset(training_data_input, training_data_output)
    validate = torch.utils.data.TensorDataset(validation_data_input, validation_data_output)
    test = torch.utils.data.TensorDataset(testing_data_input, testing_data_output)

    trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    validateset = torch.utils.data.DataLoader(validate, batch_size=batch_size, shuffle=False)
    testset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    return trainset, validateset, testset

def train_model(model, train_set, validation_set, test_set, learning_rate, epochs):
    import torch.optim as optim
    train_loss = []
    validation_loss = []
    model.train()
    epoch_total_trainloss = 0 # the total loss for each epoch, used for plotting
    min_val_loss_epoch = 0 # the epoch with the lowest validation loss
    min_val_loss = 9999999 # the lowest validation loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    validation_direction_accuracy = []
    for epoch in range(epochs+1):

        epoch_total_trainloss = 0 # reset this for the validation epoch'''
        model.train()
        for data in train_set:  # for each batch
            features, labels = data  # split the batches up into their features and labels
            model.zero_grad()
            output = model(features) # get a prediction from the model
            output = output
            #print(output.shape)
            loss = F.mse_loss(output, labels)  # calculate the loss of our prediction
            loss.backward()  # backpropogate the loss
            optimizer.step()  # optimize weights
            epoch_total_trainloss += loss.item()/len(train_set)
            torch.cuda.synchronize()
        train_loss.append(epoch_total_trainloss) # add this epoch's loss in order to plot it later
        epoch_total_trainloss = 0 # reset this for the validation epoch

        # now we'll calculate the direction accuracy for the training and validation sets
        correct=0
        total_points = 0
        model.eval()
        for data in validation_set:
            
            inputs, labels = data
            output = model(inputs)
            total_points += len(output)
            '''for i in range(len(output)):
                pred = output[i]
                actual = labels[i]
                if pred < 0 and actual < 0 or pred > 0 and actual > 0: #or (pred-actual)<0.01:
                    correct += 1
                #print(output[0],labels[0])'''
            loss = F.mse_loss(output, labels)  # calculate the loss of our prediction
            epoch_total_trainloss += loss.item()/len(validation_set)
            torch.cuda.synchronize()
        
        if epoch_total_trainloss < min_val_loss:
            #torch.save(model.state_dict(), 'TCN temp.pt')
            min_val_loss = epoch_total_trainloss
            min_val_loss_epoch = epoch
        validation_direction_accuracy.append(correct/(total_points))
        validation_loss.append(epoch_total_trainloss) # we'll need to plot validation loss too
    #print(f"Lowest validation loss: {min_val_loss} at epoch {min_val_loss_epoch}")
    return min_val_loss



# ------------ MLP ------------ #

hyperparemeters = [[64,128], # number of nodes per hidden layer
                   [0.0001,0.001], # learning rate
                   [64, 128,256], # batch size
                   [4,6,8], # sequence length
                   [0.1,0.2] # dropout
                   ]

total_models = 1
for set in hyperparemeters:
    total_models *= len(set)
total_models *= len(data_sets)
index = 0
print("\nStarting MLP grid optimisation...")
start_time = time.time()

bar = progressbar.ProgressBar(maxval=total_models+1, \
    widgets=[progressbar.Bar('█', '|', '|'), ' ', progressbar.Percentage()])
bar.start()
for i in range(len(data_sets)):
    data_set = data_sets[i]
    lowest_validation = 999999
    best_hyper_params = []
    for batch_size in hyperparemeters[2]:
        for seq_length in hyperparemeters[3]:
            # load in data
            trainset, validationset, testset = dataload(sliding_window_MLP, batch_size, data_set, seq_length)

            for lr in hyperparemeters[1]:
                for nodes in hyperparemeters[0]:

                    
                    for dropout in hyperparemeters[4]:
                        index += 1
                        # create the model 
                        model = models.MLP(seq_length*2, nodes, dropout).to(dev)
                        bar.update(index)
                        val_loss = train_model(model, trainset, validationset, testset, lr, epochs_per_set[i])

                        if val_loss < lowest_validation:
                            lowest_validation = val_loss
                            best_hyper_params = [nodes, lr, batch_size, seq_length, dropout]
    f = open(output_file, 'a')
    f.write(f'MLP - Lowest validation loss: {lowest_validation} with following hyperparams: \n'
            +f'Nodes: {best_hyper_params[0]}\n'
            +f'LR: {best_hyper_params[1]}\n'
            +f'Batch Size: {best_hyper_params[2]}\n '
            +f'Sequence Length: {best_hyper_params[3]}\n '
            +f'Dropout:{best_hyper_params[4]}\n\n')
    f.close()
bar.finish()
elapsed = time.time()-start_time
print(f'MLP optimisation completed, results stored in {output_file}'
     +f'\nModels Considered: {total_models}'
     +f'\nCompletion Time: {int((elapsed/60)//60)}:{int(elapsed//60)}:{round(elapsed%60)}s\n\n')
        


# ------------ CNN ------------ #

hyperparemeters = [[64,128], # number of nodes per hidden layer
                   [0.0001,0.001,0.01], # learning rate
                   [64,128,256], # batch size
                   [4,6], # sequence length
                   [0.1,0.2,0.3], # dropout
                   ]

total_models = 1
for set in hyperparemeters:
    total_models *= len(set)
total_models *= len(data_sets)
index = 0
print("Starting CNN grid optimisation...")
start_time = time.time()
bar = progressbar.ProgressBar(maxval=total_models+1, \
    widgets=[progressbar.Bar('█', '|', '|'), ' ', progressbar.Percentage()])
bar.start()
# grid optimise
for i in range(len(data_sets)):
    data_set = data_sets[i]
    lowest_validation = 999999
    best_hyper_params = []
    for batch_size in hyperparemeters[2]:
        for seq_length in hyperparemeters[3]:
            # load in data
            trainset, validationset, testset = dataload(sliding_window_CNN, batch_size, data_set, seq_length)
            for lr in hyperparemeters[1]:
                for nodes in hyperparemeters[0]:      
                    for dropout in hyperparemeters[4]:
                        # create the model 
                        model = models.CNN(seq_length, nodes, dropout, 2).to(dev)
                        index+=1
                        bar.update(index)
                        val_loss = train_model(model, trainset, validationset, testset, lr,epochs_per_set[i])

                        if val_loss < lowest_validation:
                            lowest_validation = val_loss
                            best_hyper_params = [nodes, lr, batch_size, seq_length, dropout]

    f = open(output_file, 'a')
    f.write(f'CNN -- Lowest validation loss: {lowest_validation} with following hyperparams: \n'
            +f'Nodes: {best_hyper_params[0]}\n'
            +f'LR: {best_hyper_params[1]}\n'
            +f'Batch Size: {best_hyper_params[2]}\n '
            +f'Sequence Length: {best_hyper_params[3]}\n '
            +f'Dropout:{best_hyper_params[4]}\n\n')
    f.close()
bar.finish()
elapsed = time.time()-start_time
print(f'CNN optimisation completed, results stored in {output_file}'
     +f'\nModels Considered: {total_models}'
     +f'\nCompletion Time: {int((elapsed/60)//60)}:{int(elapsed//60)}:{round(elapsed%60)}s\n\n')



# ------------ TCN ------------ #

hyperparemeters = [[64,128], # number of nodes per hidden layer
                   [0.001,0.01], # learning rate
                   [128,256], # batch size
                   [4,6,8], # sequence length
                   [0.1,0.2], # dropout
                   [2,4]] # kernel size]

print("Starting TCN grid optimisation...")
start_time = time.time()
total_models = 1
for set in hyperparemeters:
    total_models *= len(set)
total_models *= len(data_sets)
index = 0
bar = progressbar.ProgressBar(maxval=total_models+1, \
    widgets=[progressbar.Bar('█', '|', '|'), ' ', progressbar.Percentage()])
bar.start()
for i in range(len(data_sets)):
    data_set = data_sets[i]
    lowest_validation = 999999
    best_hyper_params = []
    for batch_size in hyperparemeters[2]:
        for seq_length in hyperparemeters[3]:
            # load in data
            trainset, validationset, testset = dataload(sliding_window_CNN, batch_size, data_set, seq_length)
            for lr in hyperparemeters[1]:
                for nodes in hyperparemeters[0]:      
                    for dropout in hyperparemeters[4]:
                        for kernel_size in hyperparemeters[5]:
                            # create the model 
                            model = models.TCN(seq_length,1, [nodes]*2, kernel_size, dropout).to(dev)
                            index+=1
                            bar.update(index)
                            val_loss = train_model(model, trainset, validationset, testset, lr, epochs_per_set[i])

                            if val_loss < lowest_validation:
                                lowest_validation = val_loss
                                best_hyper_params = [nodes, lr, batch_size, seq_length, dropout, kernel_size]

    f = open(output_file, 'a')
    f.write(f'TCN -- Lowest validation loss: {lowest_validation} with following hyperparams: \n'
            +f'Nodes: {best_hyper_params[0]}\n'
            +f'LR: {best_hyper_params[1]}\n'
            +f'Batch Size: {best_hyper_params[2]}\n '
            +f'Sequence Length: {best_hyper_params[3]}\n '
            +f'Dropout:{best_hyper_params[4]}\n'
            +f'Kernel Size: {best_hyper_params[5]}\n\n')
    f.close()
bar.finish()
elapsed = time.time()-start_time
print(f'TCN optimisation completed, results stored in {output_file}'
     +f'\nModels Considered: {total_models}'
     +f'\nCompletion Time: {int((elapsed/60)//60)}:{int(elapsed//60)}:{round(elapsed%60)}s\n\n')



# ------------ RNN ------------ #
hyperparemeters = [[64,128],  # number of nodes per hidden layer
                   [0.001,0.01], # learning rate
                   [64,128,256], # batch size
                   [4,6,8], # sequence length
                   [0.3,0.5,0.7] # dropout
                   ]

print("Starting RNN grid optimisation...")
start_time = time.time()
total_models = 1
for set in hyperparemeters:
    total_models *= len(set)
total_models *= len(data_sets)
index = 0
bar = progressbar.ProgressBar(maxval=total_models+1, \
    widgets=[progressbar.Bar('█', '|', '|'), ' ', progressbar.Percentage()])
bar.start()
for i in range(len(data_sets)):
    data_set = data_sets[i]
    lowest_validation = 999999
    best_hyper_params = []
    
    for nodes in hyperparemeters[0]:
        for lr in hyperparemeters[1]:
            for batch_size in hyperparemeters[2]:
                for seq_length in hyperparemeters[3]:
                    trainset, validationset, testset = dataload(sliding_window_RNN, batch_size, data_set, seq_length)
                    for dropout in hyperparemeters[4]:

                        # create the model 
                        model = models.RNN(1, 2, nodes, 1, dropout)
                        index+=1
                        bar.update(index)
                        val_loss = train_model(model, trainset, validationset, testset, lr, epochs_per_set[i])

                        if val_loss < lowest_validation:
                            lowest_validation = val_loss
                            best_hyper_params = [nodes, lr, batch_size, seq_length, dropout]

    f = open(output_file, 'a')
    f.write(f'RNN -- Lowest validation loss: {lowest_validation} with following hyperparams: \n'
            +f'Nodes: {best_hyper_params[0]}\n'
            +f'LR: {best_hyper_params[1]}\n'
            +f'Batch Size: {best_hyper_params[2]}\n '
            +f'Sequence Length: {best_hyper_params[3]}\n '
            +f'Dropout:{best_hyper_params[4]}\n\n')
    f.close()
bar.finish()
elapsed = time.time()-start_time
print(f'RNN optimisation completed, results stored in {output_file}'
     +f'\nModels Considered: {total_models}'
     +f'\nCompletion Time: {int((elapsed/60)//60)}:{int(elapsed//60)}:{round(elapsed%60)}s\n\n')


# ------------ LSTM ------------ #

hyperparemeters = [[64,128],  # number of nodes per hidden layer
                   [0.001,0.01], # learning rate
                   [64,128,256], # batch size
                   [4,6], # sequence length
                   [0.3,0.5] # dropout
                   ]

print("Starting LSTM grid optimisation...")
start_time = time.time()
total_models = 1
for set in hyperparemeters:
    total_models *= len(set)
total_models *= len(data_sets)
index = 0
bar = progressbar.ProgressBar(maxval=total_models+1, \
    widgets=[progressbar.Bar('█', '|', '|'), ' ', progressbar.Percentage()])
bar.start()
for i in range(len(data_sets)):
    data_set = data_sets[i]
    lowest_validation = 999999
    best_hyper_params = []
    
    for nodes in hyperparemeters[0]:
        for lr in hyperparemeters[1]:
            for batch_size in hyperparemeters[2]:
                for seq_length in hyperparemeters[3]:
                    trainset, validationset, testset = dataload(sliding_window_RNN, batch_size, data_set, seq_length)
                    for dropout in hyperparemeters[4]:

                        # create the model 
                        model = models.LSTM(1, 2, nodes, 1, dropout)
                        index+=1
                        bar.update(index)
                        val_loss = train_model(model, trainset, validationset, testset, lr, epochs_per_set[i])

                        if val_loss < lowest_validation:
                            lowest_validation = val_loss
                            best_hyper_params = [nodes, lr, batch_size, seq_length, dropout]

    f = open(output_file, 'a')
    f.write(f'LSTM -- Lowest validation loss: {lowest_validation} with following hyperparams: \n'
            +f'Nodes: {best_hyper_params[0]}\n'
            +f'LR: {best_hyper_params[1]}\n'
            +f'Batch Size: {best_hyper_params[2]}\n '
            +f'Sequence Length: {best_hyper_params[3]}\n '
            +f'Dropout:{best_hyper_params[4]}\n\n')
    f.close()
bar.finish()
elapsed = time.time()-start_time
print(f'LSTM optimisation completed, results stored in {output_file}'
     +f'\nModels Considered: {total_models}'
     +f'\nCompletion Time: {int((elapsed/60)//60)}:{int(elapsed//60)}:{round(elapsed%60)}s\n\n')


# ------------ Bi-LSTM ------------ #



print("Starting BiLSTM grid optimisation...")
start_time = time.time()
total_models = 1
for set in hyperparemeters:
    total_models *= len(set)
total_models *= len(data_sets)
index = 0
bar = progressbar.ProgressBar(maxval=total_models+1, \
    widgets=[progressbar.Bar('█', '|', '|'), ' ', progressbar.Percentage()])
bar.start()
for i in range(len(data_sets)):
    data_set = data_sets[i]
    lowest_validation = 999999
    best_hyper_params = []
    
    for nodes in hyperparemeters[0]:
        for lr in hyperparemeters[1]:
            for batch_size in hyperparemeters[2]:
                for seq_length in hyperparemeters[3]:
                    trainset, validationset, testset = dataload(sliding_window_RNN, batch_size, data_set, seq_length)
                    for dropout in hyperparemeters[4]:

                        # create the model 
                        model = models.BiLSTM(1, 2, nodes, 1, dropout)
                        index+=1
                        bar.update(index)
                        val_loss = train_model(model, trainset, validationset, testset, lr, epochs_per_set[i])

                        if val_loss < lowest_validation:
                            lowest_validation = val_loss
                            best_hyper_params = [nodes, lr, batch_size, seq_length, dropout]

    f = open(output_file, 'a')
    f.write(f'BiLSTM -- Lowest validation loss: {lowest_validation} with following hyperparams: \n'
            +f'Nodes: {best_hyper_params[0]}\n'
            +f'LR: {best_hyper_params[1]}\n'
            +f'Batch Size: {best_hyper_params[2]}\n '
            +f'Sequence Length: {best_hyper_params[3]}\n '
            +f'Dropout:{best_hyper_params[4]}\n')
    f.close()
bar.finish()
elapsed = time.time()-start_time
print(f'BiLSTM optimisation completed, results stored in {output_file}'
     +f'\nModels Considered: {total_models}'
     +f'\nCompletion Time: {int((elapsed/60)//60)}:{int(elapsed//60)}:{round(elapsed%60)}s\n\n')
