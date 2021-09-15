import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import models

if torch.cuda.is_available():  
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")

#print(f"Using {dev}")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

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

def sliding_window_MLP(data, seq_length, component):
    inputs = []
    outputs = []
    for i in range(0, len(data)-seq_length*2, 2):
        inputs.append(data[i:(i+seq_length*2)]) # the next n are the input

        outputs.append(data[i+seq_length*2+component%2:i+seq_length*2+min(component+1,2)]) # and the one after that is the output
    return Variable(torch.cuda.FloatTensor(np.array(inputs)).to(dev)), Variable(torch.cuda.FloatTensor(np.array(outputs)).to(dev))

def sliding_window_CNN(data, seq_length, component):
    inputs = []
    outputs = []
    for i in range(0, len(data)-seq_length*2, 2):
        temp = data[i:(i+seq_length*2)]
        new = []
        for x in range(0,len(temp),2):
            new.append([temp[x],temp[x+1]])
        inputs.append(new)
        outputs.append(data[i+seq_length*2+component%2:i+seq_length*2+min(component+1,2)]) # and the one after that is the output
    return Variable(torch.cuda.FloatTensor(np.array(inputs)).to(dev)), Variable(torch.cuda.FloatTensor(np.array(outputs)).to(dev))

def sliding_window_RNN(data, seq_length, component):
    seq_length *= 2
    inputs = []
    outputs = []
    for i in range(0, len(data)-seq_length, 2):
        inputs.append(np.array(data[i:(i+seq_length)]).reshape(int(seq_length/2),2))
        outputs.append(np.array(data[i+seq_length+component%2:i+seq_length+min(component+1,2)]))
    return Variable(torch.cuda.FloatTensor(inputs).to(dev)), Variable(torch.cuda.FloatTensor(outputs).to(dev))

def dataload(window_func ,batch_size, data, seq_len, train_proportion, component):
    # convert data to tensor, and apply dataloader
    total_data_input, total_data_output = window_func(data, seq_len, component)
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



def test_model(model, trainset, validateset, testset, learning_rate, component, training_epochs, models_to_average):
    import math
    import statistics
    res_1 = []
    res_2 = []
    fp = 0 # false positives
    tp = 0 # true positives
    fn = 0 # false negatives
    tn = 0 # true negatives
    total_dir_acc = 0
    for i in range(models_to_average):
        epochs = training_epochs
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
            for data in trainset:  # for each batch
                features, labels = data  # split the batches up into their features and labels
                model.zero_grad()
                output = model(features) # get a prediction from the model
                loss = F.mse_loss(output, labels)  # calculate the loss of our prediction
                loss.backward()  # backpropogate the loss
                optimizer.step()  # optimize weights
                epoch_total_trainloss += loss.item()/len(trainset)
                torch.cuda.synchronize()
            train_loss.append(epoch_total_trainloss) # add this epoch's loss in order to plot it later
            epoch_total_trainloss = 0 # reset this for the validation epoch

            # now we'll calculate the direction accuracy for the training and validation sets
            correct=0
            total_points = 0
            model.eval()
            for data in validateset:
                
                inputs, labels = data
                output = model(inputs)
                total_points += len(output)
                for i in range(len(output)):
                    pred = output[i]
                    actual = labels[i]
                    #if pred < 0 and actual < 0 or pred > 0 and actual > 0: #or (pred-actual)<0.01:
                    #    correct += 1
                    #print(output[0],labels[0])
                loss = F.mse_loss(output, labels)  # calculate the loss of our prediction
                epoch_total_trainloss += loss.item()/len(validateset)
                torch.cuda.synchronize()
            if epoch_total_trainloss < min_val_loss:
                torch.save(model.state_dict(), 'temp.pt')
                min_val_loss = epoch_total_trainloss
                min_val_loss_epoch = epoch
            validation_direction_accuracy.append(correct/(total_points))
            validation_loss.append(epoch_total_trainloss) # we'll need to plot validation loss too
        #import matplotlib.pyplot as plt
        #plt.plot(train_loss)
        #plt.plot(validation_loss)
        #plt.show()
        #plt.plot(validation_direction_accuracy)
        #plt.show()
        #print(f"Lowest validation loss: {min_val_loss} at epoch {min_val_loss_epoch}")

        model.load_state_dict(torch.load('temp.pt'))
        model.eval()
        correct=0
        output_file = open("utils/angles.txt", "w")
        for data in trainset:

            inputs, labels = data
            output = model(inputs)
            for i in range(len(output)):
                pred = output[i]
                #output_file.write(str(pred.item()*90)+"\n")
        total_loss = 0
        for data in validateset:
            inputs, labels = data
            output = model(inputs)
            model.zero_grad()
            total_loss += F.mse_loss(output, labels).item()/len(validateset)
        #print(f'Directional Accuracy: {correct*100/len(test)} MSE on validate set: {total_loss}')

        total_loss = 0
        total_loss_slope = 0
        total_loss_length = 0

        for data in testset:
            inputs, labels = data
            output = model(inputs)

            # test for the dual model
            if component == 2:
                output_slopes = []
                for out in labels:
                    output_slopes.append(np.array([out[0]]))
                output_slopes = Variable(torch.FloatTensor(output_slopes))

                output_lengths = []
                for out in labels:
                    output_lengths.append(np.array([out[1]]))
                output_lengths = Variable(torch.FloatTensor(output_lengths))

                pred_slopes = []
                for out in output:
                    pred_slopes.append(np.array([out[0]]))
                pred_slopes = Variable(torch.FloatTensor(pred_slopes))

                pred_lengths = []
                for out in output:
                    pred_lengths.append(np.array([out[1]]))
                pred_lengths = Variable(torch.FloatTensor(pred_lengths))
            
                for i in range(len(output)): # true and false directional classifications
                    pred = output[i][0]
                    actual = labels[i][0]
                    if pred > 0 and actual > 0 or (pred-actual)<0.022: # true positive with 2 degree lee way
                        tp += 1
                    elif pred < 0 and actual < 0: # true negative
                        tn += 1
                    elif pred > 0 and actual < 0: # false positive
                        fp += 1
                    elif pred < 0 and actual > 0: # false negative
                        fn += 1

                model.zero_grad()
                total_loss_slope += F.mse_loss(output_slopes,pred_slopes).item()/len(testset)
                model.zero_grad()
                total_loss_length += F.mse_loss(output_lengths, pred_lengths).item()/len(testset)
            
            # test for single model
            else:
                model.zero_grad()
                total_loss += F.mse_loss(output, labels).item()/len(testset)
                for i in range(len(output)): # directional accuracy check
                    pred = output[i][0]
                    actual = labels[i][0]
                    if pred > 0 and actual > 0 or (pred-actual)<0.022: # true positive with 2 degree lee way
                        tp += 1
                    elif pred < 0 and actual < 0: # true negative
                        tn += 1
                    elif pred > 0 and actual < 0: # false positive
                        fp += 1
                    elif pred < 0 and actual > 0: # false negative
                        fn += 1
                        

        #print(f'Directional Accuracy: {correct*100/len(test)} MSE test: {total_loss}, RMSE test: {math.sqrt(total_loss)}\n')
        #print(f'{math.sqrt(total_loss_slope)}, {math.sqrt(total_loss_length)}')
        #testing_file.write(f'{math.sqrt(total_loss_slope)},{math.sqrt(total_loss_length)}\n')

        if component == 2:
            res_1.append(math.sqrt(total_loss_slope))
            res_2.append(math.sqrt(total_loss_length))
            print(f'{math.sqrt(total_loss_slope)}, {math.sqrt(total_loss_length)}')
        else:
            res_1.append(math.sqrt(total_loss))
            print(f'{math.sqrt(total_loss)}')
    
    if component == 0 or component == 2:
        print(f'μ = {round(sum(res_1) / len(res_1 ),3)} | σ = {round(statistics.pstdev(res_1),3)} | tp = {tp} | tn = {tn} | fp = {fp} | fn = {fn}')

    if component == 1:
        print(f'μ = {round(sum(res_1) / len(res_1 ),3)} | σ = {round(statistics.pstdev(res_1),3)}')
    
    if component == 2:
        print(f'μ = {round(sum(res_2) / len(res_2 ),3)} | σ = {round(statistics.pstdev(res_2),3)}')


def train_and_test(model, trends, train_proportion, models_to_average, lr, batch_size, seq_length, training_epochs, component):
    trainset, validationset, testset = dataload(sliding_window_MLP ,batch_size, trends, seq_length, train_proportion, component)
    test_model(model, trainset, validationset, testset, lr, component, training_epochs, models_to_average)