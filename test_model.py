from utils.dnn_methods import *
from utils.polar_pla import preprocess
from models import *

''' This script should be used to test a configuration. Below are the datasets. 
        Fields which can be modified between tests are marked with an x.'''
    #----------1----------# #---------2---------#  #--------3--------#
    # "DataSets/CTtemp.csv" "DataSets/snp500.csv" "DataSets/hpc.csv" 
    #          5                      10                   40
    #         6000                    10                  5000

datasets = [["DataSets/CTtemp.csv",5,6000],["DataSets/snp500.csv",10,10],["DataSets/hpc.csv",40,5000]]

train_proportion = 0.7 # keep this constant across tests
models_to_average = 10 # keep this constant across tests

######## your test goes here #########

# dataset and model type #
dataset = datasets[0]  # Change this to test different datasets.                                            # x 
component = 0  # 0 to predict trend, 1 to predict duration, 2 for a dual approach (trend and duration)      # x

# hyperparameters #
hidden_size = 64                                                                                            # x
lr = 0.001                                                                                                  # x
batch_size = 64                                                                                             # x
seq_length = 8                                                                                              # x
dropout = 0.2                                                                                               # x
training_epochs = 500                                                                                       # x
# TCN only:
kernel_size = 4                                                                                             # x
n_layers = 3                                                                                                # x

trends = preprocess(dataset[0], dataset[1], dataset[2])


# now just simply uncomment the model you'd like to test:

model = MLP(seq_length*2, hidden_size, max(1,component), dropout).to(dev)
#model = CNN(seq_length, hidden_size, max(1,component), 2, dropout).to(dev)
#model = TCN(seq_length,max(1, component), [hidden_size]*n_layers, kernel_size, dropout).to(dev)
#model = LSTM(seq_length, hidden_size, max(1,component), dropout).to(dev)
#model = RNN(max(1,component), 2, hidden_size, 1, dropout).to(dev)
#model = LSTM(max(1,component), 2, hidden_size, 1, dropout).to(dev)
#model = BiLSTM(max(1,component), 2, hidden_size, 1, dropout).to(dev)


train_and_test(model, trends, train_proportion, models_to_average, lr, batch_size, seq_length, training_epochs, component)




