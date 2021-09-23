import utils.polar_pla as pla
import torch
import models
import progressbar
import time
import utils.dnn_methods as dm

if torch.cuda.is_available():  
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")

# ---------- DATA PREPROCESSING ---------- #
output_file = 'results.txt'
# filenames of the datasets
file_names = ['CTtemp.csv','snp500.csv', 'hpc.csv']#,'hpc.csv', 'sin.csv']
max_errors = [6000, 10, 5000]
filter_size = [5,10,40]
epochs_per_set = [1000,50,200]#[2000,100,200]
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

#component = 0
models_to_average = 5
# now we need to define two different set creation methods for CNNs and RNNs
train_proportion = 0.7
for component in range(3):
    
    # ------------ MLP ------------ #
    hyperparemeters = [[32, 64, 128], # number of nodes per hidden layer
                    [0.001], # learning rate
                    [64], # batch size
                    [4,6,8,10], # sequence length
                    [0.0,0.1,0.2,0.3] # dropout
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
                trainset, validationset, testset = dm.dataload(dm.sliding_window_MLP, batch_size, data_set, seq_length, train_proportion, component)

                for lr in hyperparemeters[1]:
                    for nodes in hyperparemeters[0]:

                        
                        for dropout in hyperparemeters[4]:
                            index += 1
                            # create the model 
                            val_loss = 0
                            for jj in range(models_to_average):
                                model = models.MLP(seq_length*2, nodes, max(1,component), dropout).to(dev)
                                val_loss += dm.train_model(model, trainset, validationset, testset, lr, epochs_per_set[i])/models_to_average
                            bar.update(index)
                            if val_loss < lowest_validation:
                                lowest_validation = val_loss
                                best_hyper_params = [nodes, lr, batch_size, seq_length, dropout]
        f = open(output_file, 'a')
        f.write(f'MLP - Lowest validation loss: {lowest_validation} with following hyperparams: \n'
                +f'Nodes: {best_hyper_params[0]}\n'
                +f'LR: {best_hyper_params[1]}\n'
                +f'Batch Size: {best_hyper_params[2]}\n'
                +f'Sequence Length: {best_hyper_params[3]}\n'
                +f'Dropout:{best_hyper_params[4]}\n'
                +f'Dataset: {file_names[i]}')
        f.close()
        #Now let's run the experiment on it
        model = models.MLP(best_hyper_params[3]*2, best_hyper_params[0], max(1,component), best_hyper_params[4]).to(dev)
        dm.train_and_test(model, data_set, train_proportion, 20,best_hyper_params[1],best_hyper_params[2],best_hyper_params[3],epochs_per_set[i], component, f"MLP_test_{component}.txt")

    bar.finish()


    elapsed = time.time()-start_time
    print(f'MLP optimisation completed, results stored in {output_file}'
        +f'\nModels Considered: {total_models}'
        +f'\nCompletion Time: {int((elapsed/60)//60)}:{int(elapsed//60)%60}:{round(elapsed%60)}s\n\n')
            


    # ------------ CNN ------------ #

    hyperparemeters = [[32, 64,128], # number of nodes per hidden layer
                    [0.001, 0.01], # learning rate
                    [64], # batch size
                    [4,6,8,10], # sequence length
                    [0.0,0.1,0.2], # dropout
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
                trainset, validationset, testset = dm.dataload(dm.sliding_window_CNN, batch_size, data_set, seq_length,train_proportion, component)
                for lr in hyperparemeters[1]:
                    for nodes in hyperparemeters[0]:      
                        for dropout in hyperparemeters[4]:
                            # create the model 
                            
                            index+=1
                            bar.update(index)
                            val_loss = 0
                            for jj in range(models_to_average):
                                model = models.CNN(seq_length, nodes, max(1,component), 2, dropout).to(dev)
                                val_loss += dm.train_model(model, trainset, validationset, testset, lr,epochs_per_set[i])/models_to_average

                            if val_loss < lowest_validation:
                                lowest_validation = val_loss
                                best_hyper_params = [nodes, lr, batch_size, seq_length, dropout]

        f = open(output_file, 'a')
        f.write(f'CNN -- Lowest validation loss: {lowest_validation} with following hyperparams: \n'
                +f'Nodes: {best_hyper_params[0]}\n'
                +f'LR: {best_hyper_params[1]}\n'
                +f'Batch Size: {best_hyper_params[2]}\n '
                +f'Sequence Length: {best_hyper_params[3]}\n '
                +f'Dropout:{best_hyper_params[4]}\n'
                +f'Dataset: {file_names[i]}\n\n')
        f.close()
        model = models.CNN(best_hyper_params[3], best_hyper_params[0], max(1,component), 2, best_hyper_params[4]).to(dev)
        dm.train_and_test(model, data_set, train_proportion, 20, best_hyper_params[1],best_hyper_params[2],best_hyper_params[3],epochs_per_set[i], component, f"CNN_test_{component}.txt")
    bar.finish()
    elapsed = time.time()-start_time
    print(f'CNN optimisation completed, results stored in {output_file}'
        +f'\nModels Considered: {total_models}'
        +f'\nCompletion Time: {int((elapsed/60)//60)}:{int(elapsed//60)%60}:{round(elapsed%60)}s\n\n')



    # ------------ TCN ------------ #

    hyperparemeters = [[32, 64,128], # number of nodes per hidden layer
                    [0.001], # learning rate
                    [64], # batch size
                    [4,6,8,10], # sequence length
                    [0.0,0.1,0.2], # dropout
                    [2,4,6]] # kernel size]

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
                trainset, validationset, testset = dm.dataload(dm.sliding_window_CNN, batch_size, data_set, seq_length,train_proportion,component)
                for lr in hyperparemeters[1]:
                    for nodes in hyperparemeters[0]:      
                        for dropout in hyperparemeters[4]:
                            for kernel_size in hyperparemeters[5]:
                                # create the model 
                                index+=1
                                val_loss = 0
                                for av in range(models_to_average):
                                    model = models.TCN(seq_length,max(1, component), [nodes]*3, kernel_size, dropout).to(dev)
                                    val_loss += dm.train_model(model, trainset, validationset, testset, lr, epochs_per_set[i])/models_to_average
                                bar.update(index)
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
                +f'Kernel Size: {best_hyper_params[5]}\n'
                +f'Dataset: {file_names[i]}')
        f.close()
        model = models.TCN(best_hyper_params[3], max(1, component), [best_hyper_params[0]]*3, best_hyper_params[5], best_hyper_params[4]).to(dev)
        dm.train_and_test(model, data_set, train_proportion, 20, best_hyper_params[1],best_hyper_params[2],best_hyper_params[3],epochs_per_set[i], component, f"TCN_test_{component}.txt")

    bar.finish()
    elapsed = time.time()-start_time
    print(f'TCN optimisation completed, results stored in {output_file}'
        +f'\nModels Considered: {total_models}'
        +f'\nCompletion Time: {int((elapsed/60)//60)}:{int(elapsed//60)%60}:{round(elapsed%60)}s\n\n')

    # ------------ RNN ------------ #
    hyperparemeters = [[64,65],  # number of nodes per hidden layer
                    [0.001,0.0001], # learning rate
                    [64], # batch size
                    [4,6,8,10], # sequence length
                    [0.0] # dropout
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
                        trainset, validationset, testset = dm.dataload(dm.sliding_window_RNN, batch_size, data_set, seq_length, train_proportion, component)
                        for dropout in hyperparemeters[4]:

                            # create the model 
                            model = models.RNN(max(1,component), 2, nodes, 1, dropout)
                            index+=1
                            bar.update(index)
                            val_loss = 0
                            for jj in range(models_to_average):
                                val_loss += dm.train_model(model, trainset, validationset, testset, lr, epochs_per_set[i])

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
        +f'\nCompletion Time: {int((elapsed/60)//60)}:{int(elapsed//60)%60}:{round(elapsed%60)}s\n\n')


    # ------------ LSTM ------------ #

    hyperparemeters = [[64,65],  # number of nodes per hidden layer
                    [0.001,0.0001], # learning rate
                    [64], # batch size
                    [4,6,8,10], # sequence length
                    [0.0] # dropout
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
                        trainset, validationset, testset = dm.dataload(dm.sliding_window_RNN, batch_size, data_set, seq_length,train_proportion, component)
                        for dropout in hyperparemeters[4]:

                            # create the model 
                            model = models.LSTM(max(1,component), 2, nodes, 1, dropout)
                            index+=1
                            bar.update(index)
                            val_loss = 0
                            for jj in range(models_to_average):
                                val_loss += dm.train_model(model, trainset, validationset, testset, lr, epochs_per_set[i])

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
        +f'\nCompletion Time: {int((elapsed/60)//60)}:{int(elapsed//60)%60}:{round(elapsed%60)}s\n\n')


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
                        trainset, validationset, testset = dm.dataload(dm.sliding_window_RNN, batch_size, data_set, seq_length,train_proportion, component)
                        for dropout in hyperparemeters[4]:

                            # create the model 
                            model = models.BiLSTM(max(1,component), 2, nodes, 1, dropout)
                            index+=1
                            bar.update(index)
                            val_loss = 0
                            for jj in range(models_to_average):
                                val_loss += dm.train_model(model, trainset, validationset, testset, lr, epochs_per_set[i])

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
        +f'\nCompletion Time: {int((elapsed/60)//60)}:{int(elapsed//60)%60}:{round(elapsed%60)}s\n\n')
