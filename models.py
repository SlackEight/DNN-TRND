import torch
import torch.nn as nn
from torch.autograd import Variable

#Chesney's models

class RNN(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        h_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))
        output, hidden = self.rnn(input, h_0)
        out = self.fc(hidden)

        return out


class BiLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(BiLSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            batch_first=True,
                            bidirectional=True
                            )
                            
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers * 2, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers * 2, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        _, (h_out, _) = self.bilstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size * 2)
        out = self.fc(h_out)
        
        return out

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            batch_first=True
                            )
                            
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        
        return out


# Morgan's models

class MLP(nn.Module):
    def __init__(self, seq_len, hidden_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(seq_len, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.do = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(self.fc2(x))
        x = self.do(x)
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

class CNN(nn.Module):
    """Convolutional Neural Networks"""
    def __init__(self, input_size, hidden_dim, dropout, kernel_size):
        super(CNN, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_dim, kernel_size=kernel_size),
            nn.ReLU(inplace=True),

            nn.Flatten(),
            nn.Dropout(dropout, inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.init_weights()
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        out = self.main(x)
        return out

from tcn import TemporalConvNet

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        y1 = torch.relu(self.tcn(x))
        return self.linear(y1[:, :, -1])