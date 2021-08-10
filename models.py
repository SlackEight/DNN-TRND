import torch
import torch.nn as nn
from torch.autograd import Variable

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