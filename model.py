import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self, input_size, word_vocab_size, hidden_size, num_layers,data):
        super(SiameseNetwork, self).__init__()

        self.word_emb = nn.Embedding(word_vocab_size, input_size)
        # initialize word embedding with GloVe
        self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
        # no fine-tuning for word vectors
        self.word_emb.weight.requires_grad = False

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2*4, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2)
        )

        
        

    def forward_once(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)) # 2 for bidirection 
        c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        
        # Forward propagate RNN
        output, _ = self.lstm(x, (h0, c0))
        return output

    def forward(self, input1, input2):
        p = self.word_emb(input1)
        h = self.word_emb(input2)
        output1 = self.forward_once(p)
        output2 = self.forward_once(h)
        # Decode hidden state of last time step
        out1 = output1[:, -1, :]
        out2 = output2[:, -1, :]
        #concatenazione out1, out2
        out = torch.cat((out1, out2, torch.pow((out1 - out2),2), out1 * out2),1)
        out = self.fc(out)
        return out

        