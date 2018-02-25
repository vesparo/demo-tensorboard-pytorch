import copy
import os
import torch 
from torch import nn, optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from time import gmtime, strftime
from quora_utils import Quora
from model import SiameseNetwork
from test import test


# Hyper Parameters
max_sent_len = 60
input_size = 50
hidden_size = 100
num_layers = 2
num_classes = 2
batch_size = 32
num_epochs = 50
learning_rate = 0.003
print_freq = 500
model_time = strftime('%H:%M:%S', gmtime())
#num-perspective = 20?
#dropout=0.1

#  Dataset
print('loading Quora data...')
data = Quora(batch_size,input_size)
word_vocab_size=len(data.TEXT.vocab)
siamese = SiameseNetwork(input_size,word_vocab_size, hidden_size, num_layers,data)

parameters = filter(lambda p: p.requires_grad, siamese.parameters())

# Loss and Optimizer
optimizer = torch.optim.Adam(parameters, lr=learning_rate)
criterion = nn.CrossEntropyLoss()


writer = SummaryWriter(log_dir='runs/' +model_time )

siamese.train()
loss, last_epoch = 0, -1
max_dev_acc, max_test_acc = 0, 0

best_model = copy.deepcopy(siamese)
    
# Train the Model 
print('training start!')
iterator = data.train_iter
for i, batch in enumerate(iterator):
    present_epoch = int(iterator.epoch)
    if present_epoch == num_epochs:
        break
    if present_epoch > last_epoch:
        print('epoch:', present_epoch + 1)
    last_epoch = present_epoch
    s1, s2 = 'q1', 'q2'
    s1, s2 = getattr(batch, s1), getattr(batch, s2)
    # limit the lengths of input sentences up to max_sent_len
    if max_sent_len >= 0:
        if s1.size()[1] > max_sent_len:
            s1 = s1[:, :max_sent_len]
        if s2.size()[1] > max_sent_len:
            s2 = s2[:, :max_sent_len]
    # Forward + Backward + Optimize
    pred = siamese(s1,s2)
    optimizer.zero_grad()
    batch_loss = criterion(pred, batch.label)
    loss += batch_loss.data[0]
    batch_loss.backward()
    optimizer.step()
    
    if (i + 1) % print_freq == 0:
        dev_loss, dev_acc = test(siamese, data, mode='dev')
        test_loss, test_acc = test(siamese, data)
        c = (i + 1) // args.print_freq

        writer.add_scalar('loss/train', loss, c)
        writer.add_scalar('loss/dev', dev_loss, c)
        writer.add_scalar('acc/dev', dev_acc, c)
        writer.add_scalar('loss/test', test_loss, c)
        writer.add_scalar('acc/test', test_acc, c)

        print(f'train loss: {loss:.3f} / dev loss: {dev_loss:.3f} / test loss: {test_loss:.3f}'
              f' / dev acc: {dev_acc:.3f} / test acc: {test_acc:.3f}')

        if dev_acc > max_dev_acc:
            max_dev_acc = dev_acc
            max_test_acc = test_acc
            best_model = copy.deepcopy(siamese)

        loss = 0
        siamese.train()

writer.close()
print(f'max dev acc: {max_dev_acc:.3f} / max test acc: {max_test_acc:.3f}')

if not os.path.exists('saved_models'):
    os.makedirs('saved_models')
torch.save(best_model.state_dict(), f'saved_models/siamese_quora_{model_time}.pt')

print('training finished!')
