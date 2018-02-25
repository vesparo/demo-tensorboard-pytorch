import torch 
from torch import nn
from torch.autograd import Variable
#from tensorboardX import SummaryWriter
from quora_utils import Quora
from model import SiameseNetwork

def test(model, data, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0
    for batch in iterator:
        s1, s2 = 'q1', 'q2'
        s1, s2 = getattr(batch, s1), getattr(batch, s2)
        pred = model(s1,s2)

        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.data[0]

        _, pred = pred.max(dim=1)
        acc += (pred == batch.label).sum().float()
        size += len(pred)

    acc /= size
    acc = acc.cpu().data[0]
    return loss, acc