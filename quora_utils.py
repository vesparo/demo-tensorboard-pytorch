from torchtext import data
from torchtext.vocab import GloVe

class Quora():
    def __init__(self, batch_size, word_dim):
        self.RAW = data.RawField()
        self.TEXT = data.Field(batch_first=True)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        self.train, self.dev, self.test = data.TabularDataset.splits(
            path='.data/quora',
            train='train.tsv',
            validation='dev.tsv',
            test='test.tsv',
            format='tsv',
            fields=[('label', self.LABEL),
                    ('q1', self.TEXT),
                    ('q2', self.TEXT),
                    ('id', self.RAW)])

        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='6B', dim=word_dim))
        self.LABEL.build_vocab(self.train)

        sort_key = lambda x: data.interleave_keys(len(x.q1), len(x.q2))

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       device=-1,
                                       batch_sizes=[batch_size] * 3,
                                       sort_key=sort_key)

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])
