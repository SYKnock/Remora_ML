import torch
import gluonnlp as nlp
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# SKT Brain's BERT Dataset Class
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


def dataloader_generator(path, tokenizer, vocab, max_len, batch_size):
    dataset = nlp.data.TSVDataset(path, field_indices=[1, 0], num_discard_samples=1)

    # split train/valid/test dataset(7:2:1)
    train_dataset, valid_dataset = nlp.data.train_valid_split(dataset,
                                                              valid_ratio=0.3,
                                                              stratify=None)

    valid_dataset, test_dataset = nlp.data.train_valid_split(valid_dataset,
                                                             valid_ratio=0.33,
                                                             stratify=None)

    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    train_data = BERTDataset(train_dataset, 0, 1, tok, max_len, True, False)
    valid_data = BERTDataset(valid_dataset, 0, 1, tok, max_len, True, False)
    test_data = BERTDataset(test_dataset, 0, 1, tok, max_len, True, False)

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  num_workers=1, shuffle=True)
    val_dataloader = DataLoader(valid_data, batch_size=batch_size,
                                num_workers=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                 num_workers=1, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader
