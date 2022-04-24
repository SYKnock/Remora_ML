import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm


# SKT Brain's BERTClassifier
class KoBERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=5,
                 dr_rate=None,
                 params=None):
        super(KoBERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.num_classes = num_classes

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids,
                              token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)


# evaluation function
def evaluation(model, dataloader, device):
    model = model.eval()
    predictions = []
    target_labels = []

    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            labels = label.long().to(device)

            output = model(input_ids, valid_length, segment_ids)
            _, preds = torch.max(output, dim=1)

            preds = preds.cpu().data
            labels = labels.cpu().data

            predictions.append(preds)
            target_labels.append(labels)

    predictions = np.concatenate(predictions, axis=0)
    target_labels = np.concatenate(target_labels, axis=0)

    accuracy = accuracy_score(target_labels, predictions)

    return accuracy


# train function
def train(model, epochs, train_dataloader,
          valid_dataloader, loss_function, 
          optimizer, device, scheduler, max_grad_norm):

    model = model.train()
    accumulation_step = 4

    for epoch in range(epochs):
        predictions = []
        target_labels = []
        accum_cnt = 0

        for data, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()
            labels = label.long().to(device)
            input_ids = token_ids.long().to(device)
            valid_length = valid_length
            segment_ids = segment_ids.long().to(device)

            output = model(token_ids=input_ids, 
                           valid_length=valid_length, 
                           segment_ids=segment_ids)
            loss = loss_function(output, labels)

            loss = loss / accumulation_step
            _, preds = torch.max(output, dim=1)
            predictions.append(preds.cpu().data)
            target_labels.append(labels.cpu().data)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            if (accum_cnt + 1) % accumulation_step == 0:
                optimizer.step()
                scheduler.step()
            accum_cnt += 1

        predictions = np.concatenate(predictions, axis=0)
        target_labels = np.concatenate(target_labels, axis=0)
        accuracy = accuracy_score(target_labels, predictions)
        val_accuracy = evaluation(model, valid_dataloader, device)

        print("Epoch: {0}".format(epoch + 1))
        print("Train Accuracy: {0}".format(accuracy))
        print("Validation Accuracy: {0}".format(val_accuracy))

    return model
