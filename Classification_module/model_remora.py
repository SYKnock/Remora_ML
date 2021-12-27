import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm


# classfier model
class BertNewsCategoryClassifier(nn.Module):
    def __init__(self, num_labels, BERT, _dropout=0.3):
        super(BertNewsCategoryClassifier, self).__init__()
        self.num_labels = num_labels
        self.bert = BERT
        self.dropout = nn.Dropout(_dropout)
        self.classifier = nn.Linear(BERT.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.dropout(pooled_output[0][:, 0, :])
        return self.classifier(output)


# evaluation function
def evaluation(model, dataloader, device):
    model = model.eval()
    predictions = []
    target_labels = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            labels = data["label"].to(device)
            attention_mask = data["attention_mask"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
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
def train(model, epochs, train_dataloader, val_dataloader,
          loss_function, optimizer, device, scheduler):

    model = model.train()
    #accumulation_step = 4

    for epoch in range(epochs):
        predictions = []
        target_labels = []
        #cnt = 0

        for data in tqdm(train_dataloader):
            labels = data["label"].to(device)
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(output, dim=1)
            loss = loss_function(output, labels.long())
            #loss = loss / accumulation_step
            predictions.append(preds.cpu().data)
            target_labels.append(labels.cpu().data)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            #if (cnt + 1) % accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            #cnt += 1

        predictions = np.concatenate(predictions, axis=0)
        target_labels = np.concatenate(target_labels, axis=0)
        accuracy = accuracy_score(target_labels, predictions)
        val_accuracy = evaluation(model, val_dataloader, device)

        print("Epoch: {0}".format(epoch + 1))
        print("Train Accuracy: {0}".format(accuracy))
        print("Validation Accuracy: {0}".format(val_accuracy))

    return model
