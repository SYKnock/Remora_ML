# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import os
import transformers
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
root = os.getcwd()

category = {0: 'ENTERTAINMENT',
            1: 'WORLD NEWS',
            2: 'POLITICS',
            3: 'COMEDY',
            4: 'SPORTS',
            5: 'BUSINESS',
            6: 'TRAVEL',
            7: 'SCIENCE & TECH',
            8: 'PARENTS',
            9: 'ARTS & CULTURE',
            10: 'STYLE & BEAUTY',
            11: 'FOOD & DRINK',
            12: 'HEALTHY LIVING',
            13: 'WELLNESS',
            14: 'HOME & LIVING'}


# -

def script_to_list():
    PATH = root + "/../OCR_module/predict_text.txt"
    file = open(PATH, mode='r')
    script = file.readlines()
    
    cnt = 0
    total_len = len(script)
    
    list_script = []
    tmp_str = ''
    
    for line in script:
        if cnt != 0 and cnt != (total_len - 1):
            line = line.replace("\n", ". ")
            tmp_str += line
        cnt += 1
    
    tmp_str = tmp_str[:-1]
    list_script.append(tmp_str)
    file.close()
    
    return list_script


# classfier model
class TextClassifier(nn.Module):
    def __init__(self, num_labels, BERT, _dropout = 0.3):
        super(TextClassifier, self).__init__()
        self.num_labels = num_labels
        self.bert = BERT
        self.drop = nn.Dropout(_dropout)
        self.classifier = nn.Linear(BERT.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels = None):
        pooled_output = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        output = self.drop(pooled_output[0][:, 0, :])
        return self.classifier(output)


def inference(texts, bert, tokenizer):
    PATH = "./weights/classification.pth"
    model = TextClassifier(16, bert, 0.4)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    orig_token = tokenizer(texts, add_special_tokens=True,
                           max_length=256, return_token_type_ids=False,
                           padding='max_length', return_attention_mask=True,
                           truncation=True, return_tensors='pt')

    output = model(input_ids=orig_token.input_ids, attention_mask=orig_token.attention_mask)
    _, preds = torch.max(output, dim=1)

    preds_list = preds.tolist()
    result = preds_list[0]

    return result


# +
def main():
    bert = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    texts = script_to_list()
    
    result = category[inference(texts, bert, tokenizer)]
    
    PATH = root + "/category.txt"
    file = open(PATH, mode='w')
    file.write(result)
    file.close()
    
if __name__ == "__main__":
    main()
# -

