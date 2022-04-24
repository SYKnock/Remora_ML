import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split


category = {'ENTERTAINMENT': 0,
            'WORLD NEWS': 1,
            'POLITICS': 2,
            'COMEDY': 3,
            'SPORTS': 4,
            'BUSINESS': 5,
            'TRAVEL': 6,
            'SCIENCE & TECH': 7,
            'PARENTS': 8,
            'ARTS & CULTURE': 9,
            'STYLE & BEAUTY': 10,
            'FOOD & DRINK': 11,
            'HEALTHY LIVING': 12,
            'WELLNESS': 13,
            'HOME & LIVING': 14}


# Data class
class Data(Dataset):
    def __init__(self, texts, targets, ids, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        id = torch.tensor(self.ids[index], dtype=torch.long)
        label = torch.tensor(self.targets[index], dtype=torch.int)
        text = self.texts[index]
        token = self.tokenizer(text, add_special_tokens=True,
                               max_length=self.max_len,
                               return_token_type_ids=False,
                               padding="max_length",
                               return_attention_mask=True,
                               truncation=True,
                               return_tensors="pt")

        attention_mask = token["attention_mask"]
        input_ids = token["input_ids"].flatten()

        return {
            "id": id,
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }

    def __len__(self):
        length = len(self.texts)
        return length


# data pre-processing
def category_merge(x):
    if x == 'THE WORLDPOST' or x == 'WORLD NEWS' or x == 'WORLDPOST':
        return 'WORLD NEWS'
    elif x == 'TASTE':
        return 'FOOD & DRINK'
    elif x == 'STYLE':
        return 'STYLE & BEAUTY'
    elif x == 'PARENTING':
        return 'PARENTS'
    elif x == 'COLLEGE':
        return 'EDUCATION'
    elif x == 'ARTS' or x == 'CULTURE & ARTS':
        return 'ARTS & CULTURE'
    elif x == 'SCIENCE' or x == 'TECH':
        return 'SCIENCE & TECH'
    elif x == 'BLACK VOICES' or x == 'DIVORCE' or x == 'LATINO VOICES' or x == 'QUEER VOICES' or x == 'GOOD NEWS':
        return 'DROP'
    elif x == 'WEDDINGS' or x == 'WOMEN' or x == 'IMPACT' or x == 'CRIME' or x == 'MEDIA' or x == 'WEIRD NEWS':
        return 'DROP'
    elif x == 'GREEN' or x == 'RELIGION' or x == 'EDUCATION' or x == 'MONEY' or x == 'FIFTY' or x == 'ENVIRONMENT':
        return 'DROP'
    else:
        return x


def drop_small_data(data):
    data['category'] = data['category'].apply(category_merge)
    data["text"] = data["headline"] + '. ' + data["short_description"]
    data.drop(["authors", "link", "date", "headline", "short_description"], axis=1, inplace=True)

    drop_data = []
    cnt = 0
    for i in data['category']:
        if i == 'EDUCATION' or i == 'DROP':
            drop_data.append(cnt)
        cnt = cnt + 1
    data.drop(index=drop_data, inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data


def drop_large_data(data):
    drop_large_data = []
    cnt = [0, 0, 0, 0, 0]
    index = 0

    for i in data['category']:
        if i == 'POLITICS':
            cnt[0] = cnt[0] + 1
            if cnt[0] > 10000:
                drop_large_data.append(index)
        elif i == 'WELLNESS':
            cnt[1] = cnt[1] + 1
            if cnt[1] > 10000:
                drop_large_data.append(index)
        elif i == 'PARENTS':
            cnt[2] = cnt[2] + 1
            if cnt[2] > 10000:
                drop_large_data.append(index)
        elif i == 'ENTERTAINMENT':
            cnt[3] = cnt[3] + 1
            if cnt[3] > 10000:
                drop_large_data.append(index)
        elif i == 'STYLE & BEAUTY':
            cnt[4] = cnt[4] + 1
            if cnt[4] > 10000:
                drop_large_data.append(index)
        index = index + 1

    data.drop(index=drop_large_data, inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data


def pre_processing(data):
    data = drop_small_data(data)
    data = drop_large_data(data)
    return data


def category_to_integer(category_str):
    return category.get(category_str)


def data_setting(PATH):
    raw_data = pd.read_json(PATH, lines=True)

    data = pre_processing(raw_data)
    data["category"] = data["category"].apply(category_to_integer)
    text_set = data.drop("category", axis=1)
    target = data["category"]

    return text_set, target


# data loader setting
def dataloader_generate(PATH, tokenizer, MAX_LEN, BATCH_SIZE):
    text_set, target = data_setting(PATH)

    # splitting data to train, val, test(7:2:1)
    train_text, val_text, train_target, val_target = train_test_split(text_set,
                                                                      target,
                                                                      test_size=0.2,
                                                                      shuffle=True)

    train_text, test_text, train_target, test_target = train_test_split(train_text,
                                                                        train_target,
                                                                        test_size=0.15,
                                                                        shuffle=True)

    train_ids = train_text.index.to_numpy()
    val_ids = val_text.index.to_numpy()
    test_ids = test_text.index.to_numpy()

    train_text = train_text.text.to_numpy()
    val_text = val_text.text.to_numpy()
    test_text = test_text.text.to_numpy()

    train_target = train_target.to_numpy()
    val_target = val_target.to_numpy()
    test_target = test_target.to_numpy()

    train_data = Data(texts=train_text, targets=train_target,
                      ids=train_ids, tokenizer=tokenizer, max_len=MAX_LEN)
    val_data = Data(texts=val_text, targets=val_target,
                    ids=val_ids, tokenizer=tokenizer, max_len=MAX_LEN)
    test_data = Data(texts=test_text, targets=test_target,
                     ids=test_ids, tokenizer=tokenizer, max_len=MAX_LEN)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,
                                  num_workers=1, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE,
                                num_workers=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                                 num_workers=1, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader 

