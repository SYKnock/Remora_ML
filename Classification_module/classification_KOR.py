import torch
from train.model_remora_KOR import KoBERTClassifier
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import gluonnlp as nlp
import os
from train.data_remora_KOR import BERTDataset
from torch.utils.data import DataLoader

root = os.getcwd()

category = {0: '정치',
            1: '세계',
            2: '사회',
            3: 'IT과학',
            4: '스포츠',
            5: '경제',
            6: '생활문화'
            }


def script_to_tsv(PATH):
    tsv_path = "./tmp_data_file.tsv"
    file = open(PATH, mode='r')
    script = file.read()
    file.close()
    script = script.replace("\n", " ")

    tsv_script = "Category\tArticle\n-1\t"
    tsv_script += script
    file = open(tsv_path, mode='w')
    file.write(tsv_script)
    file.close()


def inference(bert, tokenizer, vocab, device, model, batch_size, max_len):
    tsv_path = "./tmp_data_file.tsv"

    dataset = nlp.data.TSVDataset(tsv_path, field_indices=[1, 0], num_discard_samples=1)
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    evaluation_data = BERTDataset(dataset, 0, 1, tok, max_len, True, False)
    dataloader = DataLoader(evaluation_data, batch_size=batch_size, num_workers=1)

    model.eval()
    predictions = []

    with torch.no_grad():
        for i, (token_ids, valid_length, segment_ids, label) in enumerate(dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length

            output = model(token_ids, valid_length, segment_ids)
            _, preds = torch.max(output, dim=1)
            preds = preds.cpu().data
            predictions.append(preds)

    return predictions[0].item()


def parameter_setting():
    batch_size = 64
    max_len = 128
    category_num = len(category)

    return batch_size, max_len, category_num


def classification(PATH):
    weight_path = "./weights/classification_KOR.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()

    batch_size, max_len, category_num = parameter_setting()

    model = KoBERTClassifier(bert, dr_rate=0.5, num_classes=category_num)
    model.load_state_dict(torch.load(weight_path))
    model = model.cuda()

    script_to_tsv(PATH)
    result = inference(bert, tokenizer, vocab, device, model, batch_size, max_len)

    PATH = root + "/classification_result.txt"
    file = open(PATH, mode='w')
    file.write(category[result])
    file.close()