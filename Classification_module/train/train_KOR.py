import torch
from torch import nn
import model_remora_KOR as model_remora
import data_remora_KOR as data_remora
from transformers import get_cosine_schedule_with_warmup
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from adamp import AdamP
import argparse


def test_model(model, dataloader, device):
    test_accuracy = model_remora.evaluation(model, dataloader, device)
    print("\nTest Accuracy: {0}".format(test_accuracy))


def save_model(model, path):
    torch.save(model.state_dict(), path)


def train_process(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()

    train_dataloader, valid_dataloader, test_dataloader = data_remora.dataloader_generator(args.DATA_PATH,
                                                                                           tokenizer,
                                                                                           vocab,
                                                                                           args.MAX_LEN,
                                                                                           args.BATCH_SIZE)
    learning_rate = 2e-5
    max_grad_norm = 1.0
    warmup_ratio = 0.5

    model = model_remora.KoBERTClassifier(bert, dr_rate=0.5, num_classes=args.CATEGORY_NUM)
    model = model.to(device)

    model = model.train()
    parameter = model.parameters()
    optimizer = AdamP(parameter, learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)
    total_steps = len(train_dataloader) * args.EPOCHS
    warmup_step = int(len(train_dataloader) * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(warmup_step),
                                                num_training_steps=total_steps)
    loss_function = nn.CrossEntropyLoss().to(device)

    model = model_remora.train(model=model,
                               epochs=args.EPOCHS,
                               train_dataloader=train_dataloader,
                               valid_dataloader=valid_dataloader,
                               device=device,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               loss_function=loss_function,
                               max_grad_norm=max_grad_norm
                               )

    test_model(model, test_dataloader, device)
    save_model(model, args.SAVE_PATH)


def main():
    parser = argparse.ArgumentParser(description='PATH, EPOCHS, BATCH_SIZE, MAX_LEN, SAVE_PATH')
    parser.add_argument('--EPOCHS', type=int, default=10, help='')
    parser.add_argument('--BATCH_SIZE', type=int, default=64, help='Batch size')
    parser.add_argument('--DATA_PATH', required=True, help='Path to training data')
    parser.add_argument('--MAX_LEN', type=int, default=128, help='Maximum data length')
    parser.add_argument('--SAVE_PATH', required=True, help='Path to save the trained model')
    parser.add_argument('--CATEGORY_NUM', type=int, default=5, help='Number of categories')
    args = parser.parse_args()

    train_process(args)


if __name__ == "__main__":
    main()