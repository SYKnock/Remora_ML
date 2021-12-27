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
import torch
from torch import nn
import model_remora
import data_remora
from transformers import AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup
from adamp import AdamP
import argparse
# -


# +
def train_parameter_set(model, train_len, EPOCHS, device):
    parameter = model.parameters()
    optimizer = AdamP(parameter, lr=2e-5, betas=(0.9, 0.999), weight_decay=1e-2)
    total_steps = train_len * EPOCHS
    warmup_step = train_len / 2

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(warmup_step),
                                                num_training_steps=total_steps)

    loss_function = nn.CrossEntropyLoss().to(device)

    return optimizer, scheduler, loss_function
# -


# +
def test_and_save(model, test_dataloader, device, PATH):
    test_accuracy = model_remora.evaluation(model, test_dataloader, device)
    print("\nTest Accuracy: {0}".format(test_accuracy))

    torch.save(model.state_dict(), PATH)
# -


# +    
def main():
    parser = argparse.ArgumentParser(description='PATH, EPOCHS, BATCH_SIZE, MAX_LEN, SAVE_PATH')
    parser.add_argument('--EPOCHS', type=int, default=10, help='')
    parser.add_argument('--BATCH_SIZE', type=int, default=64, help='Batch size')
    parser.add_argument('--DATA_PATH', required=True, help='Path to training data')
    parser.add_argument('--MAX_LEN', type=int, default=128, help='Maximum data length')
    parser.add_argument('--SAVE_PATH', required=True, help='Path to save the trained model')

    args = parser.parse_args()

    # set BERT and tokenizer and device and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert = AutoModel.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_remora.BertNewsCategoryClassifier(len(data_remora.category), bert, 0.4)
    model = model.to(device)

    train_dataloader, val_dataloader, test_dataloader = data_remora.dataloader_generate(args.DATA_PATH, 
                                                                                        tokenizer, 
                                                                                        args.MAX_LEN,
                                                                                        args.BATCH_SIZE)

    optimizer, scheduler, loss_function = train_parameter_set(model, 
                                                              len(train_dataloader), 
                                                              args.EPOCHS, 
                                                              device)

    model = model_remora.train(model=model, 
                               epochs=args.EPOCHS, 
                               train_dataloader=train_dataloader, 
                               val_dataloader=val_dataloader,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               loss_function=loss_function, 
                               device=device)

    test_and_save(model, test_dataloader, device, args.SAVE_PATH)


if __name__ == "__main__":
    main()
# -
