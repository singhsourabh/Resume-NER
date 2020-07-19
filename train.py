import numpy as np
import torch
from transformers import BertForTokenClassification, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from utils import trim_entity_spans, convert_goldparse, ResumeDataset, tag2idx, idx2tag, get_hyperparameters, train_and_val_model


MAX_LEN = 500
EPOCHS = 1
MAX_GRAD_NORM = 1.0
MODEL_PATH = 'bert-base-uncased/'
TOKENIZER = BertTokenizerFast('bert-base-uncased/vocab.txt', lowercase=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = trim_entity_spans(convert_goldparse('data/Resumes.json'))

total = len(data)
train_data, val_data = data[:180], data[180:]

train_d = ResumeDataset(train_data, TOKENIZER, tag2idx, MAX_LEN)
val_d = ResumeDataset(val_data, TOKENIZER, tag2idx, MAX_LEN)

train_sampler = RandomSampler(train_d)
train_dl = DataLoader(train_d, sampler=train_sampler, batch_size=8)

val_dl = DataLoader(val_d, batch_size=4)

model = BertForTokenClassification.from_pretrained(
    MODEL_PATH, num_labels=len(tag2idx))
model.to(DEVICE)
optimizer_grouped_parameters = get_hyperparameters(model, True)
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

train_and_val_model(
    model,
    TOKENIZER,
    optimizer,
    EPOCHS,
    idx2tag,
    tag2idx,
    MAX_GRAD_NORM,
    DEVICE,
    train_dl,
    val_dl
)

torch.save(
    {
        "model_state_dict": model.state_dict()
    },
    'model_e6.tar',
)
