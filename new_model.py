from tqdm import tqdm 
import torch
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertModel
from sklearn.model_selection import train_test_split

CHECKPOINT_NAME = 'kykim/bert-kor-base'

df = pd.read_csv('/content/drive/MyDrive/Resume/wandb/concat.csv')
df = df.drop('Unnamed: 0', axis=1)

train, test = train_test_split(df, test_size=0.2, random_state=42)


class TokenDataset(Dataset):

    def __init__(self, dataframe, tokenizer_pretrained):
        self.data = dataframe
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrained)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['answer']
        label = self.data.iloc[idx]['text_label']

        # 토큰화 처리
        tokens = self.tokenizer(
            sentence,                # 1개 문장
            return_tensors='pt',     # 텐서로 반환
            truncation=True,         # 잘라내기 적용
            padding='max_length',    # 패딩 적용
            add_special_tokens=True  # 스페셜 토큰 적용
        )

        input_ids = tokens['input_ids'].squeeze(0)           # 2D -> 1D
        attention_mask = tokens['attention_mask'].squeeze(0) # 2D -> 1D
        token_type_ids = torch.zeros_like(attention_mask)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }, torch.tensor(label)

tokenizer_pretrained = CHECKPOINT_NAME

train_data = TokenDataset(train, tokenizer_pretrained)
test_data = TokenDataset(test, tokenizer_pretrained)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=8)
test_loader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=8)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'사용 디바이스: {device}')

inputs, labels = next(iter(train_loader))

# 데이터셋을 device 설정
inputs = {k: v.to(device) for k, v in inputs.items()}
labels.to(device)
config = BertConfig.from_pretrained(CHECKPOINT_NAME)

model_bert = BertModel.from_pretrained(CHECKPOINT_NAME).to(device)
model_bert


class CustomBertModel(nn.Module):
    def __init__(self, bert_pretrained, dropout_rate=0.5):
        super(CustomBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained)
        self.dr = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        last_hidden_state = output['last_hidden_state']
        x = self.dr(last_hidden_state[:, 0, :])
        x = self.fc(x)
        return x


bert = CustomBertModel(CHECKPOINT_NAME)
bert.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(bert.parameters(), lr=1e-5)




def model_train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0
    corr = 0
    counts = 0

    prograss_bar = tqdm(data_loader, unit='batch', total=len(data_loader), mininterval=1)

    for idx, (inputs, labels) in enumerate(prograss_bar):
        inputs = {k:v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(**inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        _, pred = output.max(dim=1)
        corr += pred.eq(labels).sum().item()
        counts += len(labels)
        running_loss += loss.item() * labels.size(0)
        prograss_bar.set_description(f"training loss: {running_loss/(idx+1):.5f}, training accuracy: {corr / counts:.5f}")
    acc = corr / len(data_loader.dataset)
    return running_loss / len(data_loader.dataset), acc



def model_evaluate(model, data_loader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        corr = 0
        running_loss = 0
        for inputs, labels in data_loader:
            inputs = {k:v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            output = model(**inputs)
            _, pred = output.max(dim=1)
            corr += torch.sum(pred.eq(labels)).item()
            running_loss += loss_fn(output, labels).item() * labels.size(0)
        acc = corr / len(data_loader.dataset)
        return running_loss / len(data_loader.dataset), acc
    
num_epochs = 10
model_name = 'bert-kor-base'
min_loss = np.inf

for epoch in range(num_epochs):

    train_loss, train_acc = model_train(bert, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc = model_evaluate(bert, test_loader, loss_fn, device)
    if val_loss < min_loss:
        print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
        min_loss = val_loss
        torch.save(bert.state_dict(), f'{model_name}.pth')

    print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')