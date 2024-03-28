from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

class GoodreadsDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, target=None):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target = target
        if target is not None:
            self.target = list(target)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])

        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        if self.target is not None:
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "target": torch.tensor(self.target[item], dtype=torch.float32)  # For regression or binary classification tasks, use torch.float32. For multi-class classification, use torch.long.
            }
        else:
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)
            }

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def get_dataloader(model_name, fold_count, text_col, max_len, label_col, train_batch_size, valid_batch_size):
    df = pd.read_csv('data_file_path')  # Use the provided data file path instead of hardcoding it.

    le = LabelEncoder()
    df["label_encode"] = le.fit_transform(df[label_col])

    kf = StratifiedKFold(n_splits=fold_count, random_state=42, shuffle=True)
    for f, (t_, v_) in enumerate(kf.split(X=df[text_col], y=df.label_encode)):
        df.loc[v_, 'fold'] = f

    train_df = df[df.fold != 0]
    valid_df = df[df.fold == 0]

    tokenizer = get_tokenizer(model_name)

    train_ds = GoodreadsDataset(train_df[text_col],
                                tokenizer,
                                max_len,
                                train_df[label_col])
    valid_ds = GoodreadsDataset(valid_df[text_col],
                                tokenizer,
                                max_len,
                                valid_df[label_col])

    train_loader = DataLoader(
        dataset=train_ds, batch_size=int(train_batch_size), shuffle=True)
    valid_loader = DataLoader(
        dataset=valid_ds, batch_size=int(valid_batch_size), shuffle=False)

    return train_loader, valid_loader
