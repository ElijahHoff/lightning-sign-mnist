import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import argparse
import os
import urllib.request
import zipfile


# Функция для скачивания файлов

def download_dataset():
    data_files = {
        "sign_mnist_train.csv.zip": "https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/sign_mnist_train.csv.zip",
        "sign_mnist_test.csv.zip": "https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/sign_mnist_test.csv.zip"
    }

    if not os.path.exists("data"):
        os.mkdir("data")

    for file_name, url in data_files.items():
        file_path = os.path.join("data", file_name)
        if not os.path.exists(file_path):
            print(f"Скачивание {file_name}...")
            urllib.request.urlretrieve(url, file_path)

            # Распаковка
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall("data")


# Создание CustomDataset
class SignLanguageDataset(Dataset):
    def __init__(self, df):
        self.labels = df['label'].values
        self.images = df.drop(columns=['label']).values.astype(np.float32) / 255.

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).reshape(1, 28, 28)
        label = torch.tensor(self.labels[idx]).long()
        return image, label


# LightningDataModule
class SignLanguageDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, batch_size=64):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = SignLanguageDataset(self.train_df)
        self.test_dataset = SignLanguageDataset(self.test_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1)


# LightningModule
class SignLanguageModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 25)
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast_dev_run', default=False, type=bool)
    args = parser.parse_args()

    download_dataset()

    train_df = pd.read_csv('data/sign_mnist_train.csv')
    test_df = pd.read_csv('data/sign_mnist_test.csv')

    data_module = SignLanguageDataModule(train_df, test_df)
    model = SignLanguageModel()

    trainer = pl.Trainer(max_epochs=10, fast_dev_run=args.fast_dev_run)

    try:
        trainer.fit(model, data_module)
        if args.fast_dev_run:
            print("Тестовый прогон успешно пройден")
    except Exception as e:
        if args.fast_dev_run:
            print("Тестовый прогон завершился с ошибкой")
            exit()

    trainer.save_checkpoint("sign_language_model.ckpt")

    # Инференс на одном образце
    sample_loader = data_module.test_dataloader()
    sample_image, sample_label = next(iter(sample_loader))
    prediction = model(sample_image).argmax(dim=1)

    print(f'Predicted label: {prediction.item()}, True label: {sample_label.item()}')
