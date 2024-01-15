import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
from lightning.pytorch.loggers import MLFlowLogger
import random
from sklearn.model_selection import train_test_split
import copy
import pandas as pd

batches = 128
epochs = 800


class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.linear1 = nn.Linear(24443, 15000)
        self.linear2 = nn.Linear(15000, 8000)
        self.linear3 = nn.Linear(8000, 3000)

  
    def forward(self, x):
        
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)

        return (x-torch.min(x))/(torch.max(x)-torch.min(x))
    
class Decoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.linear1 = nn.Linear(3000, 8000)
        self.linear2 = nn.Linear(8000, 15000)
        self.linear3 = nn.Linear(15000, 24443)

    def forward(self, x):

        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)

        return self.sigmoid(x)

    
class Encdec(pl.LightningModule):
    def __init__(self,):
        super().__init__()

        self.epochh=0
        self.val_loss=99999
        self.best_val_loss=999999

        self.encoder = Encoder().to(mps_device)
        self.decoder = Decoder().to(mps_device)

    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def _step(self, batch, batch_idx):

        outputs = self(batch)
        #print("for loss:", outputs.shape, batch.shape)
        loss = nn.functional.mse_loss(outputs, batch)
        return loss
    

    def training_step(self,batch, batch_idx):

        # Called to compute and log the training loss
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):

        # Called to compute and log the validation loss
        val_loss = self._step(batch, batch_idx)

        self.val_loss = val_loss
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):

        # Optimizer and LR scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def on_train_epoch_start(self):

        if self.epochh>30 and self.val_loss<=self.best_val_loss:
            torch.save(self.encoder.state_dict(), "encoder.chkpt")
            torch.save(self.decoder.state_dict(), "decoder.chkpt")
            self.best_val_loss = self.val_loss

        self.epochh+=1
        self.train()
    

class SequenceDataset(Dataset):
    def __init__(self, dir, sample_frac=1):

        data = pd.read_csv(dir)
        self.data = data.sample(frac=sample_frac)

        print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        features = self.data.iloc[idx, 1:].values.astype("float32")
        features_tensor = torch.from_numpy(features).to(mps_device)
        return features_tensor
    
path_train  = os.path.join('/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole', 'train_dnam.csv')
path_test  = os.path.join('/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole', 'test_dnam.csv')

#print(len(train_files))
train_dataset = SequenceDataset(path_train)
print("training opened")
test_dataset = SequenceDataset(path_test)
print("testing opened")

print("training set", len(train_dataset))
print("test set", len(test_dataset))

dataloader_train = DataLoader(train_dataset, batch_size=batches, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=batches, shuffle=True)

print("before mps device")
#Devise:
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    print("found device: mps")
    mps_device = torch.device("mps")

model = Encdec()
model.to(mps_device)

print("model init done")

mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")

trainer = pl.Trainer(max_epochs=epochs, accelerator='cpu', logger=mlf_logger, log_every_n_steps=1)
trainer.fit(model, dataloader_train, dataloader_test)