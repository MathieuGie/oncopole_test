import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import pytorch_lightning as pl
from lightning.pytorch.loggers import MLFlowLogger
import random
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import pandas as pd
from encdec import Encoder
import numpy as np

batches = 128
epochs = 800

    
class Contrastive(pl.LightningModule):
    def __init__(self,):
        super().__init__()

        self.epochh=0

        self.encoder_dnam = Encoder("dnam")
        self.encoder_dnam.load_state_dict(torch.load("encoder_dnam.chkpt"))
        self.encoder_dnam.eval()

        self.encoder_rna = Encoder("rna")
        self.encoder_rna.load_state_dict(torch.load("encoder_rna.chkpt"))
        self.encoder_rna.eval()


        self.relu = nn.ReLU()

        self.linear1 = nn.Linear(3000+2000, 8000)
        self.linear2 = nn.Linear(8000, 2000)
        self.linear3 = nn.Linear(2000, 800)
        self.linear4 = nn.Linear(800, 100)
        self.linear5 = nn.Linear(100, 20)
        self.linear6 = nn.Linear(20, 1)

    def forward(self, x_dnam, x_rna):

        x_dnam = self.encoder_dnam(x_dnam)
        x_rna = self.encoder_rna(x_rna)

        #print(x_rna.shape)

        x = torch.cat((x_dnam, x_rna), axis=1)
        
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.relu(self.linear5(x))
        x = self.linear6(x)

        return x
    
    def _step(self, dnam, rna, comparison):

        outputs = self(dnam, rna)
        loss = nn.functional.mse_loss(comparison, outputs)
        return loss

    def training_step(self,batch, batch_idx):

        dnam, rna, rand_rna = batch
        positive = torch.ones((dnam.shape[0],1)).to(mps_device)
        negative = torch.zeros((dnam.shape[0],1)).to(mps_device)

        loss_pos = self._step(dnam, rna, positive)
        loss_neg = self._step(dnam, rand_rna, negative)
        loss = torch.mean(torch.stack([loss_pos, loss_neg]))

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):

        dnam, rna, rand_rna = batch
        positive = torch.ones((dnam.shape[0],1)).to(mps_device)
        negative = torch.zeros((dnam.shape[0],1)).to(mps_device)

        loss_pos = self._step(dnam, rna, positive)
        loss_neg = self._step(dnam, rand_rna, negative)
        loss = torch.mean(torch.stack([loss_pos, loss_neg]))

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):

        # Optimizer and LR scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def on_train_epoch_start(self):
        
        #torch.save(self.state_dict(), "contrastive_.chkpt")

        self.epochh+=1
        self.train()
    

class ContrastiveDataset(Dataset):
    def __init__(self, sett):

        path_dnam = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/'+sett+'_dnam.csv'
        path_rna = '/Users/mathieugierski/Nextcloud/Macbook M3/Oncopole/'+sett+'_rna.csv'

        self.dnam_df = pd.read_csv(path_dnam)
        self.rna_df = pd.read_csv(path_rna)

    def __len__(self):
        return self.dnam_df.shape[0]

    def __getitem__(self, idx):
        dnam_seq = self.dnam_df.iloc[idx, 1:].values.astype("float32")
        rna_seq = self.rna_df.iloc[idx, 1:].values.astype("float32")

        rand = np.random.randint(0, self.dnam_df.shape[0])
        while rand==idx:
            rand = np.random.randint(0, self.dnam_df.shape[0])

        random_rna_seq = self.rna_df.iloc[rand, 1:].values.astype("float32")


        #print("dnam", dnam_seq)
        #print("rna", rna_seq)
        #print("rand_rna", random_rna_seq)

        dnam = torch.from_numpy(dnam_seq).to(mps_device)
        rna = torch.from_numpy(rna_seq).to(mps_device)
        rand_rna = torch.from_numpy(random_rna_seq).to(mps_device)

        return dnam, rna, rand_rna
    
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    # Add more transformations as needed
])

dir = '/Users/mathieugierski/Nextcloud/Macbook M3/Diffusion/CAT_00_treated'

image_files = os.listdir(dir)
random.shuffle(image_files)

train_files, test_files = train_test_split(image_files, test_size=0.3)  # Adjust test_size as needed
#print(len(train_files))
train_dataset = ContrastiveDataset("train")
test_dataset = ContrastiveDataset("test")

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
    mps_device = torch.device("mps")

model = Contrastive()
model.to(mps_device)

print("model init done")

mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")

trainer = pl.Trainer(max_epochs=epochs, accelerator="mps", logger=mlf_logger, log_every_n_steps=1)
trainer.fit(model, dataloader_train, dataloader_test)