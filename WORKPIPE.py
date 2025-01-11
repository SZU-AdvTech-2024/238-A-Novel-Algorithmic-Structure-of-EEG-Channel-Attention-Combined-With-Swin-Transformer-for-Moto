import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy
from strokesinit import StrokePatientsMIDataset
import numpy as np
from downsample import SetSamplingRate
from baseline import BaselineCorrection
from eca import ECA
from pytorch_lightning.callbacks import EarlyStopping
from model import SwinTransformer
from torcheeg.transforms import  Select,BandSignal,Compose,ToTensor
from torcheeg.trainers import ClassifierTrainer
from torcheeg.model_selection import KFoldPerSubject

HYPERPARAMETERS = {
    "seed": 42,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 0,
    "num_epochs": 50,
}

class ECASwinTransformerModel(nn.Module):
    def __init__(self, num_channels):
        super(ECASwinTransformerModel, self).__init__()
        self.eca = ECA(input_size=484,num_attention_heads=1,hidden_size=484,hidden_dropout_prob=0.5)
        self.swin_transformer = SwinTransformer(
                                                in_chans=30,
                                                num_classes=2,
                                                embed_dim=192,
                                                depths=(2, 2, 18, 2),
                                                num_heads=(6, 12, 24, 48)
                                                )

    def forward(self, x):
        x = x.view(x.size(0),30,484)
        x = self.eca(x)
        x = x.view(x.size(0),30,22,22)
        x = self.swin_transformer(x)
        return x


dataset = StrokePatientsMIDataset(root_path='./subdataset',
                        chunk_size=500,  # 1 second
                        overlap = 0,
                        offline_transform=Compose(
                                [BaselineCorrection(),SetSamplingRate(500,484),
                                BandSignal(sampling_rate=484,band_dict={'frequency_range':[8,40]})]),
                        online_transform=Compose(
                                [ToTensor()]),
                        label_transform=Select('label'),
                        num_worker=8
)


k_fold = KFoldPerSubject(
    n_splits=5,
    shuffle=True,
    random_state=42)

training_metrics = []
test_metrics = []

for i, (training_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
    delete_folder_if_exists(target_folder_name='lightning_logs')
    model = ECASwinTransformerModel(num_channels=30)
    trainer = ClassifierTrainer(model=model,
                                num_classes=2,
                                lr=HYPERPARAMETERS['lr'],
                                weight_decay=HYPERPARAMETERS['weight_decay'],
                                metrics=["accuracy"],
                                accelerator="gpu")
    
    training_loader = DataLoader(training_dataset,
                             batch_size=HYPERPARAMETERS['batch_size'],
                             shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=HYPERPARAMETERS['batch_size'],
                             shuffle=False)
    # 提前停止回调
    early_stopping_callback = EarlyStopping(
        monitor='train_loss',
        patience=8, 
        mode='min',
        verbose=True
    )
    
    trainer.fit(training_loader,
                test_loader,
                max_epochs=HYPERPARAMETERS['num_epochs'],
                callbacks=[early_stopping_callback],
                enable_progress_bar=True,
                enable_model_summary=False,
                limit_val_batches=0.0)

    test_result = trainer.test(test_loader,
                               enable_progress_bar=True,
                               enable_model_summary=True)[0]
    test_metrics.append(test_result["test_accuracy"])
     
print({
    "test_metric_avg": np.mean(test_metrics),
    "test_metric_std": np.std(test_metrics)
})

for i, score in enumerate(test_metrics):
    print(i,score)

print({
    "len": len(test_metrics),
    "test_metric_avg": np.mean(test_metrics),
    "test_metric_std": np.std(test_metrics)
})