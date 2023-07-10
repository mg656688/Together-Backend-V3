import torchmetrics
import torchvision.models as models
import pytorch_lightning as pl
import torch.nn as nn


class ResNet50Model(pl.LightningModule):

    def __init__(self, pretrained=True, in_channels=3, num_classes=15, lr=3e-4, freeze=False):
        super(ResNet50Model, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr

        self.model = models.resnet50(pretrained=pretrained)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)
