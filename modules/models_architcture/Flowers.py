import torch
import torchvision.models as models


# Define your model architecture
class ResNet50Model(torch.nn.Module):
    def __init__(self, pretrained=True, in_channels=3, num_classes=16):
        super(ResNet50Model, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.model = models.resnet50(pretrained=pretrained)

        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(self.model.fc.in_features, 128),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        return self.model(x)
