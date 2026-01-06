import torch.nn as nn
from torchvision.models import vgg16

def get_model():
    model = vgg16(pretrained=False)

    model.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(512, 10)
    )

    return model
