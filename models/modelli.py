import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models import vgg11_bn, VGG11_BN_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import torch.nn.functional as F
import matplotlib.pyplot as plt
import timm

#from lib.fastkan.fastkan import FastKANLayer
#from lib.fasterkan.fasterkan_layers import FasterKANLayer
from lib.efficient_kan.kan import KANLinear as EfficientKan
#from lib.kan.KANLayer import KANLayer
from lib.fourier_kan.KAF import FastKAFLayer

class EFFNETB0_UNLOCK_DROP_FOURIER_6(nn.Module):
    """
        EfficientNet-B0-based model with unlocked final layers and a FourierKan head for 6-parameter regression.

        This architecture is designed for single-view input images, leveraging a pretrained
        EfficientNet-B0 backbone (with global average pooling) followed by a custom FourierKan layer 
        to predict 6 continuous outputs, with dropout applied before the final regression layer.

        Args:
            num_output (int): Number of regression outputs. Default is 6.
            dropout_rate (float): Dropout rate applied before the final layer. Default is 0.1.
            nome_rete (str): Optional model name identifier for tracking/logging.

        Returns:
            torch.Tensor: Output tensor of shape [B, num_output] representing the predicted parameters.
    """
    def __init__(self, num_output=6, dropout_rate=0.1, nome_rete=None):
        super().__init__()
        self.nome_rete = nome_rete

        # Load EfficientNet-B0 backbone without classifier (global_pool='avg')
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        # Unfreeze all layers for fine-tuning
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.fc1 = FastKAFLayer(input_dim=1280, output_dim=num_output, use_layernorm=False)
        self.dropout = nn.Dropout(dropout_rate)

        self.name = nome_rete

    def forward(self, x):
        """
            Forward pass of the model.

            Args:
                x (torch.Tensor): Input tensor of shape [B, 3, 224, 224] representing a batch of RGB images.

            Returns:
                torch.Tensor: Output tensor of shape [B, num_output] (default: [B, 6]).
        """
        features = self.backbone(x)        # [B, 1280]

        features = self.dropout(features)

        x = self.fc1(features)             # [B, output_dim]

        return x

class EFFNETB0_UNLOCK_DROP_EFFK_6(nn.Module):
    """
        EfficientNet-B0-based model with unlocked final layers and a EfficientKan head for 6-parameter regression.

        Args:
            num_output (int): Number of regression outputs. Default is 6.
            dropout_rate (float): Dropout rate applied before the final layer. Default is 0.1.
            nome_rete (str): Optional model name identifier for tracking/logging.

        Returns:
            torch.Tensor: Output tensor of shape [B, num_output] representing the predicted parameters.
    """
    def __init__(self, num_output=6, dropout_rate=0.1, nome_rete=None):
        super().__init__()
        self.nome_rete = nome_rete

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.fc1 = EfficientKan(in_features=1280, out_features=num_output)
        self.dropout = nn.Dropout(dropout_rate)

        self.name = nome_rete

    def forward(self, x):
        features = self.backbone(x)        # [B, 1280]

        features = self.dropout(features)

        x = self.fc1(features)             # [B, out_features]

        return x

class SQUEEZENET_DROP4_CLASSICA_DROP_256(nn.Module):
    """
        SqueezeNet 1.1 based regression model with a custom simple MLP nn.Linear connected head.

        This architecture uses SqueezeNet 1.1 as a lightweight feature extractor, removing 
        its final classifier and applying global average pooling to produce a 512-dimensional 
        feature vector. The output is passed through a custom MLP with multiple dropout layers 
        and ReLU activations to regress to `num_output` continuous values.

        Args:
            num_output (int): Number of regression outputs. Default is 6.
            dropout_rate (float): Dropout probability used between fully connected layers.
            nome_rete (str): Optional model identifier used for logging or naming.

        Returns:
            torch.Tensor: Output tensor of shape [B, num_output] containing predicted values.
    """
    def __init__(self, num_output=6, dropout_rate=0.1, nome_rete=None):
        super().__init__()
        self.nome_rete = nome_rete

        # Load pretrained SqueezeNet 1.1
        self.backbone = models.squeezenet1_1(pretrained=True)

        # Remove the final classifier
        self.backbone.classifier = nn.Identity()

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=32)
        self.fc5 = nn.Linear(in_features=32, out_features=num_output)

        self.relu = nn.ReLU()
        self.name = nome_rete

    def forward(self, x):
        x = self.backbone.features(x)

        x = self.global_avgpool(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc5(x)
        
        return x

class SQUEEZENET_DROP4_FOURIER_DROP_256(nn.Module):
    """
        SqueezeNet 1.1 based regression model with a custom FourierKan connected head.

        Args:
            num_output (int): Number of regression outputs. Default is 6.
            dropout_rate (float): Dropout probability used between fully connected layers.
            nome_rete (str): Optional model identifier used for logging or naming.

        Returns:
            torch.Tensor: Output tensor of shape [B, num_output] containing predicted values.
    """
    def __init__(self, num_output=6, dropout_rate=0.1, nome_rete=None):
        super().__init__()
        self.nome_rete = nome_rete

        self.backbone = models.squeezenet1_1(pretrained=True)

        self.backbone.classifier = nn.Identity()

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc1 = FastKAFLayer(input_dim=512, output_dim=256, use_layernorm=False)
        self.fc2 = FastKAFLayer(input_dim=256, output_dim=128, use_layernorm=False)
        self.fc3 = FastKAFLayer(input_dim=128, output_dim=64, use_layernorm=False)
        self.fc4 = FastKAFLayer(input_dim=64, output_dim=32, use_layernorm=False)
        self.fc5 = FastKAFLayer(input_dim=32, output_dim=num_output, use_layernorm=False)

        self.name = nome_rete

    def forward(self, x):
        x = self.backbone.features(x)

        x = self.global_avgpool(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)        
        x = self.dropout(x)
        x = self.fc2(x)        
        x = self.dropout(x)
        x = self.fc3(x)        
        x = self.dropout(x)
        x = self.fc4(x)        
        x = self.dropout(x)
        x = self.fc5(x)        

        return x