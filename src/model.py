#This file holds both our models: The basic CNN and the ResNet 

import torch.nn as nn #Contains neural network layers to build out model
from torchvision import models # Imported the resnet model

# Our basic CNN model from scratch 
class BasicCNN(nn.Module): # We inherit from nn.Module which is a base class for all PyTorch Models
    def __init__(self, num_classes=2): # Constructor with num_classes = 2 for binary classification
        super().__init__() # Initializes PyTorch internals 

        self.features = nn.Sequential( # Our feature extractor will extract patterns from images in 3 layers 
            # Layer 1 - 
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 3 input channels (RGB), 32 output channels (each produces a feature map and captures a different pattern i the image), 3x3 kernel size, 1 padding size to keep image the same
            nn.ReLU(), # ReLU activation function which adds non-linearity. Without this we can't learn complex patterns
            nn.MaxPool2d(2), # Down samples image by 2 - Ex: 10x10 -> 5x5

            # Layer 2 - 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32 input channels from prev layer and outputs 64 
            nn.ReLU(), # ReLU activation
            nn.MaxPool2d(2), # Down samples 

            # Layer 3 - 
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 64 input and outputs 128
            nn.ReLU(), 
            nn.MaxPool2d(2),

            # Over the 3 layers the size changes to 128 channels of 28 x 28
        )

        self.classifier = nn.Sequential( # Classifier that turns features into predictions 
            nn.Flatten(), # Converts our 128 x 28 x 28 into a 100352 vector 
            nn.Linear(128 * 28 * 28, 256), # Inputs huge prev vector to output 256 features 
            nn.ReLU(), # Activation
            nn.Dropout(0.5), # In an attempt to prevent overfitting we drop out neurons; this is important for small datasets which are prone to overfit 
            nn.Linear(256, num_classes), # Outputs binary
        )

    def forward(self, x): # Defines how data flows through the network. Image -> convolution layers ; features -> classifier ; output -> predictions 
        x = self.features(x) 
        x = self.classifier(x)
        return x
    
    # An output can look like: 
    # [2.3, -1.2] -> malignant 
    # [-0.5, 1.8] -> non_malignant
    # Where our loss function can interpret


def get_resnet18(num_classes=2): # Pretrained model 
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # Loads deep CNN which was trained on ImageNet (millions of images)
    model.fc = nn.Linear(model.fc.in_features, num_classes) #Replace final layer with our 2 classes

    return model