# This is the evaluation file which will check how good our model preforms on unseen data! 
# This file will
# 1) Load test data 
# 2) Load trained model 
# 3) Make predictions 
# 4) Compute metrics 

import torch # Standard library
from torch.utils.data import DataLoader # To test images in batches 
from torchvision import datasets, transforms # Loads images from folders with dataset and transforms to preprocess them  
from sklearn.metrics import classification_report, confusion_matrix # These will be our metrics for the model 
from model import BasicCNN, get_resnet18 # This is our trained model architecture 

device = torch.device("cpu") # I do not have a gpu, just a mac cpu

test_transforms = transforms.Compose([ # We resize just the image not augmentations 
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # We turn the image into a PyTorch Tensor Object 
])

test_dataset = datasets.ImageFolder("data/processed/test", transform=test_transforms) # Loads the test dataset 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Loads test data in batches of 32 ; We don't shuffle because order does not matter for evaluation

print("Class mapping:", test_dataset.class_to_idx) # Class mapping reminder

# model = BasicCNN(num_classes=2) # Load our beauitufl and sexy model
# model.load_state_dict(torch.load("models/basic_cnn.pth", map_location=device)) # Loads the trained weights from the saved model we had 

model = get_resnet18(num_classes=2)
model.load_state_dict(torch.load("models/resnet18.pth", map_location=device))

model = model.to(device) # Move the model to the CPU 
model.eval() # Turn off the training behavior and go into eval mode 

# Lists to collect the results 
all_preds = []
all_labels = []

with torch.no_grad(): # Evaluation loop without gradients 
    for images, labels in test_loader: # Loop through batches of test data 
        images = images.to(device) # Cpu again 

        outputs = model(images)
        _, preds = torch.max(outputs, 1) # Pick the highest score to determine what class 

        all_preds.extend(preds.cpu().numpy()) #Store results 
        all_labels.extend(labels.numpy())

#Print out report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

accuracy = sum([p == t for p, t in zip(all_preds, all_labels)]) / len(all_labels)
print("\nAccuracy:", accuracy)