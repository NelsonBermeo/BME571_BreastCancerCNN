# This file will 
# 1) Load dataset 
# 2) Create DataLoader
# 3) Choose Model
# 4) Train
# 5) Save the best

import torch # Import main libary
import torch.nn as nn # Import pytorch neural network tools 
from torch.utils.data import DataLoader # Import dataloader to give model images in batches like 32 images at a time
from torchvision import datasets, transforms # Import datasets to load images from folders & transforms to resize/augment images
from model import BasicCNN, get_resnet18 #Imports the models from model.py

# No access to GPU 
device = torch.device("cpu")

train_transforms = transforms.Compose([ # This prepares training images 
    transforms.Resize((224, 224)), # Resize 
    transforms.RandomHorizontalFlip(), # Randomly flip for 
    transforms.RandomRotation(10), # randomly rotate to add data varitety so the model doesn't overfit 
    transforms.ToTensor(), # converts image to PyTorch tensor which is a data structure in pytorch optimized for ml 
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)), #Validation will not be augmented
    transforms.ToTensor(), # converts images to tensor
])

train_dataset = datasets.ImageFolder("data/processed/train",  transform=train_transforms) 
val_dataset = datasets.ImageFolder("data/processed/val", transform=eval_transforms) # loads folders with data and applies transformations

print("Class mapping:", train_dataset.class_to_idx) # class to index is just our binary 0 1
print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset)) # Some basic stats 


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # This gives the model 32 images at a time and shuffling to mix training images even more
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#model = BasicCNN(num_classes=2) # Comment out line depending on whicih model to train
model = get_resnet18(num_classes=2)

model = model.to(torch.device("cpu")) # Training on cpu

criterion = nn.CrossEntropyLoss() # Standard loss for classification with 2+ classes 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # We use adam to update the weights with a learning rate of 0.001 which is a normal default that works great with adam


def train_one_epoch(model, loader): # This trains the model once through the training dataset 
    model.train() 
    correct = 0
    total_loss = 0 # These track performance during each epoch
    total = 0

    for images, labels in loader: # Traning loop
        images = images.to(device) # Each loop gives one match the shape and labels then moves to CPU
        labels = labels.to(device)

        outputs = model(images) # Here the model makes predictions 
        loss = criterion(outputs, labels) # Measures how wrong the model was (error score)

        optimizer.zero_grad() # Back propigation steps - clear old gradients
        loss.backward() # Compute how each weight contributed to err
        optimizer.step() # Updates weight

        total_loss += loss.item() # Tracking loss

        _, preds = torch.max(outputs, 1) # Gets the class with the highest score 
        correct += (preds == labels).sum().item() # Counts how many predictions were correct 
        total += labels.size(0) # Adds batch size to total count 

    avg_loss = total_loss / len(loader) # Return stats
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, loader): # This checks the model performance on validation data ; validation is important to tune hyperparameters and prevent overfitting 
    model.eval() # Puts the model into evaluate mode
    total_loss = 0 # Stats
    correct = 0
    total = 0

    with torch.no_grad(): # Tells PyTorch not to calculate gradients because validation does not train the model we only test it 

    # The rest of the code is similar to training except there is no weight update
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


num_epochs = 10 # Train for 10 full passes through the training set
best_val_acc = 0 # Track best validation accuracy so far 

for epoch in range(num_epochs): # Loop through 10 epochs 
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader) # Train once then validate once 

    print(f"Epoch {epoch+1}/{num_epochs}") #Print results
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    # Ex: 
    # Epoch 1/10
    # Train Loss: 0.6500 | Train Acc: 0.7000
    # Val Loss:   0.6100 | Val Acc:   0.7400

    if val_acc > best_val_acc: # We save the model when the validation accuracy improves
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/resnet18.pth") # We are saving the weights instead of the pull python object 
        print("  Saved best model")
