import os 
import shutil 
import random
import torch 
import torch.nn as nn 
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

source_base = '/Users/mary/Downloads/CellData/OCT/train'
dest_base = 'data'

n_images_per_class = 150


model = models.resnet18(weights='IMAGENET1K_V1')

for param in model.parameters():
    param.requires_grad = False 

model.fc = nn.Linear(in_features=512, out_features=1)

transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

train_dataset = datasets.ImageFolder('data/train', transform=transform)
print(train_dataset.classes)
print(len(train_dataset))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

val_dataset = datasets.ImageFolder('data/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

num_epochs = 3

for epoch in range(num_epochs):
    running_loss = 0.0
    running_correct = 0.0
    running_total = 0.0


    for images, labels in train_loader:
        labels = labels.float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        probs = torch.sigmoid(outputs)
        predictions = (probs > 0.50).float()
        running_correct += (predictions == labels).sum().item()
        running_total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    avg_accuracy = running_correct / running_total

    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

torch.save(model.state_dict(), "oct_model.path")

model.eval()

all_predictions = [] 
all_labels = [] 

with torch.no_grad():
    for images, labels in val_loader:
        labels = labels.float().unsqueeze(1)
        outputs = model(images)

        probs = torch.sigmoid(outputs)
        predictions = (probs > 0.50).float()
        
        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

cm = confusion_matrix(all_labels, all_predictions)
print(cm)

precision = precision_score(all_labels, all_predictions, pos_label=0)
recall = recall_score(all_labels, all_predictions, pos_label=0)
f1 = f1_score(all_labels, all_predictions, pos_label=0)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")



cnv_folder = os.path.join(source_base, 'CNV')
cnv_files = os.listdir(cnv_folder)
selected_cnv = random.sample(cnv_files, n_images_per_class)