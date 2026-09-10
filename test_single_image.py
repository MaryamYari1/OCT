import torch 
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image 

model = models.resnet18(weights=None)
model.fc = nn.Linear(in_features=512, out_features=1)
model.load_state_dict(torch.load("oct_model.path"))
model.eval()

image_path = "mh_train_1030.jpg"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)
    prob = torch.sigmoid(output)
    prediction = (prob > 0.50).float()

print(f"Row Output: {output.item():.4f}")
print(f"Probability: {prob.item():.4f}")

if prediction.item()==0:
    print("Prediction: Abnormal")
else: 
    print("Prediction: Normal")