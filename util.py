import torchvision
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
from model import FoodFinder
import io
from flask import Flask, request, send_file, jsonify

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = r"model\five_foods_model.pth"

# Create a new instance of FashionMNISTModelV2 (the same class as our saved state_dict())
# Note: loading model will error if the shapes here aren't the same as the saved version
num_classes = 5  # Number of classes in your dataset
loaded_model = models.resnet50(pretrained=False)  # Load ResNet-50 without pre-trained weights

# Modify the final fully connected layer to match the number of classes
loaded_model.fc = nn.Linear(in_features=loaded_model.fc.in_features, out_features=num_classes) 

# Load in the saved state_dict()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, map_location=torch.device('cpu')))

# Send model to GPU
loaded_model = loaded_model.to(device)

# Load your trained model
loaded_model.eval()

# Define image transformations (adjust based on your model's requirements)
custom_image_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to the input size expected by your model
    transforms.ToTensor() # Normalize with ImageNet statistics
])

def pred_plot(model:torch.nn.Module,
              image_path:str,
              class_names:list[str],
              transform=None,
              device="cpu"):
  if transform:
    target_image = transform(Image.open(image_path))

  model.to(device)

  # model.eval()
  with torch.inference_mode():
    target = target_image.unsqueeze(dim=0)
    y_pred = model(target.to(device))

  y_probs = torch.softmax(y_pred,dim=1)
  y_label = torch.argmax(y_probs,dim=1)
  # plt.imshow(target_image.permute(1,2,0))
  if class_names:
    title = f"prediction : {class_names[y_label.cpu()]} | probability : {y_probs.max().cpu():.3f}"
  else:
      title = f"Pred: {y_label.item()} | Prob: {y_probs.max().cpu():.3f}"
  
  response = {'prediction': title}

  return jsonify(response)



