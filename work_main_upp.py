import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import *
from dataloader import polypSegmentationDataset
from model import *
from loss_func import *
from trainning import *
from plots import *
from test import *
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import segmentation_models_pytorch as smp
from PIL import Image
import matplotlib.pyplot as plt
from pred_test import *



train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


root_dir = 'Data/kvasir-seg/Kvasir-SEG/'
image_filenames = os.listdir(os.path.join(root_dir, 'images'))
mask_filenames = os.listdir(os.path.join(root_dir, 'masks'))

train_images, test_images, train_masks, test_masks = train_test_split(
    image_filenames, mask_filenames, test_size=0.2, random_state=42
)

val_images, test_images, val_masks, test_masks = train_test_split(
    test_images, test_masks, test_size=0.5, random_state=42
)

train_dataset = polypSegmentationDataset(root_dir, train_images, train_masks,
                                         transform=train_transform, mask_transform=mask_transform)
val_dataset = polypSegmentationDataset(root_dir, val_images, val_masks,
                                       transform=test_transform, mask_transform=mask_transform)
test_dataset = polypSegmentationDataset(root_dir, test_images, test_masks,
                                        transform=test_transform, mask_transform=mask_transform)

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.UnetPlusPlus(
    encoder_name='resnet34',        
    encoder_weights='imagenet',     
    in_channels=3,                  
    classes=1,                          
).to(device)

loss_fn = DICE_BCE_Loss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader),
                       epochs=30, anneal_strategy='cos', cycle_momentum=False)


metrics = train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=30, scheduler=scheduler)



train_losses = metrics['train_losses']
val_losses = metrics['val_losses']
train_dices = metrics['train_dices']
val_dices = metrics['val_dices']
train_ious = metrics['train_ious']
val_ious = metrics['val_ious']

plot_dice_coefficients(train_dices, val_dices)
plot_iou_scores(train_ious, val_ious)
plot_training_metrics(train_losses, val_losses, train_dices, val_dices, train_ious, val_ious)

print(f"train_dice: {train_dices[-1]:.4f} | val_dice: {val_dices[-1]:.4f}")
print(f"train_iou: {train_ious[-1]:.4f} | val_iou: {val_ious[-1]:.4f}")
print(f"train_loss: {train_losses[-1]:.4f} | val_loss: {val_losses[-1]:.4f}")

visualize_predictions(model, val_loader, device, num_samples=5)


model_save_path = "models/unetpp_model.pth"

torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

