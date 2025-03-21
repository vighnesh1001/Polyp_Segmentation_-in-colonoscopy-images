import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms
import segmentation_models_pytorch as smp
from load_and_train.dataloader import polypSegmentationDataset
from utils.loss_func import DICE_BCE_Loss
from load_and_train.trainning import train
from utils.plots import plot_dice_coefficients, plot_iou_scores, plot_training_metrics
from utils.pred_test import *
from albumentations.pytorch import ToTensorV2
import albumentations as A
from utils.Image_onlu_normalize import mask_scaling





root = 'Data/kvasir-seg/Kvasir-SEG/'
img_dir = os.path.join(root, 'images')
mask_dir = os.path.join(root, 'masks')
img_paths = os.listdir(img_dir)
mask_paths = os.listdir(mask_dir)

train_images, test_images, train_masks, test_masks = train_test_split(
    img_paths, mask_paths, test_size=0.2, random_state=42
)

val_images, test_images, val_masks, test_masks = train_test_split(
    test_images, test_masks, test_size=0.5, random_state=42
)


transform = A.Compose([
    A.Rotate(limit=30, p=1.0),      
    A.HorizontalFlip(p=0.5),             
    A.VerticalFlip(p=0.5),               
    A.Resize(256, 256),                  
    A.Lambda(mask=mask_scaling),  # Use regular function here
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0),
    ToTensorV2()                       
],
additional_targets={'mask': 'mask'}
)
root_dir = 'Data/kvasir-seg/Kvasir-SEG/'

 


train_dataset = polypSegmentationDataset(root_dir, train_images, train_masks,transform=transform)
val_dataset = polypSegmentationDataset(root_dir, val_images, val_masks,transform=transform)
test_dataset = polypSegmentationDataset(root_dir, test_images, test_masks,transform=transform)
batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(
    encoder_name='resnet34',        # Choose encoder, e.g., resnet34, efficientnet-b0, etc.
    encoder_weights='imagenet',     # Use pre-trained weights; set to None if not required
    in_channels=3,                  # Number of input channels (e.g., 3 for RGB images)
    classes=1,                      # Number of output channels (e.g., 1 for binary segmentation)
).to(device)


loss_fn = DICE_BCE_Loss()  # Custom loss function combining Dice and BCE losses
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




    