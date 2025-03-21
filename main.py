import os
import argparse
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Local imports
from load_and_train.dataloader import polypSegmentationDataset
from utils.loss_func import DICE_BCE_Loss
from load_and_train.trainning import train
from utils.plots import (
    plot_dice_coefficients, 
    plot_iou_scores, 
    plot_training_metrics
)
from utils.pred_test import visualize_predictions
from utils.Image_onlu_normalize import mask_scaling


def prepare_data(root_dir, test_size=0.2, val_size=0.5, random_state=42):
   
    image_filenames = os.listdir(os.path.join(root_dir, 'images'))
    mask_filenames = os.listdir(os.path.join(root_dir, 'masks'))
    
    
    train_images, test_images, train_masks, test_masks = train_test_split(
        image_filenames, mask_filenames, test_size=test_size, random_state=random_state
    )
    
    
    val_images, test_images, val_masks, test_masks = train_test_split(
        test_images, test_masks, test_size=val_size, random_state=random_state
    )
    
   
    transform = A.Compose([
        A.Rotate(limit=30, p=1.0),      
        A.HorizontalFlip(p=0.5),             
        A.VerticalFlip(p=0.5),               
        A.Resize(256, 256),                  
        A.Lambda(mask=mask_scaling),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),
        ToTensorV2()                       
    ],
    additional_targets={'mask': 'mask'}
    )
    
   
    train_dataset = polypSegmentationDataset(root_dir, train_images, train_masks, transform=transform)
    val_dataset = polypSegmentationDataset(root_dir, val_images, val_masks, transform=transform)
    test_dataset = polypSegmentationDataset(root_dir, test_images, test_masks, transform=transform)
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=2):
  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def create_model(device, encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1):
   
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    ).to(device)
    
    return model


def train_model(model, train_loader, val_loader, device, epochs=30, lr=1e-4, weight_decay=1e-5):
  
    loss_fn = DICE_BCE_Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
   
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        steps_per_epoch=len(train_loader),
        epochs=epochs, 
        anneal_strategy='cos', 
        cycle_momentum=False
    )
    
    
    metrics = train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=epochs, scheduler=scheduler)
    
    return metrics


def evaluate_and_visualize(model, val_loader, metrics, device, num_samples=5):
   
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
    
   
    visualize_predictions(model, val_loader, device, num_samples=num_samples)


def save_model(model, save_path="model/unetpp_model.pth"):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def main(args):
    
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
   
    train_dataset, val_dataset, test_dataset = prepare_data(
        args.data_dir, 
        test_size=args.test_size, 
        val_size=args.val_size, 
        random_state=args.random_state
    )
    
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, 
        val_dataset, 
        test_dataset, 
        batch_size=args.batch_size
    )
    
  
    model = create_model(
        device, 
        encoder_name=args.encoder_name, 
        encoder_weights=args.encoder_weights, 
        in_channels=args.in_channels, 
        classes=args.classes
    )
   
    metrics = train_model(
        model, 
        train_loader, 
        val_loader, 
        device, 
        epochs=args.epochs, 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
   
    evaluate_and_visualize(
        model, 
        val_loader, 
        metrics, 
        device, 
        num_samples=args.num_samples
    )
    
   
    save_model(model, save_path=args.model_save_path)


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Train UNet++ for polyp segmentation")
    
   
    parser.add_argument("--data_dir", type=str, default="Data/kvasir-seg/Kvasir-SEG/", 
                        help="Path to dataset directory")
    parser.add_argument("--test_size", type=float, default=0.2, 
                        help="Proportion of data to use for testing")
    parser.add_argument("--val_size", type=float, default=0.5, 
                        help="Proportion of test data to use for validation")
    parser.add_argument("--random_state", type=int, default=42, 
                        help="Random seed for reproducibility")
    
   
    parser.add_argument("--encoder_name", type=str, default="resnet34", 
                        help="Name of the encoder backbone")
    parser.add_argument("--encoder_weights", type=str, default="imagenet", 
                        help="Pre-trained weights for the encoder")
    parser.add_argument("--in_channels", type=int, default=3, 
                        help="Number of input channels")
    parser.add_argument("--classes", type=int, default=1, 
                        help="Number of output classes")
    
    
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, 
                        help="Weight decay for regularization")
    
   
    parser.add_argument("--num_samples", type=int, default=5, 
                        help="Number of samples to visualize")
    
    
    parser.add_argument("--model_save_path", type=str, default="model/unetpp_model.pth", 
                        help="Path to save the trained model")
    
    args = parser.parse_args()
    
  
    main(args)

