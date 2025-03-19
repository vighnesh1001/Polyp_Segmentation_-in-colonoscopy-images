from dataloader import *
from dataloader import polypSegmentationDataset
from model import *
from loss_func import *
from trainning import *
from plots import *
from test import *

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = 'Data/kvasir-seg/Kvasir-SEG/'
    img_dir = 'Data/kvasir-seg/Kvasir-SEG/images'
    mask_dir = 'Data/kvasir-seg/Kvasir-SEG/masks'
    img_path = os.listdir(img_dir)   
    mask_path = os.listdir(mask_dir) 

    train_images, test_images, train_masks, test_masks = train_test_split(
    img_path, mask_path, test_size=0.2, random_state=42
    )

    val_images, test_images, val_masks, test_masks = train_test_split(
    test_images, test_masks, test_size=0.5, random_state=42
    )

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ElasticTransform(alpha=50.0, sigma=5.0),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    mask_transforms = transforms.Compose([
        transforms.Resize((256, 256)),  # Match the image size
        transforms.ToTensor()
    ])


    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    


    train_dataset = polypSegmentationDataset(root,train_images,train_masks,transform=transform,mask_transform=mask_transforms)
    test_dataset = polypSegmentationDataset(root,test_images,test_masks,transform=test_transform,mask_transform=mask_transforms)
    val_dataset = polypSegmentationDataset(root,val_images,val_masks,transform=test_transform,mask_transform=mask_transforms)
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    
   
    model = UNet(3,1).to(device)
    loss_fn = DICE_BCE_Loss() 
    #criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=30, anneal_strategy='cos', cycle_momentum=False)
    metrics = train(model, train_loader, test_loader, optimizer, loss_fn, device, epochs=30,scheduler=scheduler)

    train_losses    = metrics['train_losses']
    val_losses      = metrics['val_losses']
    train_dices     = metrics['train_dices']
    val_dices       = metrics['val_dices']
    train_ious      = metrics['train_ious']
    val_ious        = metrics['val_ious']
    

    plot_dice_coefficients(train_dices, val_dices)
    plot_iou_scores(train_ious, val_ious)
    plot_training_metrics(train_losses, val_losses, train_dices, val_dices, train_ious, val_ious)
    

    print(f"train_dice: {train_dices[-1]:.4f} | val_dice: {val_dices[-1]:.4f}")
    print(f"train_iou: {train_ious[-1]:.4f} | val_iou: {val_ious[-1]:.4f}")
    print(f"train_loss: {train_losses[-1]:.4f} | val_loss: {val_losses[-1]:.4f}")
    

    model_save_path = "model/attention_unet_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
   
    images, masks = next(iter(test_loader))
    with torch.no_grad():
        
        outputs = model(images.to(device)).cpu().detach()
        
        preds = (torch.sigmoid(outputs) > 0.5).float()
    
    
    display_batch(images, masks, preds)