from dataloader import *
from model import *
from loss_func import *




def train(model, train_loader, test_loader, optimizer, loss_fn, device, epochs=15, scheduler=None):
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    train_ious, val_ious = [], []
    train_accuracies, val_accuracies = [], []
    train_precisions, val_precisions = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_dice = 0
        train_iou = 0
        train_accuracy = 0
        train_precision = 0
        
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss_value = loss_fn(outputs, masks)
            loss_value.backward()
            optimizer.step()
            
            
            if scheduler is not None:
                scheduler.step()
            
            train_loss += loss_value.item()
            train_dice += DiceLoss(outputs, masks)
            train_iou += compute_iou(outputs, masks)
            acc, prec = calculate_metrics(outputs, masks)
            train_accuracy += acc
            train_precision += prec

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_precision /= len(train_loader)
        
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        train_ious.append(train_iou)
        train_accuracies.append(train_accuracy)
        train_precisions.append(train_precision)
        
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        val_accuracy = 0
        val_precision = 0
        
        with torch.no_grad():
            for i, (images, masks) in enumerate(test_loader):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss_value = loss_fn(outputs, masks)
                val_loss += loss_value.item()
                val_dice += DiceLoss(outputs, masks)
                val_iou += compute_iou(outputs, masks)
                acc, prec = calculate_metrics(outputs, masks)
                val_accuracy += acc
                val_precision += prec
        
        val_loss /= len(test_loader)
        val_dice /= len(test_loader)
        val_iou /= len(test_loader)
        val_accuracy /= len(test_loader)
        val_precision /= len(test_loader)
        
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        val_ious.append(val_iou)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        
        print(f"Epoch: {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train DICE: {train_dice:.4f} | Train IoU: {train_iou:.4f}")
        print(f"  Train Accuracy: {train_accuracy:.4f} | Train Precision: {train_precision:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val DICE: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f} | Val Precision: {val_precision:.4f}")
    
    print("\nFinal Results:")
    print(f"  Train - Loss: {train_losses[-1]:.4f}, DICE: {train_dices[-1]:.4f}, IoU: {train_ious[-1]:.4f}")
    print(f"  Train - Accuracy: {train_accuracies[-1]:.4f}, Precision: {train_precisions[-1]:.4f}")
    print(f"  Val - Loss: {val_losses[-1]:.4f}, DICE: {val_dices[-1]:.4f}, IoU: {val_ious[-1]:.4f}")
    print(f"  Val - Accuracy: {val_accuracies[-1]:.4f}, Precision: {val_precisions[-1]:.4f}")
    
    metrics = {
        'train_losses': train_losses,
        'train_dices': train_dices,
        'train_ious': train_ious,
        'train_accuracies': train_accuracies,
        'train_precisions': train_precisions,
        'val_losses': val_losses,
        'val_dices': val_dices,
        'val_ious': val_ious,
        'val_accuracies': val_accuracies,
        'val_precisions': val_precisions
    }
    
    return metrics
