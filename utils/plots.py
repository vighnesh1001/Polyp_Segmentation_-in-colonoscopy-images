from dataloader import *

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def load_mask(mask_path):
    return Image.open(mask_path).convert("L")

def visualize_image_and_mask(image, mask):
    image_np = np.array(image)
    mask_np = np.array(mask)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

image_dir = 'Data/kvasir-seg/Kvasir-SEG/images'
mask_dir = 'Data/kvasir-seg/Kvasir-SEG/masks'
image_filename = 'cju0qoxqj9q6s0835b43399p4.jpg'
mask_filename = 'cju0qoxqj9q6s0835b43399p4.jpg'

image_path = os.path.join(image_dir, image_filename)
mask_path = os.path.join(mask_dir, mask_filename)

image = load_image(image_path)
mask = load_mask(mask_path)

visualize_image_and_mask(image, mask)




def plot_dice_coefficients(train_dice, val_dice, title="Dice Coefficient over Epochs",
                             xlabel="Epoch", ylabel="Dice Coefficient", save_path=None):
    
    if len(train_dice) != len(val_dice):
        raise ValueError("The length of train_dice and val_dice must be the same.")

    epochs = range(1, len(train_dice) + 1)
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, train_dice, marker='o', linestyle='-', color='blue', label='Training Dice')
    ax.plot(epochs, val_dice, marker='s', linestyle='--', color='red', label='Validation Dice')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(epochs)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()
    return fig, ax





def plot_iou_scores(train_iou, val_iou, title="IoU Score over Epochs",
                    xlabel="Epoch", ylabel="IoU Score", save_path=None):
    
    if len(train_iou) != len(val_iou):
        raise ValueError("train_iou and val_iou must have the same length.")
    
    epochs = range(1, len(train_iou) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, train_iou, marker='o', linestyle='-', color='green', label='Training IoU')
    ax.plot(epochs, val_iou, marker='s', linestyle='--', color='orange', label='Validation IoU')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(epochs)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()
    return fig, ax




def plot_probability_heatmap(prob_map, original_image=None, colormap='jet', alpha=0.5, save_path=None):
    
    prob_map = np.clip(prob_map, 0, 1)
    
    cmap = plt.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    heatmap = cmap(norm(prob_map))
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if original_image is not None:
        ax.imshow(original_image)
        ax.imshow(heatmap, alpha=alpha)
        ax.set_title("Probability Heatmap Overlay")
    else:
        ax.imshow(heatmap)
        ax.set_title("Probability Heatmap")
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()
    return fig, ax





def display_batch(images, masks, pred):
    images = images.permute(0, 2, 3, 1)
    masks = masks.permute(0, 2, 3, 1)
    pred = pred.permute(0, 2, 3, 1)
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    pred = pred.cpu().numpy()
    images_concat = np.concatenate(images, axis=1)
    masks_concat = np.concatenate(masks, axis=1)
    pred_concat = np.concatenate(pred, axis=1)
    fig, ax = plt.subplots(3, 1, figsize=(20, 12))
    fig.tight_layout(pad=3.0)
    ax[0].imshow(images_concat.astype(np.uint8))
    ax[0].set_title('Original Images')
    ax[0].axis('off')
    ax[1].imshow(masks_concat.squeeze(), cmap='gray')
    ax[1].set_title('Ground Truth Masks')
    ax[1].axis('off')
    ax[2].imshow(pred_concat.squeeze(), cmap='gray')
    ax[2].set_title('Predicted Masks')
    ax[2].axis('off')
    plt.show()




import matplotlib.pyplot as plt

def plot_training_metrics(train_losses, val_losses, train_dices, val_dices, train_ious, val_ious):
   
    epochs_range = range(1, len(train_losses) + 1)
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs_range, val_losses, label="Val Loss", marker='s')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_dices, label="Train Dice", marker='o')
    plt.plot(epochs_range, val_dices, label="Val Dice", marker='s')
    plt.title("Dice Coefficient over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_ious, label="Train IoU", marker='o')
    plt.plot(epochs_range, val_ious, label="Val IoU", marker='s')
    plt.title("IoU over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
