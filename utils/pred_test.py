import matplotlib.pyplot as plt
import torch

def visualize_predictions(model, data_loader, device, num_samples=5):
    model.eval()
    samples = 0

    with torch.no_grad():
        for images, true_masks in data_loader:
            images = images.to(device)
            true_masks = true_masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5

            for i in range(images.size(0)):
                if samples >= num_samples:
                    return
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(images[i].cpu().permute(1, 2, 0))
                axes[0].set_title('Input Image')
                axes[0].axis('off')
                axes[1].imshow(true_masks[i].cpu().squeeze(), cmap='gray')
                axes[1].set_title('Ground Truth Mask')
                axes[1].axis('off')
                axes[2].imshow(preds[i].cpu().squeeze(), cmap='gray')
                axes[2].set_title('Predicted Mask')
                axes[2].axis('off')
                plt.show()
                samples += 1
