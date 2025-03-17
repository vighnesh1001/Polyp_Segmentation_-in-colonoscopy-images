from dataloader import *


class DICE_BCE_Loss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        intersection = 2*(logits * targets).sum() + self.smooth
        union = (logits + targets).sum() + self.smooth
        dice_loss = 1. - intersection / union

        loss = nn.BCELoss() 
        bce_loss = loss(logits, targets)

        return dice_loss + bce_loss
    
def DiceLoss(logits, targets):
    intersection = 2*(logits * targets).sum()
    union = (logits + targets).sum()
    if union == 0:
        return 1
    dice_coeff = intersection / union
    return dice_coeff.item()



def compute_iou(outputs, targets, threshold=0.5, smooth=1e-6):
    
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def calculate_metrics(outputs, targets, threshold=0.5):
   
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    accuracy = correct / total
    true_positives = ((preds == 1) & (targets == 1)).sum().item()
    predicted_positives = (preds == 1).sum().item()
    precision = true_positives / (predicted_positives + 1e-6)  

    return accuracy, precision
