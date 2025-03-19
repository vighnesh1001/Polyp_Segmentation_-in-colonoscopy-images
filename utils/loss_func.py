from dataloader import *


import torch
import torch.nn as nn

class DICE_BCE_Loss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()  # Handles raw logits properly

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)  # Convert logits to probabilities

        intersection = 2 * (logits * targets).sum() + self.smooth
        union = logits.sum() + targets.sum() + self.smooth
        dice_loss = 1. - (intersection / union)

        bce_loss = self.bce(logits, targets)

        return dice_loss + bce_loss

def DiceCoefficient(logits, targets, smooth=1):
    logits = torch.sigmoid(logits)  # Ensure values are between 0 and 1

    intersection = 2 * (logits * targets).sum() + smooth
    union = logits.sum() + targets.sum() + smooth

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
