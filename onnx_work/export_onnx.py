import torch
import onnx
import os
import sys
from model import UNet

def export_to_onnx(model_path="model/unet_model.pth", save_path="model/unet.onnx"):
    """Export PyTorch model to ONNX format"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = UNet(3,1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    
    try:
        torch.onnx.export(
            model,
            dummy_input,               
            save_path,                 
            export_params=True,        
            opset_version=12,          
            do_constant_folding=True, 
            input_names=['input'],    
            output_names=['output'],   
            dynamic_axes={             
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print(f"Model exported to {save_path}")
        return save_path
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")
        sys.exit(1)

if __name__ == "__main__":
    model_path = "model/attention_unet_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: PyTorch model file not found at {model_path}")
        print("Please make sure to train the model first.")
        sys.exit(1)
    
    export_to_onnx(model_path) 