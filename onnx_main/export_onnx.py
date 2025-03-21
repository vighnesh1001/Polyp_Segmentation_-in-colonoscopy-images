import torch
import onnx
import os
import sys
import segmentation_models_pytorch as smp

def export_to_onnx(model_path="model/unet_model.pth", save_path="model/unet.onnx"):
   
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
       
        model = smp.UnetPlusPlus(
            encoder_name='resnet34',
            encoder_weights=None,  
            in_channels=3,
            classes=1,
        ).to(device)
        
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
  
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
   
    possible_paths = [
        os.path.join(project_root, "model", "unetpp_model.pth"), 
        os.path.join(project_root, "models", "unetpp_model.pth"),  
        "../model/unetpp_model.pth",  
        "model/unetpp_model.pth",     
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("Error: PyTorch model file not found. Checked the following paths:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease specify the correct path to your model file.")
        sys.exit(1)
    
    
    output_dir = os.path.join(project_root, "model")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "unetpp.onnx")
    print(f"Attempting to export model from: {model_path}")
    print(f"Output will be saved to: {output_path}")
    
    export_to_onnx(model_path, output_path)