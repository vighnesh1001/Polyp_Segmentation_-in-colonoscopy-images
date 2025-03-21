import torch
import numpy as np
import onnxruntime
import cv2
import matplotlib.pyplot as plt
from unet_train.model import UNet
import os
import sys
from torchvision import transforms

def load_pytorch_model(model_path="model/attention_unet_model.pth"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = UNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        sys.exit(1)

def preprocess_for_pytorch(image_path, device):
   
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.array(image_path)
    
    original_h, original_w = image.shape[:2]
    
    image_resized = cv2.resize(image, (224, 224))
    
    image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
    image_tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )(image_tensor)
    
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    return image_tensor, (original_h, original_w), image

def preprocess_for_onnx(image_path):
   
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.array(image_path)
    
    original_h, original_w = image.shape[:2]
    
    
    image_resized = cv2.resize(image, (224, 224))
    
    image_normalized = image_resized / 255.0
    image_normalized = (image_normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    image_chw = image_normalized.transpose(2, 0, 1)
    
    image_batch = np.expand_dims(image_chw, axis=0).astype(np.float32)
    
    return image_batch, (original_h, original_w), image

def pytorch_inference(model, image_tensor, original_dims, device):
   
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        
    mask = output[0, 0].cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (original_dims[1], original_dims[0]))
    
    return mask

def onnx_inference(onnx_path, image_tensor, original_dims):
    
    try:
        session = onnxruntime.InferenceSession(onnx_path)
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        output = session.run([output_name], {input_name: image_tensor})[0]
        
        
        mask = output[0, 0]  
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (original_dims[1], original_dims[0]))
        
        return mask
    except Exception as e:
        print(f"Error running ONNX inference: {e}")
        sys.exit(1)

def compare_outputs(pytorch_output, onnx_output, image, image_path=None):
   
    pixel_match = np.mean(pytorch_output == onnx_output) * 100
    mse = np.mean((pytorch_output.astype(float) - onnx_output.astype(float)) ** 2)
    
    print(f"Pixel Match: {pixel_match:.2f}%")
    print(f"Mean Squared Error: {mse:.6f}")
    
    h, w = pytorch_output.shape[:2]
    pytorch_colored = np.zeros((h, w, 3), dtype=np.uint8)
    onnx_colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    pytorch_colored[pytorch_output > 0] = [255, 0, 0] 
    onnx_colored[onnx_output > 0] = [255, 0, 0]  
    
    alpha = 0.5
    pytorch_overlay = cv2.addWeighted(image, 1, pytorch_colored, alpha, 0)
    onnx_overlay = cv2.addWeighted(image, 1, onnx_colored, alpha, 0)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(pytorch_output, cmap='gray')
    plt.title("PyTorch Model Output")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(pytorch_overlay)
    plt.title("PyTorch Overlay")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(onnx_output, cmap='gray')
    plt.title("ONNX Model Output")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(onnx_overlay)
    plt.title("ONNX Overlay")
    plt.axis('off')
    
    plt.suptitle(f"Model Comparison - Pixel Match: {pixel_match:.2f}%, MSE: {mse:.6f}")
    plt.tight_layout()
    
    if image_path and isinstance(image_path, str):
        filename = os.path.basename(image_path)
        plt.savefig(f"comparison_{filename}.png")
    
    plt.show()

def verify_models(image_path, pytorch_model_path="model/unet_model.pth", onnx_model_path="model/unet.onnx"):
    
    if not os.path.exists(pytorch_model_path):
        print(f"Error: PyTorch model file not found at {pytorch_model_path}")
        sys.exit(1)
    if not os.path.exists(onnx_model_path):
        print(f"Error: ONNX model file not found at {onnx_model_path}")
        sys.exit(1)
    
    pytorch_model, device = load_pytorch_model(pytorch_model_path)
    
    pytorch_input, original_dims, original_image = preprocess_for_pytorch(image_path, device)
    
    onnx_input, _, _ = preprocess_for_onnx(image_path)
    
    pytorch_output = pytorch_inference(pytorch_model, pytorch_input, original_dims, device)
    
    onnx_output = onnx_inference(onnx_model_path, onnx_input, original_dims)
    
    compare_outputs(pytorch_output, onnx_output, original_image, image_path)
    
    return pytorch_output, onnx_output

if __name__ == "__main__":
    pytorch_model_path = "model/attention_unet_model.pth"
    onnx_model_path = "model/unet_main.onnx"
    
    
    image_path = "Data/kvasir-seg/Kvasir-SEG/images/cju0qx73cjw570799j4n5cjze.jpg"
    if not os.path.exists(image_path):
        print(f"Warning: Example image not found at {image_path}")
        print("Please provide a valid image path as an argument.")
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
            image_path = sys.argv[1]
        else:
            print("No valid image path provided. Exiting.")
            sys.exit(1)
    
    print(f"Verifying models with image: {image_path}")
    verify_models(image_path, pytorch_model_path, onnx_model_path) 