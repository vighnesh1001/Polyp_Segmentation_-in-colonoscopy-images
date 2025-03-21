import gradio as gr
import numpy as np
import cv2
import onnxruntime
import os
import torch
from PIL import Image

def preprocess_image(image):
  
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    
    image = cv2.resize(image, (224, 224))
    
    image = image / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    image = image.transpose(2, 0, 1)
    
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def onnx_inference(image, onnx_path="model/unetpp.onnx"):
   
    image = np.array(image)
    
    original_h, original_w = image.shape[:2]
    
    input_tensor = preprocess_image(image)
    
    try:
        session = onnxruntime.InferenceSession(onnx_path)
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        output = session.run([output_name], {input_name: input_tensor})[0]
        
        mask = output[0, 0] 
        mask = (mask > 0.5).astype(np.uint8) * 255  
        mask = cv2.resize(mask, (original_w, original_h))  
        
       
        colored_mask = np.zeros((original_h, original_w, 3), dtype=np.uint8)
        colored_mask[mask > 0] = [255, 0, 0] 
        
        
        alpha = 0.5
        overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
        
        return mask, overlay
    except Exception as e:
        print(f"Error during inference: {e}")
        
        blank_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        return blank_mask, image

def predict(image, model_path="model/unetpp.onnx"):
   
    if image is None:
        return None, None
    mask, overlay = onnx_inference(image, model_path)
    return mask, overlay

def create_interface(model_path="model/unetpp.onnx"):
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("The interface will be created but won't work until the model is available.")
    
    
    examples_dir = "examples"
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
    
   
    example_files = []
    if os.path.exists(examples_dir):
        example_files = [
            os.path.join(examples_dir, f) 
            for f in os.listdir(examples_dir) 
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ]
    
    demo = gr.Interface(
        fn=lambda img: predict(img, model_path),
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Image(type="numpy", label="Segmentation Mask"),
            gr.Image(type="numpy", label="Overlay")
        ],
        title="Polyp Segmentation with UNet++",
        description="Upload an endoscopic image to detect polyps using UNet++ segmentation.",
        examples=example_files if example_files else None,
        allow_flagging="never"
    )
    return demo

if __name__ == "__main__":
    possible_model_paths = [
        "model/unetpp.onnx",
        "models/unetpp.onnx",
        "model/unet.onnx" 
    ]
    
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("Error: ONNX model file not found. Checked the following paths:")
        for path in possible_model_paths:
            print(f"  - {path}")
        print("\nPlease export your model to ONNX format first using export_onnx.py")
        exit(1)
    
    print(f"Using model: {model_path}")
    demo = create_interface(model_path)
    demo.launch(share=False)  