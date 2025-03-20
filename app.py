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

def onnx_inference(image, onnx_path="model/unet.onnx"):
    image = np.array(image)
    
    original_h, original_w = image.shape[:2]
    
    
    input_tensor = preprocess_image(image)
    
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

def predict(image, model_path="model/unet.onnx"):
    mask, overlay = onnx_inference(image, model_path)
    return mask, overlay

examples_dir = "examples"
if not os.path.exists(examples_dir):
    os.makedirs(examples_dir)

def create_interface(model_path="model/unet.onnx"):
    demo = gr.Interface(
        fn=lambda img: predict(img, model_path),
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Image(type="numpy", label="Segmentation Mask"),
            gr.Image(type="numpy", label="Overlay")
        ],
        title="Polyp Segmentation with UNet",
        description="Upload an image to detect polyps in endoscopic images.",
        examples=["examples/example1.jpg", "examples/example2.jpg"] if os.path.exists("examples/example1.jpg") else None
    )
    return demo

if __name__ == "__main__":
    model_path = "model/unet.onnx"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure to train the model and export it to ONNX format first.")
        exit(1)
    
    demo = create_interface(model_path)
    demo.launch() 