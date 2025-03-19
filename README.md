# Polyp_Segmentation_-in-colonoscopy-images


U-Net is a convolutional neural network architecture designed specifically for biomedical image segmentation. It was introduced in 2015 by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in their paper "U-Net: Convolutional Networks for Biomedical Image Segmentation."
Key Features
The U-Net architecture has several distinctive characteristics:

U-shaped design: The network consists of a contracting path (encoder) and an expansive path (decoder), forming a U-shape.
Contracting path: A typical CNN that captures context through a series of convolutions and max pooling operations, reducing spatial dimensions while increasing feature channels.
Expansive path: Performs upsampling of feature maps and combines them with high-resolution features from the contracting path via skip connections.
Skip connections: Direct connections between corresponding layers in the contracting and expansive paths, allowing the network to preserve spatial information lost during downsampling.
Symmetric architecture: The expansive path is roughly symmetric to the contracting path, yielding a u-shaped architecture.



U-Net's skip connection design effectively combines spatial information from the contracting path with contextual information from the expansive path, addressing the fundamental trade-off between localization accuracy and the use of context that existed in previous fully convolutional networks.



## Dice Coefficient (DICE)

Definition: Measures overlap between two segmentation masks
Formula: DICE = 2|A∩B|/(|A|+|B|) = 2TP/(2TP+FP+FN)
Range: [0,1] where 1 indicates perfect overlap
Use case: Commonly used in medical image segmentation; more sensitive to true positives

## Intersection over Union (IoU)

Definition: Measures overlap divided by union of two segmentation masks
Formula: IoU = |A∩B|/|A∪B| = TP/(TP+FP+FN)
Range: [0,1] where 1 indicates perfect overlap
Use case: Standard in computer vision benchmarks; more strictly penalizes errors




## Data set - Segmented Polyp Dataset for Computer Aided Gastrointestinal Disease Detection.
The Kvasir-SEG dataset consists of 1,000 polyp images with corresponding ground truth segmentation masks, collected during real colonoscopy examinations. This dataset serves as a valuable resource for developing and evaluating medical image segmentation algorithms focused on colorectal cancer screening.
Dataset Specifications

Images: High-resolution RGB images in JPG format
Segmentation Masks: Binary masks in PNG format
Resolution: Variable, predominantly 720×576 pixels
Annotations: Pixel-level annotations by medical experts, verified by experienced endoscopists
