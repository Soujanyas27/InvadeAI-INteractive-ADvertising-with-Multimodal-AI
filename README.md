# InvadeAI-INteractive-ADvertising-with-Multimodal-AI

## Overview
This project explores the integration of **computer vision and large language models (LLMs)** for **interactive advertising**. The system enables seamless product discovery and purchasing within visual media by detecting products in images and retrieving purchase links.

The key components of this project include:
- **Object Detection**: Utilizes a fine-tuned **YOLOv8** model for identifying consumer products.
- **Image Captioning**: Leverages **BLIP3** for generating detailed descriptions of detected objects.
- **Product Link Retrieval**: Employs **LLaMA3 and Gemini 1.5 Pro** to fetch e-commerce product links.

## Motivation
Interactive advertising is a promising avenue for monetizing **LLMs** while enhancing digital marketing experiences. Our approach automates traditionally manual processes like **product tagging and link retrieval**, transforming passive content consumption into **revenue-generating opportunities**.

## Dataset
The project utilizes the **Open Images Dataset V7**, which includes:
- **1.9 million images** with **16 million bounding boxes**.
- Focus on **product-related** categories (e.g., clothing, electronics, food items).
  
## Data Preprocessing
Steps Before Training YOLOv8 : Preprocessing involved annotation conversion, normalization, and filtering for optimal model training.

### 1. Image and Class Selection
- Load and filter datasets from class description and annotation files (`CSV`).
- Ensure that there are sufficient images per class.

### 2. Image Availability Check
- Validate the existence of `.jpg` image files and corresponding annotation files.

### 3. Per-Class Sampling
- Count and sample training/validation images and bounding boxes per class.

### 4. Dataset Creation
- Organize dataset directories:
  - `train/`: Contains training images and labels.
  - `val/`: Contains validation images and labels.
- Normalize bounding box coordinates (values between `0–1`).

### 5. Data Validation
- Verify label formats, bounding box ranges, and class indices.
- Ensure minimum data requirements are met for effective training.

### 6. Configure YAML File
- Define dataset paths, class names, and dataset structure in `dataset.yaml`.
- Save the `dataset.yaml` file in the dataset directory.

### 7. Initiate Training
- Use YOLOv8.train() with the YAML file for training execution.
 
## Proposed Framework ( Methodology) 
The interactive advertising system implements a pipeline for product link retrieval, allowing users to seamlessly discover and purchase products from visual content. The process begins with receiving input media from various sources, such as **video players, social media feeds, or web browsers**. The input image is processed using a **fine-tuned YOLOv8 model**, which detects objects within the image and filters relevant **product-related classes**, eliminating unnecessary detections such as people or background elements.

In the **user interface**, when a user expresses interest in a specific product by clicking on it, the system utilizes the corresponding **bounding box coordinates** to extract a **cropped image** of the selected product. This cropped image then flows through a processing stream where **BLIP3** is used for **caption generation**. BLIP3 produces a **detailed description** of the product’s visual characteristics, capturing essential attributes such as **shape, color, texture, and distinguishing features**.

The final stage of the pipeline involves a **multimodal Large Language Model (LLM)**, which receives a structured prompt constructed from three elements: The **cropped product image**, The **BLIP3-generated caption** and a **query** asking the model to find purchase links for the detected product. The system leverages **LLaMA3 and Gemini 1.5 Pro** to perform efficient **product link retrieval**. These LLMs process the input and retrieve relevant product links from various online sources, ensuring that users receive accurate and up-to-date shopping options.

To enhance efficiency and performance, two versions of the pipeline were designed and tested. One version incorporated BLIP3 for caption generation before sending the data to the LLM, while the other version relied solely on the LLM for product link retrieval. The comparison between these versions provided insights into optimizing system performance while maintaining high accuracy in retrieving relevant product purchase links.

Framework of interactive advertising system 


<img width="438" alt="Image" src="https://github.com/user-attachments/assets/5625a935-5b34-42a8-a449-a60aaf3acfc6" />

## Performance Metrics
- **YOLOv8 Object Detection**: Achieved an **mAP@50 of 0.535** across 25 fine-tuned product categories.
- **BLIP3 Captioning**: Evaluated using **CLIPScore**, achieving an average **score of 0.2852**.
- **Product Link Retrieval**:
  - LLaMA3 & Gemini **achieved up to 97% accuracy** in retrieving relevant product links.
  - **Ablation studies** revealed that bypassing **BLIP3 improved LLM performance**, suggesting **LLMs can directly handle multimodal inputs**.
