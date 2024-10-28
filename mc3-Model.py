#!/usr/bin/env python
# coding: utf-8

# In[11]:


# get_ipython().run_line_magic('pip', 'install opencv-python')
# get_ipython().run_line_magic('pip', 'install numpy')
# get_ipython().run_line_magic('pip', 'install torch torchvision')
# get_ipython().run_line_magic('pip', 'install wget')
# get_ipython().run_line_magic('pip', 'install pytorchvideo')
# get_ipython().run_line_magic('pip', 'install json')


# In[12]:


import cv2
import torch
import numpy as np
import time
import json

import torchvision.transforms as transforms
from torchvision.models.video import mc3_18, MC3_18_Weights


# IMPORT DETECTION PACKAGE
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


# In[13]:


# DOWNLOAD THE MODEL

# Device on which to run the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Step 1: Initialize the model with the best available weights
weights = MC3_18_Weights.DEFAULT
model_mc3_18 = mc3_18(weights=weights)
model_mc3_18.eval()
model_mc3_18 = model_mc3_18.to(device)

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()


# In[14]:


# DOWNLOAD THE KINETIC-400 LABEL

# url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
# wget.download(url, 'kinetics_classnames.json')

with open("kinetics_classnames.json", "r") as f:
    kinetics_classnames = json.load(f)
    labels = [line.strip() for line in kinetics_classnames]

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

print(labels[0:10], np.shape(labels))


# In[15]:


# COMBINE THE DETECTION AND RECOGNITION

# Step 1: Initialize model with the best available weights
weights_detect = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
model_detection = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights_detect, box_score_thresh=0.9)
model_detection.eval()

# Step 2: Initialize the inference transforms
preprocess_detect = weights_detect.transforms()


# In[16]:


# DETECTION FUNCTION
def detect(img):
    # Convert OpenCV BGR frame to PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Convert PIL Image to tensor and add batch dimension
    img_tensor = pil_to_tensor(pil_img).unsqueeze(0)

    # Apply inference preprocessing transforms
    batch = [preprocess_detect(img_tensor[0])]

    # Step 5: Perform object detection
    with torch.no_grad():
        prediction = model_detection(batch)[0]

    # Extract labels and draw bounding boxes on the frame
    boxes = prediction["boxes"]
    box = draw_bounding_boxes(img_tensor[0], boxes=boxes, colors="red", width=2)

    # Convert tensor back to OpenCV format for display
    result_frame = cv2.cvtColor(np.array(to_pil_image(box.detach())), cv2.COLOR_RGB2BGR)
    return result_frame


# In[17]:


def recognize(video_tensor):
    video_tensor = video_tensor.to(device)

    with torch.no_grad():
        prediction = model_mc3_18(video_tensor).squeeze(0).softmax(0)
    label = prediction.argmax().item()
    score = prediction[label].item()
    category_name = weights.meta["categories"][label]
    print(f"{category_name}: {100 * score:.2f}%")
    
    return category_name, score


# In[18]:


# MAIN PROCESSING FUNCTION
def process_video(use_webcam=False, video_path=None):

    # Initialize webcam or video file
    if use_webcam:
        capture = cv2.VideoCapture(0)
    elif video_path:
        capture = cv2.VideoCapture(video_path)
    else:
        print("Error: Provide video path or function for using webcam")
        return
    if not capture.isOpened():
        print("Error: Could not open video source.")
        return
    
    # Step 4: Initialize a frame buffer to collect frames for prediction
    frame_buffer = []
    num_frames_to_process = 4  # Number of frames needed for action recognition
    prev_time = 0
    start_time = time.time()
    frame_count = 0
    input_size = (480, 480)
    
    try:
        while True:
            frames = []
            for _ in range(num_frames_to_process):
                ret, img = capture.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break
                    
                frame_count += 1
                
                resized_frame = cv2.resize(img, input_size)
                
                detect_img = detect(resized_frame)
                print(f"Original shape of video: {detect_img.shape}") 
                
                frames.append(cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB))

            if len(frames) < num_frames_to_process:
                break

            video_tensor = torch.tensor(np.array(frames)).permute(0, 3, 1, 2).float() / 255.0  # Normalize values between 0 and 1

            # Apply the transform to normalize the input
            video_tensor = preprocess(video_tensor).unsqueeze(0)  # Add batch dimension
            print(f"After of inputs: {video_tensor.shape}")
            
            # Move tensor to the device
            (category_name, score) = recognize(video_tensor)

            cv2.putText(detect_img, f"{category_name}: {100 * score:.2f}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 153, 0), 2)
            
            current_time = time.time()
            total_time = current_time - start_time  # Total elapsed time since the start

            # Calculate FPS (instantaneous and average)
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            average_fps = frame_count / total_time if total_time > 0 else 0
            prev_time = current_time

            cv2.putText(detect_img, f"FPS: {float(fps):.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(detect_img, f"Average FPS: {float(average_fps):.2f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show the video stream with predictions
            cv2.imshow('MC3 Action Recognition', detect_img)

            # Press 'Esc' to exit
            if cv2.waitKey(30) & 0xFF == 27:
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()


# In[19]:


# RUNNING THE MODEL WITH OR WITHOUT WEBCAME
# For webcam:
# process_video(use_webcam=True)

# For video file:
process_video(video_path="C:/Users/nyok/Desktop/OpenCV/Videos/eatinglive.MOV")


# In[11]:





# In[11]:


import torch
print(torch.cuda.is_available())


# In[ ]:




