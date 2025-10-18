import torch # Imports the PyTorch library, which is essential for running the YOLOv5 model.

# Load pretrained YOLOv5 model, this is alrady pretarined and creates the model, the training loop and tsting loop for us, all we haev to do it input the data and choose a pretrained ID. 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # Downloads and loads the pre-trained 'yolov5s' (small) model from the official Ultralytics repository on PyTorch Hub.

def has_traffic_light(image_path, conf_thres=0.25): # Defines a function that checks for a traffic light, accepting an image path (or URL) and a confidence threshold.
    results = model(image_path) # Runs the object detection model on the input image, performing the actual inference.
    # YOLOv5 returns results.pred[0] as tensor [x1, y1, x2, y2, conf, class]
    # COCO class 'traffic light' is id 9 in YOLOv5 (indexing from 0)

    traffic_light_class = 9 # Assigns the COCO dataset class ID for 'traffic light' (which is 9) to a variable.
    detections = results.pred[0] # Extracts the tensor containing all detected objects (bounding boxes, confidence scores, and class IDs) from the results for the first image.
    
    for *box, conf, cls in detections: # Starts a loop to iterate through every detection found, unpacking the box coordinates, conf score and class ID
        if int(cls) == traffic_light_class and conf > conf_thres: # Checks two conditions, if the detected class ID matches the trafficlight ID (9) AND if the detection's confi score is above the required threshold (0.25).
            return True 
    return False 

# Test
img_path = 'any local path to an image then/filename.jpg' # 
if has_traffic_light(img_path): 
    print("Traffic light is present!") 
else: 
    print("No traffic light found.") 
