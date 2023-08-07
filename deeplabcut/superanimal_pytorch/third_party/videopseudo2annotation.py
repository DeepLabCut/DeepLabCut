import json
import sys
import numpy as np
import cv2
import argparse
from collections import defaultdict
import os 
parser = argparse.ArgumentParser()
parser.add_argument('--video_result_path', type = str)
parser.add_argument('--video_path', type = str)

args = parser.parse_args()

with open(args.video_result_path, 'r') as f:    
    video_result = json.load(f)

resultname = args.video_result_path.split('/')[-1].replace('.mp4.json','')

videopath = args.video_path

root = os.path.join(os.path.dirname(args.video_result_path),f'annotation_{resultname}')

os.makedirs(root, exist_ok = True)

os.makedirs(os.path.join(root, 'images'), exist_ok = True)

os.makedirs(os.path.join(root, 'annotations'), exist_ok = True)

def video_to_frames(input_video, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    video = cv2.VideoCapture(input_video)
    # Get the frames per second (fps) of the video
    fps = int(video.get(cv2.CAP_PROP_FPS))
    # Initialize a frame counter
    frame_count = 0
    while True:
        # Read a frame from the video
        ret, frame = video.read()
        # Break the loop if we have reached the end of the video
        if not ret:
            break
        # Save the frame as an image file
        frame_file = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_file, frame)
        # Increment the frame counter
        frame_count += 1
    # Release the video object and close the window (if open)
    video.release()
    cv2.destroyAllWindows()


categories = [
    {
        "supercategory": "animal",
        "id": 0,
        "name": "individual0",
        "keypoints": [
            "nose",
            "upper_jaw",
            "lower_jaw",
            "mouth_end_right",
            "mouth_end_left",
            "right_eye",
            "right_earbase",
            "right_earend",
            "right_antler_base",
            "right_antler_end",
            "left_eye",
            "left_earbase",
            "left_earend",
            "left_antler_base",
            "left_antler_end",
            "neck_base",
            "neck_end",
            "throat_base",
            "throat_end",
            "back_base",
            "back_end",
            "back_middle",
            "tail_base",
            "tail_end",
            "front_left_thai",
            "front_left_knee",
            "front_left_paw",
            "front_right_thai",
            "front_right_knee",
            "front_right_paw",
            "back_left_paw",
            "back_left_thai",
            "back_right_thai",
            "back_left_knee",
            "back_right_knee",
            "back_right_paw",
            "belly_bottom",
            "body_middle_right",
            "body_middle_left"
        ]
    }
]

              
def result_2_train_test(img_root, result):
    annotation_id = 0
    num_kpts = 39 # only for quadruped

    images = []
    for image_id, (frameid, data) in enumerate(result.items()):

        filename = f'frame_{image_id}.jpg'
        filepath = os.path.join(img_root, filename)
        image = cv2.imread(filepath)
        height, width, channels = image.shape        
        
        temp = {'file_name': filename,
                'id': image_id,
                'width': width,
                'height': height}
        
        images.append(temp)
    
    annotations = []
    annotation_id = 0
    for image_id, (frameid, data) in enumerate(result.items()):
        for individual_data in data:            
            keypoints = np.array(individual_data['keypoints'])
            num_kpts = len(keypoints)
            keypoints[:,2] = 2
            
            bbox = individual_data['bbox']
            x,y,w,h = bbox[:4]
            area = w * h
            category_id = 0
            keypoints = keypoints.reshape(-1)
            temp = {'keypoints': list(keypoints),
                    'num_keypoints': num_kpts,
                    'bbox': [x,y,w,h],
                    'area': area,
                    'image_id': image_id,
                    'id': annotation_id,
                    'category_id': 0}
            annotations.append(temp)
            annotation_id+=1
            
    obj = {'images': images,
           'annotations': annotations,
           'categories': categories}

    return obj 

video_to_frames(videopath, os.path.join(root, 'images'))

obj = result_2_train_test(os.path.join(root, 'images'),video_result)


with open(os.path.join(root, 'annotations', 'train.json'), 'w') as f:
    json.dump(obj, f)

with open(os.path.join(root, 'annotations', 'test.json'), 'w') as f:
    json.dump(obj, f)    

