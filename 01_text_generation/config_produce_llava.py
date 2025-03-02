import os
import json
from classes import IMAGENET2012_CLASSES
# Base directory where the subfolders are located
base_dir = 'train'
# Text and category which is the same for all

category = "detail"

# Initialize an empty list to hold the JSON objects
json_objects = []

# Iterate through each subfolder in the base directory
question_id = 0  # Initialize the question_id

with open('output.json', 'w') as json_file:
    # Iterate through each subfolder in the base directory
    for subfolder in os.listdir(base_dir):

        text =  f"Describe the physical appearance of the {IMAGENET2012_CLASSES[subfolder]} in the image.Include details about its shape, posture, color, and any distinct features."
        subfolder_path = os.path.join(base_dir, subfolder)
        
        if os.path.isdir(subfolder_path):
            # Iterate through each image in the subfolder
            for image_file in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder, image_file)
                # Create a JSON object for each image
                json_object = {
                    "question_id": question_id,  # Unique ID for each image
                    "image": image_path,
                    "text": text,
                    "category": category
                }
                # Write the JSON object to the file, each on a new line
                json_file.write(json.dumps(json_object) + '\n')
                question_id += 1  # Increment the question_id for the next image

print(f"JSON file generated with {question_id} entries.")