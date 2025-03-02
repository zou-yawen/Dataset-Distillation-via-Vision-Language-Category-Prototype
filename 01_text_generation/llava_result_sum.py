import json
import argparse
parser = argparse.ArgumentParser(description='Train or Test ResNet-18 on an image dataset.')
parser.add_argument('--text_dir',type=str, 
                        help='text dir')
args = parser.parse_args()
# Function to read JSON objects line by line from a file
def read_json_objects(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)

# Reading JSON objects from both files
data1 = list(read_json_objects(f'{args.text_dir}/answer-file-our.jsonl'))
data2 = list(read_json_objects(f'{args.text_dir}/output.json'))

# Create a dictionary to store the image based on question_id
image_dict = {item['question_id']: item['image'] for item in data2}

# Combine the text with the corresponding image based on question_id
combined_data = {}
for item in data1:
    question_id = item['question_id']
    if question_id in image_dict:
        combined_data[image_dict[question_id]] = item['text']

# Output the combined result to a new JSON file
output_file = f'{args.text_dir}/combined_data.json'
with open(output_file, 'w', encoding='utf-8') as out_f:
    with open('rename.sh','w') as f2: 
        for key, value in combined_data.items():
            key_list = key.split('/')
            # Modify the key and value
            key1 = key_list[1].replace("_", "MM")
            key2 = f'{key_list[0]}/{key1}'
            f2.write(f"mv {key} {key2} \n")
            value = value.replace('"', '').replace("'s", "")
            
            # Create a JSON object and write it to the file in one line
            json_obj = {"file_name": key2, "text": value}
            out_f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

