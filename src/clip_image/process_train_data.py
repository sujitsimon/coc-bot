import json
import os
import xml.etree.ElementTree as ET

image_path_prefix = ""

# def process_object(annotation_object, path_prefix = image_path_prefix):
#     array = []
#     image_name = os.path.join(path_prefix, annotation_object['image'])
#     for each_annotation in annotation_object['annotations']:
#         string = "{}, {}, {}, {},\
#                   {}, {}\n".format(image_name,
#                             each_annotation['label'].strip(),
#                             each_annotation['coordinates']['x'],
#                             each_annotation['coordinates']['y'],
#                             each_annotation['coordinates']['width'],
#                             each_annotation['coordinates']['height'])
#         array.append(string)
#     return array

def process_object(annotation_object):
    array = []
    image_name = annotation_object.find('path').text
    image_width = annotation_object.find('size//width').text
    image_height = annotation_object.find('size//height').text
    for each_annotation in annotation_object.iter('object'):
        string = "{}, {}, {}, {},\
                  {}, {}, {}, {}\n".format(image_name,
                            image_width,
                            image_height,
                            each_annotation.find('name').text,
                            each_annotation.find('bndbox//xmin').text,
                            each_annotation.find('bndbox//ymin').text,
                            each_annotation.find('bndbox//xmax').text,
                            each_annotation.find('bndbox//ymax').text)     
        array.append(string)
    return array

master_record = ["image_name, width, height, label, x_min, y_min, x_max, y_max\n"]
for root_folder, folders, files in os.walk(".\\train_data\\"):
    for file in files:
        if file.endswith('.xml'):
            tree = ET.parse(os.path.abspath(os.path.join(root_folder, file)))
            root = tree.getroot()
            master_record.extend(process_object(root))

with open(".\\train_data\\Dataset.csv", "w") as fptr:
    fptr.writelines(master_record)
