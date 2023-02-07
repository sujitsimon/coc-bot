import os
import xml.etree.ElementTree as ET
from PIL import Image

PATH = '.\\train_data'
CLIP_PATH = '.\\clipped_img'
FILES = []

OBJECTS = {}

def process_object(object_tag):
    temp_object = {}
    name = None
    for each_child in object_tag:
        temp_object[each_child.tag] = each_child.text
        if each_child.tag == 'name':
            name = each_child.text
        if each_child.tag == 'bndbox':
            temp_object[each_child.tag] = {}
            for internal_child in each_child:
                temp_object[each_child.tag][internal_child.tag] = internal_child.text
    return (name, temp_object)

for home, dir_, files in os.walk(PATH):
    for each_file in files:
        if '.xml' in each_file and os.path.join(home, each_file) not in FILES:
#            FILES.append(os.path.join(home, each_file))
            tree = ET.parse(os.path.join(home, each_file))
            root = tree.getroot()
            screen_shot_path = None
            for child in root:
                if child.tag == 'path':
                    screen_shot_path = child.text
                elif child.tag == 'object':
                    name, temp_object = process_object(child)
                    temp_object['path'] = screen_shot_path
                    try:
                        OBJECTS[name].append(temp_object)
                    except:
                        OBJECTS[name] = []
                        OBJECTS[name].append(temp_object)
                        
for each_objects in OBJECTS:
    for index, value in enumerate(OBJECTS[each_objects]):
        image = Image.open(value['path'])
        xmin = int(value['bndbox']['xmin'])
        xmax = int(value['bndbox']['xmax'])
        ymin = int(value['bndbox']['ymin'])
        ymax = int(value['bndbox']['ymax'])
        image_new = image.crop((xmin, ymin, xmax, ymax))
        file_format = '{}\\{}_{}.png'.format(CLIP_PATH, each_objects, index)
        image_new.save(file_format)
    