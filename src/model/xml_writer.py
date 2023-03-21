import xml.etree.ElementTree as ET
import os

class XML_writer:
    def __init__(self, file_path, results, labels,
                 width, height, depth, target_folder):
        self.file_name = os.path.basename(file_path)
        self.folder_name = os.path.basename(os.path.dirname(file_path))
        self.file_path = file_path
        self.results = results
        self.labels = labels
        self.width = width
        self.height = height
        self.depth = depth
        self.target_folder = target_folder
        
    def check_and_make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def __call__(self):
        annotation_tag = ET.Element('annotation')
        annotation_tag.append(self.__get_folder_element())
        annotation_tag.append(self.__get_file_element())
        annotation_tag.append(self.__get_path_element())
        annotation_tag.append(self.__get_sources_element())
        annotation_tag.append(self.__get_size_element())
        annotation_tag.append(self.__get_segment_element())
        for i in range(len(self.results['labels'])):
            if self.results['scores'][i]:
                annotation_tag.append(self.__get_result_object_element(i))
        tree = ET.ElementTree(annotation_tag)
        ET.indent(tree, space="\t", level=0)
        self.check_and_make_dir(os.path.join(self.target_folder, self.folder_name))
        tree.write(os.path.join(self.target_folder, self.folder_name, self.file_name.replace('.jpg', '.xml')), encoding="utf-8")
        return 1

    def __create_element(self, element_name, text):
        element = ET.Element(element_name)
        element.text = str(text)
        return element

    def __get_folder_element(self):
        return self.__create_element('folder', self.folder_name)

    def __get_file_element(self):
        return self.__create_element('filename', self.file_name)

    def __get_path_element(self):
        return self.__create_element('path', self.file_path)

    def __get_sources_element(self):
        element = ET.Element('source')
        element.append(self.__create_element('database', 'Unknown'))
        return element

    def __get_size_element(self):
        element = ET.Element('size')
        element.append(self.__create_element('width', self.width))
        element.append(self.__create_element('height', self.height))
        element.append(self.__create_element('depth', self.depth))
        return element

    def __get_segment_element(self):
        return self.__create_element('segmented', 0)

    def __get_result_object_element(self, index):
        element = ET.Element('object')
        element.append(self.__create_element('name', self.labels[int(self.results['labels'][index])]))
        element.append(self.__create_element('pose', 'Unspecified'))
        element.append(self.__create_element('truncated', 0))
        element.append(self.__create_element('difficult', 0))
        bndbox = ET.Element('bndbox')
        bndbox.append(self.__create_element('xmin', int(self.results['boxes'][index][0])))
        bndbox.append(self.__create_element('ymin', int(self.results['boxes'][index][1])))
        bndbox.append(self.__create_element('xmax', int(self.results['boxes'][index][2])))
        bndbox.append(self.__create_element('ymax', int(self.results['boxes'][index][3])))
        element.append(bndbox)
        return element


