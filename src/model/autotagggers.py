import os
from predict import Predict
from ImageDataSet import ImageDataSet
from xml_writer import XML_writer
from model import Model


class Autotaggers:
    def __init__(self, root_folder, target_folder, model, transforms, labels):
        self.root_folder = root_folder
        self.target_folder = target_folder
        self.predictor = Predict(model, transforms)
        self.labels = labels
        
    
    def run(self):
        for root, _folder, files in os.walk(self.root_folder):
            for file in files:
                print(f"Generating for file {os.path.join(root, file)}")
                if not(".jpg" in file):
                    continue
                results = self.predictor.predict(os.path.join(root, file))[0]
                response = XML_writer(os.path.realpath(os.path.join(root, file)),
                                results,
                                self.labels,
                                self.predictor.width,
                                self.predictor.height,
                                self.predictor.depth,
                                self.target_folder
                                )()

if __name__ == "__main__":
    model = Model(3)
    save_path = "..\\playground\\save\\basic_model.pt"
    model.load_custom_to_model(save_path)
    image_data_set = ImageDataSet(None)
    transforms = image_data_set.get_default_transforms()
    labels = image_data_set.get_labels_decoding()
    root_folder = "..\\base_downloader\\layouts\\th6"
    target_folder = ".\\autotagged\\"
    auto_taggers = Autotaggers(root_folder, target_folder, model, transforms, labels)
    auto_taggers.run()
    
    
    
        