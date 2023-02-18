from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToPILImage
from PIL import Image

class Predict:
    def __init__(self, model, transform):
        self.transform = transform
        self.model = model
        self.width = None
        self.height = None
        
    def load_image(self, image):
        self.orignal_image = Image.open(image)
        self.width = self.orignal_image.width
        self.height = self.orignal_image.height
        self.depth = len(self.orignal_image.getbands())
        
    def predict(self, image):
        self.load_image(image)
        processed_image = self.transform(self.orignal_image)
        self.model.eval()
        predited = self.model(processed_image.unsqueeze(0))
        predited[0]['boxes'][:, 0] = predited[0]['boxes'][:, 0] * (self.width/224)
        predited[0]['boxes'][:, 2] = predited[0]['boxes'][:, 2] * (self.width/224)
        predited[0]['boxes'][:, 1] = predited[0]['boxes'][:, 1] * (self.height/224)
        predited[0]['boxes'][:, 3] = predited[0]['boxes'][:, 3] * (self.height/224)
        return predited
        
    def plot_predicted_image(self, image, label, label_inverse_mapper):
        predited = self.predict(image)
        labels = [label_inverse_mapper[each_lable] for each_lable in predited[0]['labels'].numpy().tolist()]
        img = draw_bounding_boxes(self.orignal_image,
                                    predited[0]['boxes'],
                                    width=3,
                                    labels= labels,
                                    fill =True,
                                    font_size=20
                                 )
        img = ToPILImage()(img)
        return img.show()