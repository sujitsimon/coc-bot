from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import nn
import torch

DEFAULT_PATH = ".\\save\\basic_model.pt"

class Model(nn.Module):
    def __init__(self, out_classes):
        super(Model, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(weights=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.out_classes = out_classes
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.out_classes)

    def load_default_state(self):
        print(f"Loading Default Path: {DEFAULT_PATH}")
        self.load_state_dict(torch.load(DEFAULT_PATH))

    def forward(self, x, y):
        x = self.model(x, y)
        return x

if __name__ == "__main__":
    model = Model(3)
    save_path = "..\\playground\\save\\basic_model.pt"
    model.load_state_dict(torch.load(save_path))