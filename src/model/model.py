from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import nn
import torch

class Model(nn.Module):
    def __init__(self, out_classes):
        super(Model, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(weights=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.out_classes = out_classes
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.out_classes)

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    model = Model(3)
    save_path = "..\\playground\\save\\basic_model.pt"
    model.load_state_dict(torch.load(save_path))