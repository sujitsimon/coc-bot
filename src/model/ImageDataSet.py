import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ImageDataSet(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms
        self.image_names = self.data.image_name.unique()

    def __getitem__(self, idx):
        img_path = self.image_names[idx]
        img = Image.open(img_path)
        num_objs = self.data[self.data['image_name'] == img_path].shape[0]
        bbox = []
        labels = []
        for i in range(num_objs):
            x_min = self.data[self.data['image_name'] == img_path]['x_min'].iloc[i]
            x_max = self.data[self.data['image_name'] == img_path]['x_max'].iloc[i]
            y_min = self.data[self.data['image_name'] == img_path]['y_min'].iloc[i]
            y_max = self.data[self.data['image_name'] == img_path]['y_max'].iloc[i]
            target = self.data[self.data['image_name'] == img_path]['label'].iloc[i]
            bbox.append([x_min, y_min, x_max, y_max])
            labels.append(target)
        bbox = torch.as_tensor(bbox, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)
            bbox[:, 0] = bbox[:, 0] * (224/self.data[self.data['image_name'] == img_path]['width'].iloc[0])
            bbox[:, 2] = bbox[:, 2] * (224/self.data[self.data['image_name'] == img_path]['width'].iloc[0])
            bbox[:, 1] = bbox[:, 1] * (224/self.data[self.data['image_name'] == img_path]['height'].iloc[0])
            bbox[:, 3] = bbox[:, 3] * (224/self.data[self.data['image_name'] == img_path]['height'].iloc[0])

        target = {}
        target["boxes"] = bbox
        target["labels"] = labels

        return img, target

    def __len__(self):
        return len(self.image_names)
