from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from torchvision.transforms import ToPILImage
from PIL import Image
from torchvision import transforms as TF

def predict(model, image, label, transform, label_inverse_mapper):
    orignal_image = Image.open(image)
    width = orignal_image.width
    height = orignal_image.height
    processed_image = transform(orignal_image)
    transform_image = TF.Compose([TF.PILToTensor()])
    orignal_image = transform_image(orignal_image)
    model.eval()
    predited = model(processed_image.unsqueeze(0))
    labels = [label_inverse_mapper[each_lable] for each_lable in predited[0]['labels'].numpy().tolist()]
    predited[0]['boxes'][:, 0] = predited[0]['boxes'][:, 0] * (width/224)
    predited[0]['boxes'][:, 2] = predited[0]['boxes'][:, 2] * (width/224)
    predited[0]['boxes'][:, 1] = predited[0]['boxes'][:, 1] * (height/224)
    predited[0]['boxes'][:, 3] = predited[0]['boxes'][:, 3] * (height/224)
    img=draw_bounding_boxes(orignal_image, predited[0]['boxes'], width=3, labels= labels, fill =True, font_size=20)
    img = ToPILImage()(img)
    plt.imshow(img)
    return img.show()