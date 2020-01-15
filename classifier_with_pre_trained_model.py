import torch

from torchvision import models
from torchvision import transforms
from PIL import Image

# import matplotlib.pyplot as plt

# load the model
model = models.squeezenet1_1(pretrained=True)
model.eval()

print(model)

transform = transforms.Compose([  # [1]
    transforms.Resize(256),  # [2]
    transforms.CenterCrop(224),  # [3]
    transforms.ToTensor(),  # [4]
    transforms.Normalize(  # [5]
        mean=[0.485, 0.456, 0.406],  # [6]
        std=[0.229, 0.224, 0.225]  # [7]
    )])

# show one img
img = Image.open("dog.jpg")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# show img
# grid_img = torchvision.utils.make_grid(batch_t, nrow=5)
# plt.imshow(grid_img.permute(1, 2, 0))

# run the prediction
out = model(batch_t)
print(out.shape)
_, index = torch.max(out, 1)
print(index)  # index of the label

# TODO: transfer learing on our own dataset , refer to https://github.com/pytorch/examples/blob/master/imagenet/main.py
# with open('imagenet_classes.txt') as f:
#  classes = [line.strip() for line in f.readlines()]
