import torch
import torchvision
from torch.nn import Linear
from torchvision import models
from torchvision import transforms
from PIL import Image
from torch import nn
from torchvision.datasets import ImageNet
import torch.optim as optim

SUPPORTED_MODELS = {
    'squeezenet1_1': {'model': models.squeezenet1_1, 'classifier_attr_name': 'classifier', 'in_features': 512}}


# re-use a pre-trained model as feature extractor, and update the output classifier
def _load_pre_trained_model_and_customize(model_type, num_class=2):
    pre_trained_model = SUPPORTED_MODELS[model_type]['model'](pretrained=True)
    new_classifier = _customized_classifier(SUPPORTED_MODELS[model_type]['in_features'], num_class)
    classifier_attr_name = SUPPORTED_MODELS[model_type]['classifier_attr_name']
    return _customized_model(pre_trained_model, new_classifier, classifier_attr_name), new_classifier


def _customized_model(pre_trained_model, new_classifier, classifier_attr_name='classifier'):
    # fix the parameters of the pre trained model
    for param in pre_trained_model.parameters():
        param.requires_grad = False
    # replace the classifier
    setattr(pre_trained_model, new_classifier, classifier_attr_name)
    return pre_trained_model


def _customized_classifier(in_features, num_class):
    return Linear(in_features=in_features, out_features=num_class, bias=True)


def _get_transform():
    # TODO: configurable transformer
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    return transform


# image dataset organized in imagenet style
class KOLImages(ImageNet):
    # the folder root/train/1/11.png
    def __init__(self, root, split='train', **kwargs):
        super(KOLImages, self).__init__(root, split, download=False, **kwargs)


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available()
                        else "cpu")


def _get_data_loader(data_root, split, transform, batch_size=4, shuffle=True):
    dataset = KOLImages(root=data_root, split=split,
                        transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=2)


# train
def _train_model(model, classifier, data_loader, device):
    criterion = nn.CrossEntropyLoss()
    # only to update the parameters of the classifier
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    model.to(device)
    # put into training mode
    model.train()

    epochs = 10
    print_every = 2000  # print every 2000 mini-batches

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % print_every == print_every - 1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_every))
                running_loss = 0.0
    print('Finished Training')


# save model
def _save_model_state(model, model_path):
    pickle_module = model.state_dict()
    torch.save(pickle_module, model_path)


# load model
def _load_model_state(model, model_path):
    pickle_module = torch.load(model_path)
    model.load_state_dict(pickle_module)


# evaluation
def _test_model(model, data_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


# prediction
def _predict(model, transform, img, device):
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    input = batch_t.to(device)
    output = model(input)
    _, index = torch.max(output, 1)
    index = torch.squeeze(index, 0)
    return index.item()


def main():
    # train the model and save
    model_type = 'squeezenet1_1'
    data_root = '.'
    model_path = 'img_classifier.pth'
    transform = _get_transform()
    batch_size = 10
    train_data_loader = _get_data_loader(data_root, 'train', transform, batch_size, shuffle=True)
    num_class = len(train_data_loader.dataset.classes)
    # TODO: put classifier as a property of model
    model, classifier = _load_pre_trained_model_and_customize(model_type, num_class)
    device = _get_device()
    _train_model(model, classifier, train_data_loader, device)
    _save_model_state(model, model_path)
    # test the model
    model_re_loaded, _ = _load_pre_trained_model_and_customize(model_type, num_class)
    _load_model_state(model_re_loaded, model_path)
    test_data_loader = _get_data_loader(data_root, 'val', transform, batch_size, shuffle=False)
    _test_model(model_re_loaded, test_data_loader, device)
