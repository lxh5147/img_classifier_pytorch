import json
import os
from collections import namedtuple

import torch
import torch.optim as optim
from torch import nn
from torchvision import models
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder


def _fix_random_seed():
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(0)


def _customized_classifier_squeezenet1_1(num_class):
    return nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                         nn.ReLU(),
                         nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1)),
                         nn.ReLU(inplace=True),
                         nn.AdaptiveAvgPool2d(output_size=(1, 1)))


SUPPORTED_MODELS = {
    'squeezenet1_1': {'model': models.squeezenet1_1,
                      'classifier_attr_name': 'classifier',
                      'classifier_builder': _customized_classifier_squeezenet1_1,
                      }}


def _customized_classifier(builder, num_class):
    return builder(num_class)


# re-use a pre-trained model as feature extractor, and update the output classifier
def _load_pre_trained_model_and_customize(model_type, num_class):
    pre_trained_model = SUPPORTED_MODELS[model_type]['model'](pretrained=True)
    new_classifier = _customized_classifier(SUPPORTED_MODELS[model_type]['classifier_builder'], num_class)
    classifier_attr_name = SUPPORTED_MODELS[model_type]['classifier_attr_name']
    return _customized_model(pre_trained_model, new_classifier, classifier_attr_name)


def _customized_model(pre_trained_model, new_classifier, classifier_attr_name):
    # fix the parameters of the pre trained model
    for param in pre_trained_model.parameters():
        param.requires_grad = False
    # replace the classifier
    setattr(pre_trained_model, classifier_attr_name, new_classifier, )
    # add the 'classifier' attribute if not existed
    if not hasattr(pre_trained_model, 'classifier'):
        setattr(pre_trained_model, 'classifier', new_classifier)
    return pre_trained_model


def _get_transform():
    # TODO: to make transform configurable
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    return transform


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available()
                        else "cpu")


def _get_data_loader(data_root, split, transform, batch_size, shuffle):
    # root/split/class_1/img1.png
    dataset = ImageFolder(os.path.join(data_root, split),
                          transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=2)


def _from_id_to_label(ids, classes):
    # ids: each prediction has k ids
    return [[classes[id] for id in id_list] for id_list in ids]


_config = {
    'data_root': './data',
    'model_type': 'squeezenet1_1',
    'model_path': 'img_classifier.pth',
    'class_file_path': 'img_classes.txt',
    'train_shuffle_data': True,
    'train_epochs': 3,
    'train_print_every': 1,
    'train_lr': 0.001,
    'batch_size': 2,
    # how many labels to predict, default 1
    'prediction_top_k': 1,
    'prediction_prob_threshold': .2,
}

CONFIG = namedtuple("CONFIG", _config.keys())(*_config.values())


# train
def _train_model(model, data_loader, device, config):
    criterion = nn.CrossEntropyLoss()
    # only to update the parameters of the classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=config.train_lr)
    model.to(device)
    # put into training mode
    model.train()

    epochs = config.train_epochs
    print_every = config.train_print_every  # print every 2000 mini-batches

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


# save labels
def _save_class_labels(classes, file_path):
    with open(file_path, 'w') as f:
        json.dump(classes, f)


def _load_class_labels(file_path):
    with  open(file_path) as f:
        classes = json.load(f)
    return classes


def _save(model, model_path, classes, class_file_path):
    _save_model_state(model, model_path)
    _save_class_labels(classes, class_file_path)


def _load(model, model_path, class_file_path):
    _load_model_state(model, model_path)
    # a list of class labels
    classes_loaded = _load_class_labels(class_file_path)
    return classes_loaded


# evaluation
def _test_model(model, data_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            inputs, labels = images.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d test images: %d %%' % (
        total, 100 * correct / total))


# prediction
def _predict_batch(model, data_loader, device, top_k=1):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in data_loader:
            input = images.to(device)
            output = model(input)
            output = torch.nn.functional.softmax(output, dim=-1)
            prob, index = torch.topk(output, top_k)
            for pred in index.cpu().numpy():
                predictions.append(pred)
    return predictions, prob.cpu().numpy()


def _predict(model, transform, imgs, device, top_k=1):
    # imgs: a list of a img
    imgs_t = transform(imgs)
    input = imgs_t.to(device)
    output = model(input)
    output = torch.nn.functional.softmax(output, dim=-1)
    prob, index = torch.topk(output, top_k)
    # the index and prob of the top k predicted label
    return index.cpu().numpy(), prob.cpu().numpy()


# remove predictions according to prob threshold
def _remove_predictions_with_low_prob(predictions, probs, threshold):
    # may return no prediction (an empty list of prediction) if all predictions have a low prediction probability
    filtered = [list(filter(lambda i: i[1] >= threshold, zip(id_list, prob_list))) for id_list, prob_list in
                zip(predictions, probs)]
    filtered_predictions = [[i[0] for i in filtered_prediction] for filtered_prediction in filtered]
    filtered_probs = [[i[1] for i in filtered_prediction] for filtered_prediction in filtered]
    return filtered_predictions, filtered_probs


def main(config):
    # ensure repeatable results
    _fix_random_seed()
    transform = _get_transform()
    train_data_loader = _get_data_loader(config.data_root, 'train', transform, config.batch_size,
                                         shuffle=config.train_shuffle_data)
    classes = train_data_loader.dataset.classes
    num_class = len(classes)
    model = _load_pre_trained_model_and_customize(config.model_type, num_class)
    device = _get_device()
    _train_model(model, train_data_loader, device, config)
    _save(model, config.model_path, classes, config.class_file_path)
    # test the model, with reloadel and class labels
    model_re_loaded = _load_pre_trained_model_and_customize(config.model_type, num_class)
    classes_re_loaded = _load(model_re_loaded, config.model_path, config.class_file_path)
    test_data_loader = _get_data_loader(config.data_root, 'val', transform, config.batch_size, shuffle=False)
    _test_model(model_re_loaded, test_data_loader, device)
    # prediction
    top_k = CONFIG.prediction_top_k
    prob_threshold = CONFIG.prediction_prob_threshold
    predict_data_loader = _get_data_loader(config.data_root, 'pred', transform, config.batch_size,
                                           shuffle=False)
    # for each image, predict top k labels, and filter
    predictions, probs = _predict_batch(model_re_loaded, predict_data_loader, device, top_k)
    predictions, probs = _remove_predictions_with_low_prob(predictions, probs, prob_threshold)
    # translate the index into labels
    labels = _from_id_to_label(predictions, classes_re_loaded)
    print('predictions: ' + str(labels) + ' with probs: ' + str(probs))


if __name__ == "__main__":
    main(CONFIG)
