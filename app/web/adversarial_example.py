from flask import jsonify, request, json
from .blueprint import AD
from torchvision import transforms
import numpy as np
from app.web.network.mnist import *
from app.web.network.cifar import *
from advertorch.attacks import *
from torchvision import utils as vutils
import torch.backends.cudnn as cudnn
from torchvision.models import *
import os
from PIL import Image
import cv2

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


@AD.route('/upload', methods=['GET', 'POST'])
def upload():
    files = request.files.to_dict()
    img = request.files.get('file')
    basedir = os.path.abspath(os.path.dirname(__name__))
    path = basedir + "\\app\\static\\upload\\"
    imgName = img.filename
    file_path = path + imgName
    img.save(file_path)
    url = '/static/upload/' + imgName
    return jsonify({"url": url})


@AD.route('/LocalTest', methods=['POST'])
def LocalTest():
    data = request.get_data()
    json_data = json.loads(data.decode('utf-8'))
    dataset = json_data.get('dataset')
    model = json_data.get('model')
    imgUrl = json_data.get('img')
    img = processImg(dataset, imgUrl)
    data = img.unsqueeze(0).to(device)
    data.requires_grad = True
    model = loadModel(dataset, model)
    list = getResult(dataset, model, data, 10)
    return jsonify({
        "result": list
    })


@AD.route('/WhiteBox', methods=['POST'])
def WhiteBox():
    data = request.get_data()
    json_data = json.loads(data.decode('utf-8'))
    dataset = json_data.get('dataset')
    model = json_data.get('model')
    method = json_data.get('method')
    imgUrl = json_data.get('img')
    params = json_data.get('params')
    img = processImg(dataset, imgUrl)
    data = img.unsqueeze(0).to(device)
    data.requires_grad = True
    model = loadModel(dataset, model)
    init_pred = getResult(dataset, model, data, 10)
    if method == 'FGSM':
        epsilon = params['epsilon']
        attack_target = params['targeted']
        output = model(data)
        Top1_p, Top1_pred = output.max(1)
        data_grad = getGradient(data, Top1_pred, model, output, attack_target)
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
    else:
        perturbed_data = data
    imgUrl_perturbed = imgUrl.replace('.JPEG', '_AE.JPEG')
    basedir = os.path.abspath(os.path.dirname(__name__)) + "\\app"
    imgUrl_save = basedir + imgUrl_perturbed.replace('/', '\\')
    save_image_tensor(perturbed_data, imgUrl_save)
    final_pred = getResult(dataset, model, perturbed_data, 10)
    return jsonify({
        "init_pred": init_pred,
        "imgUrl_perturbed": imgUrl_perturbed,
        "final_pred": final_pred
    })


@AD.route('/BlackBox', methods=['POST'])
def BlackBox():
    data = request.get_data()
    json_data = json.loads(data.decode('utf-8'))
    dataset = json_data.get('dataset')
    model_g = json_data.get('model_g')
    model_a = json_data.get('model_a')
    method = json_data.get('method')
    imgUrl = json_data.get('img')
    return {}


def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)


def getGradient(data, target, model, output, attack_target):
    targeted = torch.ones_like(target) * attack_target

    # Calculate the loss
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = - criterion(output, targeted)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad.data

    return data_grad


class Result:
    def __init__(self, label, confidence):
        self.label = label
        self.confidence = confidence


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Return the perturbed image
    return perturbed_image


def getResult(dataset, model, data, k=5):
    output = model(data)
    Topk_p, Topk_pred = output.topk(k)
    Topk_pred = getImageNetClass(dataset, Topk_pred)
    Top5_p = [str(i) for i in Topk_p.cpu().detach().numpy()[0]]
    list = []
    for i in range(len(Top5_p)):
        list.append(Result(Topk_pred[i], Top5_p[i]).__dict__)
    return list


def getImageNetClass(dataset, labels):
    if dataset == 'ImageNet':
        basedir = os.path.abspath(os.path.dirname(__file__))
        classes = []
        label_file = basedir + '\\label\\synset_words.txt'
        for line in open(label_file, "r"):  # 设置文件对象并读取每一行文件
            tmp = line.split(',')[0]
            label = tmp[10:]
            label = label.replace('\n', '')
            classes.append(label)
    elif dataset == 'Cifar10':
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        return labels
    return [classes[i] for i in labels[0]]


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def loadModel(dataset, model):
    if dataset == 'Mnist':
        pass
    elif dataset == 'Cifar10':
        mean = torch.Tensor([0.4914, 0.4822, 0.4465])
        std = torch.Tensor([0.2023, 0.1994, 0.2010])

        pass
    elif dataset == 'ImageNet':
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        if model == 'Vgg16':
            pretrained_model = vgg16(pretrained=True)
        elif model == 'ResNet50':
            pretrained_model = resnet50(pretrained=True)
        elif model == 'MobileNetV2':
            pretrained_model = googlenet(pretrained=True)
        elif model == 'GoogLeNet':
            pretrained_model = vgg16(pretrained=True)
        elif model == 'DenseNet161':
            pretrained_model = densenet161(pretrained=True)
        else:
            return 'model not find'
        model = pretrained_model.to(device)
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model.eval()
        mean, std = mean.to(device), std.to(device)
        model = torch.nn.Sequential(Normalize(mean, std), model)
        return model


def processImg(dataset, imgUrl):
    basedir = os.path.abspath(os.path.dirname(__name__)) + "\\app"
    imgUrl = basedir + imgUrl.replace('/', '\\')
    with open(imgUrl, 'rb') as f:
        img = Image.open(imgUrl)
    if dataset == 'Mnist':
        pass
    elif dataset == 'Cifar10':
        transform_val = transforms.Compose([
            transforms.ToTensor(),
        ])
        img = transform_val(img)

    elif dataset == 'ImageNet':
        image_size = 224
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        img = transform_val(img)
    return img


def transpose(img):
    return np.transpose(img, (1, 2, 0))


def unnormalize(img, t_std, t_mean):
    unnorm_img = torch.Tensor(img).unsqueeze(0) * t_std + t_mean
    return unnorm_img.squeeze(0)


def Img2Cpu(img):
    return img.squeeze(0).cpu().detach().numpy()
