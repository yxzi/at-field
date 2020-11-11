import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.attacks.evasion import FastGradientMethod, DeepFool
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist, load_cifar10

import argparse
import os
import setGPU
from time import time
import torch
import datetime

from pprint import pprint as pp

from torchvision import transforms, datasets

from utils import get_num_classes, get_architecture
from smooth import Smooth

# TODO: clean up imports


## parse args
# TODO: some args are redundent

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=300, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--nb_grads", type=int, default=10, help="nb_grads parameter for DeepFool adversarial trainer")
args = parser.parse_args()

## Step 1: load model and dataset

# load the base classifier
checkpoint = torch.load(args.base_classifier)
base_classifier = get_architecture(checkpoint["arch"])
base_classifier.load_state_dict(checkpoint['model_state_dict'])

# create the smooothed classifier g
smoothed_classifier = Smooth(base_classifier, get_num_classes(), args.sigma)

# # iterate through the dataset
dataset = datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())

y_test = np.asarray(dataset.targets)    # test labels
min_pixel_value = 0.0   # min value
max_pixel_value = 1.0   # max value

# Step 2: create an interface for classifier, load trained model, to be used by attack model trainer

# Define the loss function and the optimizer for attack model trainer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(base_classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

# Create the ART classifier

classifier = PyTorchClassifier(
    model=base_classifier,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10,
)

# Step 3: Train the ART classifier

# TODO: add option to train on demand

# Step 4: Evaluate the ART classifier on benign test examples

y_test = y_test[:args.max]  # limit the length of test set
trans = transforms.ToTensor()   # transform ndarray into tensor, normalize in the process
x_data = []     # list for test data
predictions = []    # list for prediction
toPilImage = transforms.ToPILImage()    # transform tensor into PIL image to save


for i in range(args.max):

    # x is a single image, label is a single output
    (x, label) = dataset[i]
    x = x.cuda()
    pred = smoothed_classifier.predict(x, args.N, args.alpha, args.batch)

    # output image
    if i < 5:
        x = x.cpu()
        x = x + torch.randn_like(x) * args.sigma
        pil = toPilImage(x)
        pil.save("{}/img_nn_{}_{}.png".format("./output", i, args.sigma ))

    # record prediction
    predictions.append(pred)
    x_data.append(x.unsqueeze(0).cpu())

    if i%50==0:
        print("{} of {}".format(i,args.max))

# transform data into right format
predictions = np.asarray(predictions)
x_test = torch.cat(x_data).numpy()
# pp(predictions.shape)

# test accuracy on benign examples
accuracy = np.sum(predictions == y_test) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 5: Generate adversarial test examples

# attack = FastGradientMethod(estimator=classifier, eps=0.1)
# x_test_adv = attack.generate(x=x_test)

adv_crafter = DeepFool(classifier, nb_grads=args.nb_grads)
print("Craft attack on training examples")
x_test_adv = adv_crafter.generate(x_test)

# deepfool takes ~8 second for each adversarial image

# Step 6: Evaluate the ART classifier on adversarial test examples

# pp(x_test_adv.shape)
predictions = []

for i in range(x_test_adv.shape[0]):
    x = torch.from_numpy(x_test_adv[i]).cuda()
    # predict
    pred = smoothed_classifier.predict(x, args.N, args.alpha, args.batch)
    predictions.append(pred)

    # # output image
    # if i < 5:
    #     pil = toPilImage(x.cpu())
    #     pil.save("{}/img_fool_{}.png".format("./output", i ))

    if i%50==0:
        print("{} of {}".format(i,x_test.shape[0]))


predictions = np.asarray(predictions)
# pp(predictions.shape)

accuracy = np.sum(predictions == y_test) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))