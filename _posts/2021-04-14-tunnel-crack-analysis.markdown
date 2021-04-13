---
layout: post
title:  "Tunnel Crack Analysis: A Deep Learning System"
date:   2021-04-14 01:04:00 +0200
categories: DeepLearning
---
## Applying high-speed image recognition for practical use-cases

##### _By Michiel Firlefyn (M.V.M.Firlefyn@student.tudelft.nl; 4558774) and Matthijs van Wijngaarden (M.C.Vanwijngaarden@student.tudelft.nl; 4271785)_

The emergence of deep learning in real-world applications outside academia continues to grow. It's not a surprise, as it has been shown that these artificial intelligence methods can increase the efficiency, speed or accuracy of a system. The application of the technology in numerous fields is an indicator that AI is starting to mature despite continuous developments. Because of this, it is important to take a step back and take a look at projects described by researchers. Neural networks, and more specifically, the conclusions or results produced by the technology is notorious for its inexplainability, or 'black-box' behavior. The question whether conclusions are reproducable is vital, especially with applications relating to health or human safety. Problems arisen by malfunctioning systems or unethical decision-making are hot-topics in the field of AI. The ability to reproduce a system therefore does not only further cement the trust of it functioning as described in the paper, but also gives an insight into its mechanisms. 

---
### The tunnel crack analysis system


In this blog we attempt to reproduce a deep learning system presented by a group of researchers focussing on detecting cracks in tunnels. In 2019, Song and others published the paper 'Real-Time Tunnel Crack Analysis System via Deep Learning' [1]. A description of their system is given in detail, as it goes in-depth on the development of its image acquisition system, the vehicle-control system, and the crack identification and management system. The first two are physical components of their system, showcasing the researcher's goals to press this further than just research. The third part, however, is where deep learning plays a vital role.

The overview of the system framework containing the image acquisition system, the vehicle-control system and the crack identification and management system  	|           
:-------------------------:			|
![Figure1](/assets/img/Figure1.png)             |

Before we dive deeper into the deep learning mechanisms, it might help to look at the problem statement. Identifying cracks in tunnels is something currently done by humans with the naked eye. Logically, this is a slow and costly process. The authors describe in their paper how these current methods to identify cracks in tunnels are inefficient, slow and manpower-heavy. The goal of their research is to develop a system that can rapidly take high-quality photos as it drives through a tunnel and use artificial intelligence to segmentate and identify cracks in the walls.

Due to the detail of this requiring a fast-paced environment, the implementation of the convolutional neural network used to idenfity and segmentate these images are split in two parts: the training part (no high speed required) and the inference part (faster system).

The segmentation process contains training process and inference process    |
:-------------------------:			|
![Figure5](/assets/img/Figure5.png) |

In the figure above, the researchers divided their system up in two parts: training and inference. A important distinction, for example, is their choice of encoders during each phase: ResNet18 was chosen for training, and a combination of MobileNet-v1 [2] and CrossNet was used during inference. The decoder they used was ASPP [3] in both cases. MobileNet-v1 is a lightweight model that outperforms  many networks with a lot of computations, hence its use in the inference phase.

The researchers had access to a database of photos of cracks in tunnels, which were then annotated one by one by experts to denote the line of a crack, or lack thereof. This included annotations of other important, similar-looking distinctions such as structural seams and water-stained edges. Preprocessing of images by Song et al., was therefore not only cropping, but also the hand-drawn crack annotation by experts.

The original image (left) and the image with crack, structural seam, water stain and scratch annotations (right) |
:-------------------------:			|
![Figure7](/assets/img/Figure7.png) |

Unfortunately, a kind email towards the researchers that requested access to their image database, including its annotations, was left unanswered. This lead us to our first hurdle: being unable to reproduce the researcher's results due to needing to use a profoundly different dataset. On top of this, the source code of the paper was not made public, meaning the implementation of their convolutional network had to be done based off of their description of which type of encoder/decoder they used. Hyperparameters such as layer depth were available, but others like the learning rate not. This left us at a task that made it substantially harder to successfully reproduce their results. Or, at least, the results produced based off of un-augmented datasets.

---
### Reproducible results
Results of the original experiments in normal and augmented datasets |
:-------------------------:			|
![Table1](/assets/img/Table1.png)   |

The table shows the different results the authors obtained. Subsequent tables and figures were produced from the augmented set. This left us at the task of attempting to reproduce the first row, and to see how far we would get with the figures.

The final segmentation results |
:-------------------------:			|
![Figure11](/assets/img/Figure11.png)   |

The first column in the image shows the original photo, the second column the annotations done by hand, while the third is the predicted image. The rest of this blog will describe our attempts at reproducing this table and right-most column in the figure.

---

## Attempting to Reproduce The Tunnel Crack Analysis Implementation

The reproduced autoencoder is based off of the [MobileNet\_v2 encoder](https://arxiv.org/abs/1801.04381) (with pretrained imagenet weights) and a  [DeepLab\_v3 architecture (decoder)](https://arxiv.org/abs/1706.05587). This is very similar to the autoencoder that was used for the inference of the crack images, although the original autoencoder made use of a [MobileNet\_v1 encoder](https://arxiv.org/abs/1704.04861). We assume that the two subsequent versions of the encoders are sufficiently similar to obtain an accurate representation of the original autoencoder. 

### Dependencies
As we are using a python development environment to run our code, we would like to run you through the dependencies that are set up on our system. This blog does not aim to give you a thorough understanding on how to set it up. For that we recommend something like a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment) or a [virtual environment](https://docs.python.org/3/tutorial/venv.html). Dependencies can be installed by using a package manager of your choice such as [pip](https://pypi.org/project/pip/). 

The dependencies that we used for the two scripts are the following:
* movingData.py:
	* [argparse](https://docs.python.org/3/library/argparse.html): 
	A module that provides us with a user-friendly command-line interface. We 	 can use it to easily specify different folders for the data sorter.
	* [shutil](https://docs.python.org/3/library/shutil.html):
	This module contains functions that perform high level file operations.
	Data is moved by specifying a source and destination folder.	    
	* [os](https://docs.python.org/3/library/os.html):
	With the os module we can access certain system functionalities.
	We use this module to access the directory strings and compare the 
	suffixes in the original CRACK data to divide it in crack images and 
	ground truth annotations.
* autoEncoder.py:
	* [numpy](https://numpy.org/): 
	We use the numpy module for our matrix computations.
	* [cv2](https://pypi.org/project/opencv-python/):
	The open computer vision module contains a multitude of image 
	manipulations. We just use it to read the image data and convert the
	color space to RGB.
	* [matplotlib](https://matplotlib.org/):
	This module is used to display the output. We could have used opencv for
	this purpose as well, but matplotlib tends to be more stable across
	different system platforms.
	* [torch](https://pytorch.org/):
	The PyTorch module provides an open source machine learning framework. It	 contains a variety of functions and pre-build neural networks in 
	Torch hub. The decoder and encoder are taken from Torch hub.
	* [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch):
	This module contains a convenient high level interface to train neural 
	networks from Torch hub. The whole autoEncoder script is based on an 
[example](https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb) from this helper module.

### Data Set
Since the original dataset was unavailable, we used the CRACK500: a dataset that consists of cracks in concrete surfaces. This data set was supplied to us by the staff of [TUDelft's EEMCS faculty](https://www.tudelft.nl/en/eemcs). The data directory contained cropped images as well as full sized ones, only the directories with the full sized images are considered from hereon.

An example image of the CRACK500 data set and its ground truth:

CRACK500 image             |  CRACK500 ground truth
:-------------------------:|:-------------------------:
![CRACK500_example](/assets/img/CRACK500_example.jpg)  |  ![CRACK500_gt](/assets/img/CRACK500_gt.png)

A problem faced when running a toy example of the neural network was that a part of the validation set broke our code. It turned out that the network's input image pixel dimension needed to be a multiple of two to the power of the network's depth, i.e. 32. All images were sized 640x360, while 17 images at the end of the validation dataset were sized 648x484. Cropping these images to 640x360 while maintaining the crack ground truth fixed this problem.

### Structuring the Data Using the 'movingData.py' Script
CRACK500 uses 3 data directories: training, validation and test data. These
directories contain the data images (with a .jpg suffix) as well as the ground 
truths (with a .png suffix). We iterated over this data and put them in their
own respective directories (e.g. 'train' and 'trainannot') to obtain 6 data 
directories as to make the data more conform with the data structure from the [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) example ([CamVid](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)).

movingData.py 				|
:-------------------------:	|
```						
# import necessary modules
import argparse
import shutil
import os

# init arg parser and define arguments (args) for source and destination folder
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required = True, help = "Path to source folder")
ap.add_argument("-d", "--destination", required = True, help = "Path to destination folder")

# turn args in variables that can be called in the script
args = vars(ap.parse_args())
source = args["source"]
dest = args["destination"]

# returns a list with all the names in the specified directory
files = os.listdir(source)

# iterate over the list of names and move all the files with a .jpg suffix
# ".jpg" suffix is replaced with ".png" to grab ground truth instead of images
for f in files:
	if os.path.splitext(f)[1] in (".jpg"):		
		shutil.move(source + f, dest)
```

By running this script 3 times, once for every original data directory, we obtain the 6 desired directories.

### Deep Learning With PyTorch and the 'autoEncoder.py' Script
The autoEncoder script contains all the necessary functions and classes to load the 6 data directories, initialize the appropriate PyTorch dataloaders, extract the neural architectures from Torch hub, train, and validate the network. This code is our main contribution to the reproducibility effort. It is based on the PyTorch [example](https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb) as discussed before. For the reader's convenience, this script is split up in several code blocks to discuss the block's functions accordingly.

autoEncoder.py 			 	|
:-------------------------:	|
Importing the necessary modules in the python script, their description was given before in the Dependencies section. We are pointing to the `./data` directory in our current working directory to load in the data. This directory should contain the 6 CRACK500 data directories. Separate variables are specified for each directory such that these can be called by their respective dataloaders.

```
import os
os. environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt

DATA_DIR = './data/'


x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')
```
The `visualize()` function is a helper function that can be called every time we want to visualize a sample from a dataset that we have created. The images are to be plotted in a row and the amount of images to be plotted is inferred from the `**images` argument that will correspond to a PyTorch `BaseDataset` instance as you will see in the next code block.

```
def visualize(**images):
	n = len(images)
	plt.figure(figsize = (16,5))

	for i, (name,image) in enumerate(images.items()):		
		plt.subplot(1, n, i + 1)		
		plt.xticks([])
		plt.yticks([])
		plt.title(' '.join(name.split('_')).title())
		plt.imshow(image)

	plt.show()
```
We define a class that inherits the properties from a PyTorch `BaseDataset`. This class contains a function that initializes the dataset parameters. Optional arguments for the classes, augmentation and preprocessing are specified. The data along with the ground truth are also initialized and stored in lists. Thus, the `__init__()` function can be used to set the mandatory and optional input to the dataset, whereas the `__getitem__()` function will take care of returning an image sample and its ground truth. Since the `cv2` dependency is used to read the image data from the list, images are to be converted from the native read type 'BGR' to 'RGB' (color space that autoencoder will expect). 

Finally, a sample is visualized of a `Dataset` object without augmentation or preprocessing:

dataset image and ground truth  	|           
:-------------------------:			|
![outputDataset](/assets/img/outputDataset.png)	|

autoEncoder.py - continued	|
:-------------------------:	|

```
# helper class for data extraction, transformation and preprocessing
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    """
    CRACK500 Dataset. Read images, aplly augmentation and preprocessing transformations
    
    Args:
        images_dir (str): path to image folder
        masks_dir (str): path to segmentation masks folder (ground truth)  
        class values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transformation pipeline (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing (e.g. normalization, shape manipulation, etc.) 
    """

    CLASSES = ['crack']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None):

        self.id_names = [os.path.splitext(f)[0] for f in os.listdir(images_dir)]
        self.image_ids = [s + '.jpg' for s in self.id_names]
        self.mask_ids = [s + '.png' for s in self.id_names]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.image_ids)

# let's look at the data that we have
dataset = Dataset(x_train_dir, y_train_dir, classes=['crack'])

image, mask = dataset[0]    # get some sample

print("Results normal dataset")
print(f"image shape = {image.shape}")
print(f"mask shape = {mask.shape}")

visualize(
    image=image,
    mask=mask
    )
```
Augmentations of the dataset give us two main benefits: 'bigger' dataset and regularization. The CRACK500 training set is rather small (about 1800 samples), thus augmenting the samples randomly every time they are loaded means that effectively we have 9 times more training data since we can apply 9 random transforms. As explained in the data set section, the validation images had some non-desired dimensions, the `get_validation_augmentation()` function is just a double check. If any image dimension is encountered that does not match well with the network's input, we will just pad it with zeros. The preprocessing step is necessary to normalize and convert the data samples to tensors, note the `transpose(2,0,1)` function to format the input (width, height, channels) in a way PyTorch expects it (channels, width, height). These tensors are then ran through our network. Unfortunately, no significant hardware acceleration was at our disposal. The training of the network was performed on an Intel Core i7-7500U processor (cpu) with 16GB DDR3 RAM memory. It took 26 hours.

Three augmented samples from the augmented_dataset are visualized in the code, one is shown here:

augmented dataset image and ground truth  	|           
:-------------------------:					|
![outputDataset_aug](/assets/img/outputDataset_aug1.png)|

autoEncoder.py - continued	|
:-------------------------:	|
```
import albumentations as albu

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1)
            ],
            p=0.9
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3,p=1)
            ],
            p=0.9
        ),
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1)
            ],
            p=0.9
        )
    ]
    return albu.Compose(train_transform)      

def get_validation_augmentation():
    # Add padding to make image shape divisible by 32
    test_transform = [
        albu.PadIfNeeded(384,640)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2,0,1).astype('float32')     

def get_preprocessing(preprocessing_fn):
    '''
    Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function (can be specific for each 
        pretrained NN)

    Return:
        transform: albumentations.Compose()
    '''

    _transform = [      
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor)
    ]
    return albu.Compose(_transform)

# Visualize resulted random augmented images and masks
augmented_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        classes = ['crack'],
        augmentation = get_training_augmentation()      
)

for i in range(3):
    image, mask = augmented_dataset[0]
    print("Results augmented dataset")
    print(f"image shape = {image.shape}")
    print(f"mask shape = {mask.shape}")
    visualize(image=image, mask=mask)
```
In this code block, the architecture of the encoder as well as the training and validation `Dataset` objects are initialized. These data sets will be iterated over for every training epoch. At the end of this section, an overview of the autoencoder's architecture will be given.

```
# create model and train
import torch
import numpy as np
import segmentation_models_pytorch as smp

ENCODER = 'mobilenet_v2'	
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['crack']
ACTIVATION = 'sigmoid'     
DEVICE = 'cpu'      # cpu or cuda

# create segmentation model with pretrained encoder
model = smp.DeepLabV3(
        encoder_name = ENCODER,
        encoder_weights = ENCODER_WEIGHTS,
        classes = len(CLASSES),
        activation = ACTIVATION
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# datasets for training and validation data
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes = CLASSES
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes = CLASSES
)
```
There should be a warning relating to this code block. The arguments of the `DataLoader()` function should be set correctly for your hardware to perform correctly. Initially `num_workers` was chosen way too high, resulting in errors such as: 'The size of tensor a must match the size of tensor b'. Even if the code does not return an error, the network was not able to train, since the intersection of union (IoU) metric returned the same value for every training epoch. The settings below allow training for the specified hardware. `batch_size` also was adjusted as the training process was sometimes 'killed' by the operating system since it did not have enough RAM to store the input batch tensor. 

Some choices for the optimization and performance evaluation of the network are: the [Dice loss/F1 score](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), the [IoU metric/Jaccard score](https://en.wikipedia.org/wiki/Jaccard_index) and the [Adam optimizer](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). It must be noted that other loss functions and optimizers were considered ([take a look here](https://www.kaggle.com/getting-started/133156)), but due to time constraints the evaluation of the network with different loss functions, optimizers and metrics is not discussed in this blog. 

In the training loop, the data is loaded in the network for 40 epochs. Every epoch, the validation data is evaluated with the IoU metric. If the IoU score is higher for the current epoch than for the previous epochs, the model's weight and biases are stored in an `best_model.pth` file. This file can later be used to test the model's performance on 'unseen' data. 

At epoch 25, the learning rate of weight and bias updates is lowered considerably. We interpreted this as an attempt to minimize the fluctuations of new weight updates. The minimum of the loss function is to be reached more steadily such that overshooting this minimum becomes less likely. 

Later can be seen from the results that the accuracy of the autoencoder's prediction is not optimal and sometimes overfitting at the edges occurs (detects the edge of the image as a crack). We suspect this has to do with the optimizer not being able to find a global optimum of the loss function. Possible solutions could include: more training data needed to increase the accuracy of the prediction or simply more training epochs (random data augmentation). Improvements to current parameters and other regularization could not be performed at this time.

```
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)    
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)	

# checking input tensor dimensions
print(next(iter(train_loader))[0].size(), next(iter(valid_loader))[0].size())

# Defining performance metric, loss and optimizer
loss = smp.utils.losses.DiceLoss()

metrics = [
    smp.utils.metrics.IoU(threshold=0.5)
]

optimizer = torch.optim.Adam([      
    dict(params=model.parameters(), lr=0.0001)  
])

# create epoch runners
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True
)

# train model for 40 epochs

max_score = 0

for i in range(40):
    print(f'\nEpoch: {i}')
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')

    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
```
Once training is complete (even for a few epochs), the code will have saved some weights and biases to the `best_model.pth` file. This file is loaded in as a test model. The test data set is loaded and the prediction of the network are visualized. These predictions were done for models stored during low, medium and high training epochs.

The autoencoder's prediction at training epoch 1:

prediction of test dataset - epoch 1 (image, ground truth, prediction)  	|           
:-------------------------:										|
![outAutoEnc1](/assets/img/outAutoEnc1.png)							|
![outAutoEnc2](/assets/img/outAutoEnc2.png)							|
![outAutoEnc3](/assets/img/outAutoEnc3.png)							|
![outAutoEnc4](/assets/img/outAutoEnc4.png)							|
![outAutoEnc5](/assets/img/outAutoEnc5.png)							|

The autoencoder's prediction at training epoch 28:

prediction of test dataset - epoch 28 (image, ground truth, prediction)  	|           
:-------------------------:										|
![outAutoRun2_1.png](/assets/img/outAutoEncRun2_1.png)						|
![outAutoRun2_2.png](/assets/img/outAutoEncRun2_2.png)						|
![outAutoRun2_3.png](/assets/img/outAutoEncRun2_3.png)						|
![outAutoRun2_4.png](/assets/img/outAutoEncRun2_4.png)						|
![outAutoRun2_5.png](/assets/img/outAutoEncRun2_5.png)						|

The autoencoder's final prediction:

prediction of test dataset - epoch 31 (image, ground truth, prediction)  	|           
:-------------------------:										|
![outAutoRun3_1.png](/assets/img/outAutoEncRun3_1.png)						|
![outAutoRun3_2.png](/assets/img/outAutoEncRun3_2.png)						|
![outAutoRun3_3.png](/assets/img/outAutoEncRun3_3.png)						|
![outAutoRun3_4.png](/assets/img/outAutoEncRun3_4.png)						|
![outAutoRun3_5.png](/assets/img/outAutoEncRun3_5.png)						|

autoEncoder.py - continued 	|
:-------------------------:	|
```
# test best saved model
# load best saved checkpoint
best_model = torch.load('./best_model.pth')

# create test dataset
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES
)

test_dataloader = DataLoader(test_dataset)

# evaluate model on test set

test_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE
)

logs = test_epoch.run(test_dataloader)

# visualize predictions
# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir,
    y_test_dir,
    classes=CLASSES
)

# print 5 predictions compared to ground truth and org image
for i in range(5):
    n = np.random.choice(len(test_dataset))

    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    # gt_mask has shape (3,384,480), but we want (384,480,3)
    gt_mask_trans = np.transpose(gt_mask, (1,2,0))

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    print("Results prediction")
    print(f"image_vis shape = {image_vis.shape}")
    print(f"gt_mask_trans shape = {gt_mask_trans.shape}")
    print(f"pr_mask shape = {pr_mask.shape}")

    visualize(
        image=image_vis,
        ground_truth=gt_mask_trans,
        predicted=pr_mask
    )
```

We hope this code discussion has inspired you to look closer into image segmentation architectures. For the enthusiastic reader, a more detailed summary of the autoencoder's architecture is provided. 

Summary of the PyTorch autoencoder given as an output of `torchsummary.summary()`:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 160, 160]             864
       BatchNorm2d-2         [-1, 32, 160, 160]              64
             ReLU6-3         [-1, 32, 160, 160]               0
            Conv2d-4         [-1, 32, 160, 160]             288
       BatchNorm2d-5         [-1, 32, 160, 160]              64
             ReLU6-6         [-1, 32, 160, 160]               0
            Conv2d-7         [-1, 16, 160, 160]             512
       BatchNorm2d-8         [-1, 16, 160, 160]              32
  InvertedResidual-9         [-1, 16, 160, 160]               0
           Conv2d-10         [-1, 96, 160, 160]           1,536
      BatchNorm2d-11         [-1, 96, 160, 160]             192
            ReLU6-12         [-1, 96, 160, 160]               0
           Conv2d-13           [-1, 96, 80, 80]             864
      BatchNorm2d-14           [-1, 96, 80, 80]             192
            ReLU6-15           [-1, 96, 80, 80]               0
           Conv2d-16           [-1, 24, 80, 80]           2,304
      BatchNorm2d-17           [-1, 24, 80, 80]              48
 InvertedResidual-18           [-1, 24, 80, 80]               0
           Conv2d-19          [-1, 144, 80, 80]           3,456
      BatchNorm2d-20          [-1, 144, 80, 80]             288
            ReLU6-21          [-1, 144, 80, 80]               0
           Conv2d-22          [-1, 144, 80, 80]           1,296
      BatchNorm2d-23          [-1, 144, 80, 80]             288
            ReLU6-24          [-1, 144, 80, 80]               0
           Conv2d-25           [-1, 24, 80, 80]           3,456
      BatchNorm2d-26           [-1, 24, 80, 80]              48
 InvertedResidual-27           [-1, 24, 80, 80]               0
           Conv2d-28          [-1, 144, 80, 80]           3,456
      BatchNorm2d-29          [-1, 144, 80, 80]             288
            ReLU6-30          [-1, 144, 80, 80]               0
           Conv2d-31          [-1, 144, 40, 40]           1,296
      BatchNorm2d-32          [-1, 144, 40, 40]             288
            ReLU6-33          [-1, 144, 40, 40]               0
           Conv2d-34           [-1, 32, 40, 40]           4,608
      BatchNorm2d-35           [-1, 32, 40, 40]              64
 InvertedResidual-36           [-1, 32, 40, 40]               0
           Conv2d-37          [-1, 192, 40, 40]           6,144
      BatchNorm2d-38          [-1, 192, 40, 40]             384
            ReLU6-39          [-1, 192, 40, 40]               0
           Conv2d-40          [-1, 192, 40, 40]           1,728
      BatchNorm2d-41          [-1, 192, 40, 40]             384
            ReLU6-42          [-1, 192, 40, 40]               0
           Conv2d-43           [-1, 32, 40, 40]           6,144
      BatchNorm2d-44           [-1, 32, 40, 40]              64
 InvertedResidual-45           [-1, 32, 40, 40]               0
           Conv2d-46          [-1, 192, 40, 40]           6,144
      BatchNorm2d-47          [-1, 192, 40, 40]             384
            ReLU6-48          [-1, 192, 40, 40]               0
           Conv2d-49          [-1, 192, 40, 40]           1,728
      BatchNorm2d-50          [-1, 192, 40, 40]             384
            ReLU6-51          [-1, 192, 40, 40]               0
           Conv2d-52           [-1, 32, 40, 40]           6,144
      BatchNorm2d-53           [-1, 32, 40, 40]              64
 InvertedResidual-54           [-1, 32, 40, 40]               0
           Conv2d-55          [-1, 192, 40, 40]           6,144
      BatchNorm2d-56          [-1, 192, 40, 40]             384
            ReLU6-57          [-1, 192, 40, 40]               0
           Conv2d-58          [-1, 192, 40, 40]           1,728
      BatchNorm2d-59          [-1, 192, 40, 40]             384
            ReLU6-60          [-1, 192, 40, 40]               0
           Conv2d-61           [-1, 64, 40, 40]          12,288
      BatchNorm2d-62           [-1, 64, 40, 40]             128
 InvertedResidual-63           [-1, 64, 40, 40]               0
           Conv2d-64          [-1, 384, 40, 40]          24,576
      BatchNorm2d-65          [-1, 384, 40, 40]             768
            ReLU6-66          [-1, 384, 40, 40]               0
           Conv2d-67          [-1, 384, 40, 40]           3,456
      BatchNorm2d-68          [-1, 384, 40, 40]             768
            ReLU6-69          [-1, 384, 40, 40]               0
           Conv2d-70           [-1, 64, 40, 40]          24,576
      BatchNorm2d-71           [-1, 64, 40, 40]             128
 InvertedResidual-72           [-1, 64, 40, 40]               0
           Conv2d-73          [-1, 384, 40, 40]          24,576
      BatchNorm2d-74          [-1, 384, 40, 40]             768
            ReLU6-75          [-1, 384, 40, 40]               0
           Conv2d-76          [-1, 384, 40, 40]           3,456
      BatchNorm2d-77          [-1, 384, 40, 40]             768
            ReLU6-78          [-1, 384, 40, 40]               0
           Conv2d-79           [-1, 64, 40, 40]          24,576
      BatchNorm2d-80           [-1, 64, 40, 40]             128
 InvertedResidual-81           [-1, 64, 40, 40]               0
           Conv2d-82          [-1, 384, 40, 40]          24,576
      BatchNorm2d-83          [-1, 384, 40, 40]             768
            ReLU6-84          [-1, 384, 40, 40]               0
           Conv2d-85          [-1, 384, 40, 40]           3,456
      BatchNorm2d-86          [-1, 384, 40, 40]             768
            ReLU6-87          [-1, 384, 40, 40]               0
           Conv2d-88           [-1, 64, 40, 40]          24,576
      BatchNorm2d-89           [-1, 64, 40, 40]             128
 InvertedResidual-90           [-1, 64, 40, 40]               0
           Conv2d-91          [-1, 384, 40, 40]          24,576
      BatchNorm2d-92          [-1, 384, 40, 40]             768
            ReLU6-93          [-1, 384, 40, 40]               0
           Conv2d-94          [-1, 384, 40, 40]           3,456
      BatchNorm2d-95          [-1, 384, 40, 40]             768
            ReLU6-96          [-1, 384, 40, 40]               0
           Conv2d-97           [-1, 96, 40, 40]          36,864
      BatchNorm2d-98           [-1, 96, 40, 40]             192
 InvertedResidual-99           [-1, 96, 40, 40]               0
          Conv2d-100          [-1, 576, 40, 40]          55,296
     BatchNorm2d-101          [-1, 576, 40, 40]           1,152
           ReLU6-102          [-1, 576, 40, 40]               0
          Conv2d-103          [-1, 576, 40, 40]           5,184
     BatchNorm2d-104          [-1, 576, 40, 40]           1,152
           ReLU6-105          [-1, 576, 40, 40]               0
          Conv2d-106           [-1, 96, 40, 40]          55,296
     BatchNorm2d-107           [-1, 96, 40, 40]             192
InvertedResidual-108           [-1, 96, 40, 40]               0
          Conv2d-109          [-1, 576, 40, 40]          55,296
     BatchNorm2d-110          [-1, 576, 40, 40]           1,152
           ReLU6-111          [-1, 576, 40, 40]               0
          Conv2d-112          [-1, 576, 40, 40]           5,184
     BatchNorm2d-113          [-1, 576, 40, 40]           1,152
           ReLU6-114          [-1, 576, 40, 40]               0
          Conv2d-115           [-1, 96, 40, 40]          55,296
     BatchNorm2d-116           [-1, 96, 40, 40]             192
InvertedResidual-117           [-1, 96, 40, 40]               0
          Conv2d-118          [-1, 576, 40, 40]          55,296
     BatchNorm2d-119          [-1, 576, 40, 40]           1,152
           ReLU6-120          [-1, 576, 40, 40]               0
          Conv2d-121          [-1, 576, 40, 40]           5,184
     BatchNorm2d-122          [-1, 576, 40, 40]           1,152
           ReLU6-123          [-1, 576, 40, 40]               0
          Conv2d-124          [-1, 160, 40, 40]          92,160
     BatchNorm2d-125          [-1, 160, 40, 40]             320
InvertedResidual-126          [-1, 160, 40, 40]               0
          Conv2d-127          [-1, 960, 40, 40]         153,600
     BatchNorm2d-128          [-1, 960, 40, 40]           1,920
           ReLU6-129          [-1, 960, 40, 40]               0
          Conv2d-130          [-1, 960, 40, 40]           8,640
     BatchNorm2d-131          [-1, 960, 40, 40]           1,920
           ReLU6-132          [-1, 960, 40, 40]               0
          Conv2d-133          [-1, 160, 40, 40]         153,600
     BatchNorm2d-134          [-1, 160, 40, 40]             320
InvertedResidual-135          [-1, 160, 40, 40]               0
          Conv2d-136          [-1, 960, 40, 40]         153,600
     BatchNorm2d-137          [-1, 960, 40, 40]           1,920
           ReLU6-138          [-1, 960, 40, 40]               0
          Conv2d-139          [-1, 960, 40, 40]           8,640
     BatchNorm2d-140          [-1, 960, 40, 40]           1,920
           ReLU6-141          [-1, 960, 40, 40]               0
          Conv2d-142          [-1, 160, 40, 40]         153,600
     BatchNorm2d-143          [-1, 160, 40, 40]             320
InvertedResidual-144          [-1, 160, 40, 40]               0
          Conv2d-145          [-1, 960, 40, 40]         153,600
     BatchNorm2d-146          [-1, 960, 40, 40]           1,920
           ReLU6-147          [-1, 960, 40, 40]               0
          Conv2d-148          [-1, 960, 40, 40]           8,640
     BatchNorm2d-149          [-1, 960, 40, 40]           1,920
           ReLU6-150          [-1, 960, 40, 40]               0
          Conv2d-151          [-1, 320, 40, 40]         307,200
     BatchNorm2d-152          [-1, 320, 40, 40]             640
InvertedResidual-153          [-1, 320, 40, 40]               0
          Conv2d-154         [-1, 1280, 40, 40]         409,600
     BatchNorm2d-155         [-1, 1280, 40, 40]           2,560
           ReLU6-156         [-1, 1280, 40, 40]               0
MobileNetV2Encoder-157  [[-1, 3, 320, 320], [-1, 16, 160, 160], [-1, 24, 80, 80], [-1, 32, 40, 40], [-1, 96, 40, 40], [-1, 1280, 40, 40]]               0
          Conv2d-158          [-1, 256, 40, 40]         327,680
     BatchNorm2d-159          [-1, 256, 40, 40]             512
            ReLU-160          [-1, 256, 40, 40]               0
          Conv2d-161          [-1, 256, 40, 40]       2,949,120
     BatchNorm2d-162          [-1, 256, 40, 40]             512
            ReLU-163          [-1, 256, 40, 40]               0
          Conv2d-164          [-1, 256, 40, 40]       2,949,120
     BatchNorm2d-165          [-1, 256, 40, 40]             512
            ReLU-166          [-1, 256, 40, 40]               0
          Conv2d-167          [-1, 256, 40, 40]       2,949,120
     BatchNorm2d-168          [-1, 256, 40, 40]             512
            ReLU-169          [-1, 256, 40, 40]               0
AdaptiveAvgPool2d-170           [-1, 1280, 1, 1]               0
          Conv2d-171            [-1, 256, 1, 1]         327,680
     BatchNorm2d-172            [-1, 256, 1, 1]             512
            ReLU-173            [-1, 256, 1, 1]               0
          Conv2d-174          [-1, 256, 40, 40]         327,680
     BatchNorm2d-175          [-1, 256, 40, 40]             512
            ReLU-176          [-1, 256, 40, 40]               0
         Dropout-177          [-1, 256, 40, 40]               0
            ASPP-178          [-1, 256, 40, 40]               0
          Conv2d-179          [-1, 256, 40, 40]         589,824
     BatchNorm2d-180          [-1, 256, 40, 40]             512
            ReLU-181          [-1, 256, 40, 40]               0
          Conv2d-182            [-1, 1, 40, 40]             257
UpsamplingBilinear2d-183          [-1, 1, 320, 320]               0
         Sigmoid-184          [-1, 1, 320, 320]               0
      Activation-185          [-1, 1, 320, 320]               0
================================================================
Total params: 12,647,937
Trainable params: 12,647,937
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.17
Forward/backward pass size (MB): 849.44
Params size (MB): 48.25
Estimated Total Size (MB): 898.86
----------------------------------------------------------------
```

## Discussion

We ran the trainer for a total of 40 epochs. On the 31st epoch, the validation set hit its highest IoU score. Subsequently, the model we obtained from the learner is derived from this epoch, and the predicted images we showed above are derived from this model. To make reading easy, we reiterate an example of the final model's predicted crack segmentation.

prediction of test dataset - epoch 31 (image, ground truth, prediction)  	|           
:-------------------------:										|
![outAutoRun3_1.png](/assets/img/outAutoEncRun3_1.png)						|

From first sight it might be apparant that the results we obtained did not live up to the original paper's goals. Before we make any conclusions, let's take a look at some metrics that help us evaluate our segmentation model better, that were outputted after every epoch: dice-loss and IoU scores.

Dice loss for the validation set at each epoch | IoU score for the validation set at each epoch          
:-------------------------:|:-------------------------:
![dice_loss.png](/assets/img/dice_loss.png) | ![iou_score.png](/assets/img/iou_score.png) 

The graph plotting the dice loss at each epoch shows a general downtrend behavior, while the IoU score does the opposite. 

Is this good? Let's take a step back and ask ourselves what these metrics for evaluating semantic segmentation models mean.

Dice coefficient explained. | IoU score explained.
:-------------------------:|:-------------------------:
![dice_explained.png](/assets/img/dice_explained.png) | ![iou_explained.png](/assets/img/iou_explained.png)

Unlike the naive method of pixel accuracy (PA), which represents the ratio of correctly classified pixels, the IoU, also known as the Jaccard index, takes the overlap between the prediction and the ground truth into account. More specifically, it is the overlap divided by its union. 

For the dice coefficient metric, a similar calculation is made. As illustrated, you need to know the overlap between the two classes, in our case the predicted image and the ground truth. This overlap, doubled, divided by the total pixels each image makes up individiually returns the dice coefficient. Both metrics are bounded between 0 and 1.

The metrics outputted by our learner were no a fractions, however. The IoU score scaled up to 100+, and the dice coefficient was expressed as dice loss, with values around -0.9. We attempted to look into what the differences were. The dice loss is, [according to sources](https://medium.com/ai-salon/understanding-dice-loss-for-crisp-boundary-detection-bb30c2e5f62b#:~:text=Dice%20loss%20originates%20from%20S%C3%B8rensen,for%203D%20medical%20image%20segmentation.), 1 - Dice coefficient. This does not, however, explain the negative values our learner outputted after every epoch. The IoU score was even more of a mystery, as the formula does not allow values over 1 to be theoretically possible. Our assumption is that our algorithm, somewhere, applies a transformation function on the IoU score. If, after an epoch, the IoU score of a validation set was higher than the maximum IoU score thus far, the model and the maximum would be updated to that epoch's IoU score. Translating these high values of IoU back to its fraction proved not possible, though with a bit more time on our hands, we would dive into this to see where this transformation happens.

The fact that a loss metric steadily lowers, and that a metric for evaluating image segmentation improves, does show that the deep learner steadily improved itself over the array of epochs. However, it was not enough to produce satisfactory results.

There are a couple of reasons why we suspect this happened. The first, and perhaps most significant limitation in our effort is the lack of annotated data we had at our disposable. On top of that, the original model had vastly more images  as well. The model we attempted to reproduce had the luxury of training and testing on data that had cracks in tunnels annotated by experts. Furthermore, their datawas of cracks in tunnels, while ours focussed on an array of cracks in different structures. This had the effect of not having a stable environment and different illumination among others.

Another reason for our predictor's inconsistency was our lack of readily available computing power. The dataset was relatively small compared to the original paper's, but even with the small amount of data, a single run took almost 26 hours. This meant that improving parameters was not a quick task. This also meant that the number of epochs was limited, as well as the number of augmentations we could put on on each image.

Each point we discussed that might have contributed to a lesser performance of our predictor are points that could be overridden, given we had better data available, better computing power, or more insight in the original paper's code. With these limitations overcome, we believe that we could have achieved the original performance of the unannotated data set.

####References
- [1] Song, Q., Wu, Y., Xin, X., Yang, L., Yang, M., Chen, H., Liu, C., Hu, M., Chai, X., Li, J. (2019). Real-Time Tunnel Crack Analysis System via Deep Learning. IEEE Access, 7, 64186–64197. https://doi.org/10.1109/access.2019.2916330
- [2] A. G. Howard et al., "MobileNets: Efficient convolutional neural networks for mobile vision applications", Proc. CVPR, 2017. https://arxiv.org/abs/1704.04861
- [3] L.-C. Chen, G. Papandreou, F. Schroff and H. Adam, Rethinking atrous convolution for semantic image segmentation, 2017, https://arxiv.org/abs/1706.05587.

####Overview of general tasks:
Michiel Firlefyn:
- Cloned and implemented the predictor
- Tweaked code to make it work on input data
- Ran the deep learner on his machine
- Bugfixed the IoU score bug
- Wrote about the results

Matthijs van Wijngaarden:
- Tweaked code to make it work on input data
- Bugfixed the image resolution bug
- Produced results from output
- Wrote about the introduction and discussion