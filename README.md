# Colorize
Conditional GAN for colorizing grayscale images. Uses UNet architecture for generator and Patch-Discriminator for discriminator.

<br>

## Getting Started
* Clone this repository
* `cd Colorize`
* Download the [2014 COCO validation dataset](https://cocodataset.org/#download) and extract it to `data/images`
* Run `python data/split.py` to split the dataset into training and validation sets

<br>

## How to Use
* Change the hyperparameters in `config.py` if needed
* (Optional) Run `python pretrain_generator.py` to pretrain the generator (L1 Loss)
* If using pretrained generator, change the `pretrained_gen` parameter in `config.py` to `true` otherwise `false`
* Run `python train_GAN.py` to train and save the model
* Store the images to be colorized in `ip` folder
* Update `inference.py` with the path to the checkpoint
* Run `python inference.py` to colorize the images
