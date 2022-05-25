import pathlib
import torch
import os
import sys

import albumentations
from unet import UNet
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from torch.utils.data import DataLoader
from customdatasets import *
from transformations import *
from trainer import *
from torchinfo import summary
from lr_rate_finder import *
from inference import *
from matplotlib.pyplot import imsave


print(torch.__version__)

if len(sys.argv) != 2:
    print('ips_segmentation [build model / inference]')
    sys.exit(-1)

build_type = int(sys.argv[1])

if torch.cuda.is_available():
    print('Using GPU')
    device = torch.device('cuda')
else:
    torch.device('cpu')

model_name = 'iPS_segmentation.pt'
rate_finder = True
epochs = 1000
dimension = 256
crop_size = 128


# Create our dataset:
if build_type == 1:
    root = pathlib.Path.cwd() / 'iPS-Data'


    def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
        """Returns a list of files in a directory/path. Uses pathlib."""
        filenames = [file for file in path.glob(ext) if file.is_file()]
        return filenames


    inputs = get_filenames_of_path(root / 'Input')
    targets = get_filenames_of_path(root / 'Target')

    pre_transforms = ComposeDouble([
        FunctionWrapperDouble(center_crop_to_size, input=True, target=True, size=dimension),
        FunctionWrapperDouble(resize, input=True, target=False, output_shape=(dimension, dimension, 1)),
        FunctionWrapperDouble(resize,
                              input=False,
                              target=True,
                              order=0,
                              anti_aliasing=False,
                              preserve_range=True,
                              output_shape=(dimension, dimension)),
    ])

    training_transforms = ComposeDouble([
        # AlbuSeg2d(albumentations.RandomCrop(width=crop_size, height=crop_size)),
        AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
        AlbuSeg2d(albumentations.RandomRotate90(p=0.5)),
        AlbuSeg2d(albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5, rotate_limit=45, p=0.75)),
        # AlbuSeg2d(albumentations.RandomBrightnessContrast(p=0.2)),
        FunctionWrapperDouble(create_dense_target, input=False, target=True),
        FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
        FunctionWrapperDouble(normalize_01)
    ])

    validation_transforms = ComposeDouble([
        FunctionWrapperDouble(create_dense_target, input=False, target=True),
        FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
        FunctionWrapperDouble(normalize_01)
    ])

    random_seed = 42

    train_size = 0.8

    inputs_train, inputs_valid = train_test_split(inputs, random_state=random_seed, train_size=train_size, shuffle=True)
    targets_train, targets_valid = train_test_split(targets, random_state=random_seed, train_size=train_size, shuffle=True)

    dataset_train = SegmentationDataset(inputs=inputs_train,
                                        targets=targets_train,
                                        transform=training_transforms,
                                        use_cache=True,
                                        pre_transform=pre_transforms)
    dataset_valid = SegmentationDataset(inputs=inputs_valid,
                                        targets=targets_valid,
                                        transform=validation_transforms,
                                        use_cache=True,
                                        pre_transform=pre_transforms)

    batch = dataset_train[0]
    x, y = batch

    print(f'x = shape: {x.shape}; type: {x.dtype}')
    print(f'x = min: {x.min()}; max: {x.max()}')
    print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')

    dataloader_training = DataLoader(dataset=dataset_train, batch_size=2, shuffle=True)
    dataloader_validation = DataLoader(dataset=dataset_valid, batch_size=2, shuffle=True)

    # Create the UNet model
    model = UNet(in_channels=1,
                 out_channels=2,
                 n_blocks=4,
                 start_filters=32,
                 activation='relu',
                 normalization='batch',
                 conv_mode='same',
                 dim=2).to(device)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training dataset
    trainer = Trainer(model=model,
                      device=device,
                      criterion=criterion,
                      optimizer=optimizer,
                      training_DataLoader=dataloader_training,
                      validation_DataLoader=dataloader_validation,
                      lr_scheduler=None,
                      epochs=epochs,
                      epoch=0)

    training_losses, validation_losses, learning_rates = trainer.run_trainer()

    torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)

    if rate_finder:
        lrf = LearningRateFinder(model, criterion, optimizer, device)
        lrf.fit(dataloader_training, steps=1000)
        lrf.plot()

#
if build_type == 2:
    # default test image:
    root = pathlib.Path.cwd() / 'iPS-Data' / 'Test'


    def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
        """Returns a list of files in a directory/path. Uses pathlib."""
        filenames = [file for file in path.glob(ext) if file.is_file()]
        return filenames


    # input and target files
    images_names = get_filenames_of_path(root / 'Input')
    targets_names = get_filenames_of_path(root / 'Target')

    images = [imread(img_name) for img_name in images_names]
    # targets = [imread(tar_name) for tar_name in targets_names]
    targets = [np.ndarray(images[0].shape) for tgt in images_names]

    image_res = [resize(img, (1, dimension, dimension)) for img in images]
    resize_args = {'order': 0, 'anti_aliasing': False, 'preserve_range': True}
    targets_res = [resize(tar, (dimension, dimension), **resize_args) for tar in targets]

    # Create the UNet model
    model = UNet(in_channels=1,
                 out_channels=2,
                 n_blocks=4,
                 start_filters=32,
                 activation='relu',
                 normalization='batch',
                 conv_mode='same',
                 dim=2).to(device)

    model_weights = torch.load(pathlib.Path.cwd() / model_name)

    model.load_state_dict(model_weights)

    def preprocess(img: np.ndarray):
        img = normalize_01(img)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        return img

    def postprocess(img: torch.tensor):
        img = torch.argmax(img, dim=1)
        img = img.cpu().numpy()
        img = np.squeeze(img)
        img = re_normalize(img)
        return img

    output = [predict(img, model, preprocess, postprocess, device) for img in image_res]

    for i in range(0, len(output)):
        # output_full = resize(output[i], ())
        imsave(targets_names[i], output[i])

    print(len(output))
