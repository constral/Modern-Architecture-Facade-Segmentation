# Modern Architecture Facade Segmentation

AI model for facade recognition of buildings with modern architectural styles
Dataset used: Irregular Facades (https://www.mdpi.com/2075-5309/14/9/2602), purported to “show a significant advantage in terms of average WMIoU (0.722) and accuracy (0.837) over the [other available datasets]”

The whole dataset was first split into 80% training images, 20% validation images.
Afterwards, the dataset was augmented to produce 4 processed variants based on each original image.
Processing steps (for each original image):
- convert mask RGB -> grey level masks in order to make it easier to convert to floats
- horizontal flips (easy augmentation for buildings)
- scale "horizontally" (1.1x stretch) and "vertically" (0.9x stretch)
- pad to square and resize to 256x256

The script implements a standard U-Net Encoder-Decoder architecture:
- Encoder (Downsampling): Uses 4 blocks of convolutions and MaxPooling to capture image features and reduce spatial dimensions.
- Bottleneck: A central convolution block connecting the encoder and decoder.
- Decoder (Upsampling): Uses 4 blocks of UpSampling2D followed by concatenation (skip connections) to restore the image size while retaining fine details from earlier layers.
- Output: A final Conv2D layer with softmax activation to output a probability map for the 6 defined classes.

Callbacks:
- ReduceLROnPlateau: Halves the learning rate if validation loss stops improving for 3 epochs.
- EarlyStopping: Stops training completely if validation loss doesn't improve for 9 epochs.
- Custom ValIoUCallback for training evaluation

At the end of every epoch, the custom callback iterates over the entire validation set to calculate:
- Mean Intersection over Union: The average overlap accuracy.
- Weighted Mean IoU: Accuracy weighted by the size of the objects. Handles class imbalance, as not all segmented objects have the same size.
Model is saved only when WMIoU improves.
