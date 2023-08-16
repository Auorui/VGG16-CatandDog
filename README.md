# VGG16-CatandDog
Explore the effectiveness of the VGG model, which achieved significant results in the ImageNet image classification competition in 2014, and use VGG for cat and dog classification

## VGG Model Principle
[VGG16 Content Explanation](https://blog.csdn.net/m0_62919535/article/details/132189691?spm=1001.2014.3001.5501)

The main feature of VGG is the use of a series of convolutional kernels with the same size of 3x3 for multiple convolution operations. One advantage of this structure is that it can stack more convolutional layers, allowing the network to learn more complex features.

## Model Prediction

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/118079f49b234fe7b00fa2265207494a~tplv-k3u1fbpfcp-watermark.image?)

## Usage Process
### 1.Run the (annotation_txt.py) file first to generate a class_ The data.txt file roughly reads as follows:
> 0;D:/deeplearning/VGGnet/train/cat/cat000.jpg
> 
> 0;D:/deeplearning/VGGnet/train/cat/cat001.jpg
> 
> 0;D:/deeplearning/VGGnet/train/cat/cat002.jpg
> 
> ......
> 
> 1;D:/deeplearning/VGGnet/train/dog/dog198.jpg
> 
> 1;D:/deeplearning/VGGnet/train/dog/dog199.jpg
> 
> 1;D:/deeplearning/VGGnet/train/dog/dog200.jpg

### 2.Run the(net.py)file and download the pre training weights.

### 3.Run the (main. py) file for model training, and modify parameters based on comments.

  设置相关参数：
  
  * Cuda: Whether to use GPU acceleration, default to False.
  * Net: Select the VGG network version to use, which can be 'vgg16' or 'vgg19'.
  * annotation_path: The annotation file path of the dataset, which is a text file containing image paths and labels.
  * input_shape: Enter the size of the image.
  * num_classes: Number of categories classified.
  * lr: Learning rate.
  * optimizer_type: Select an optimizer, which can be 'Adam' or 'SGD'.
  * percentage: Percentage of validation set.
  * epochs: Training epochs.
  * save_period: How many epochs to save model weights once.
  * save_dir: Directory where model weights and log files are saved.

### 4.Predictive model, available in two ways: predict and dir_predict.


## Project Blog

#### CSDN:https://blog.csdn.net/m0_62919535/article/details/132319018?spm=1001.2014.3001.5502
