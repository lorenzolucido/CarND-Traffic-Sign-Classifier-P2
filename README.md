# **Traffic Sign Recognition**

_Lorenzo's version_

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./classes_repartition.png "Visualization"
[preprocess]: ./preprocessing.png "preprocessing"
[internet_images]: ./internet_images.png "Internet Predictions"
[augmented]: ./augmentation.png "Internet Predictions"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Link to the [project code](https://github.com/lorenzolucido/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing the classes repartition within the training, validation and testing sets.
I also included a simple visualization of random images from the dataset.

![alt text][image1]

### Design and Test a Model Architecture

I did try a few pre-processing steps: grayscale, HSV, standardization by channels or on the full image. Eventually I decided to go for the RGB standardization by channels as recommended [in the CS231n Data pre-processing chapter](http://cs231n.github.io/neural-networks-2/)

Below is how a normalized image would look:
![alt text][preprocess]

Additionally, I augmented the dataset by using the pre-built tensorflow functions.
For each image, during training, I did apply a small variation of brightness, contrast and saturation, as well as a random cropping. See below:
![alt text][augmented]

My final model (**LorenzoNet**)was very much inspired by the VGG architecture, it consists of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x16 	|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 16x16x16 				|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x32     									|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 8x8x32 				|
| RELU					|												|
|Convolution 3x3	    | 1x1 stride, same padding, outputs 8x8x64      									|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 4x4x64 				|
| RELU					|		|
| Fully connected		| inputs 1024, outputs 120       									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| inputs 120, outputs 84        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| inputs 84, outputs number of categories        									|
| Softmax				|    outputs final probabilities     									|



To train the model, I used a batch size of 128 training samples, 30 Epochs and learning rate of 0.001 with the Adam Optimizer.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 97.2%
* test set accuracy of 95.8%

I initially started with a LeNet-5 architecture (2 convolutional layers and 3 fully connected layers), results were around 85%-90% accuracy on the validation set. Then I started to read about the current state-of-the art in image recognition. Among [Inception](https://arxiv.org/abs/1602.07261), [VGG](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) and [ResNet](https://arxiv.org/abs/1512.03385).
VGG happened to be a good candidate, as the other 2 seem to be relevant for more complex images, eventually the traffic sign dataset does not look that complicated.

So I followed VGG guidelines and I converted my convolutions to 3x3 with a padding that ensures the dimensions along the first 2 axes remain unchanged (i.e. if the input is 32x32xA, the output will be 32x32xB). Then, I converted my max pooling layers to divide the first 2 dimensions by 2 (e.g. input: 32x32xA, output 16x16xA).
I also added dropout on the fully-connected layers.

I finally added a third convolution layer as well as deepened the progressively each of the convolution layers in order to get a good balance between training set accuracy and validation set accuracy.


### Test a Model on New Images

Here are 6 German traffic signs that I found on the web with their respective probabilities:
![alt text][internet_images]

Here are the results of the prediction:

| Number			        | Image			        |     Prediction	        					|
|:---------------------:|:---------------------:|:------------------------------:|
|1| General caution      		| General caution    									|
|2| Turn right ahead    			| Turn right ahead 										|
|3| Speed limit (60 km/h)					| Speed limit (60 km/h)										|
|4| Stop	      		| No passing for vehicles over 3.5 metric tons					 				|
|5| Roundabout mandatory			| End of no passing by vehicles over 3.5 metric tons|
|6| Pedestrians			| Pedestrians	     							|

The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66%.
I was eventually very surprised to see that a model trained in 8 minutes (with a GPU) would be able to classify random images from the Internet just roughly padded with white bands.
The classification of images 1,2,3 and 6 is a clear good choice (>60% probabiliy), while the other 2 are more uncertain (20-30% for the max probability).







_(Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?_

I was not totally sure how to get the `tf_activation` variable here.
