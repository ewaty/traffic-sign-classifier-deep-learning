
# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./index.png "Visualisation"
[image2]: ./index1.png "Visualisation"
[image3]: ./index2.png "Occurrence"
[image4]: ./lenet.png "LeNet"
[image5]: ./index3.png "German Traffic Signs"

### Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the basic python methods such as len() and set() to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed accross different classes. It is apparent that certain traffic signs are represented more often than others in the data set.

![alt text][image3]

The following are samples of images from the training data. Photos of traffic signs were taken with different levels of light exposure and at slightly different angles. Background in images also varies. All are centered in the middle.

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

For preprocessing I decided to only normalize the pictures. I divided each pixel value by 255 and subtracted 0.5 from it to achieve data with, roughly, zero mean and standard deviation of 0.5. It was suggested that students convert images to grayscale. However, I have achieved higher validation accuracy with a model accepting color images. This seems intuitive, as color is an important feature of a traffic sign.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5   	| 1x1 stride, valid padding, outputs 28x28x10 	|
| Softsign				|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x20 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x20   |
| Softsign				|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x20 	     			|
| Fully connected		| Outputs vector size 120						|
| Softsign				|												|
| Fully connected		| Outputs vector size 86						|
| Softsign				|												|
| Fully connected		| Outputs vector size 43						|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer because it was used in previous Udacity exercise. I set the batch size to 64, number of epochs to 15 and the learn rate of 0.0007. These hyperparameters were selected through trial and error and were determined to yield the best validation accuracy from all the hyperparameter sets that were tried.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.960
* test set accuracy of 0.941

If an iterative approach was chosen:

The starting point for the architecture was a LeNet-5 convolutional neural network. I chose this architecture because it has been proven effective in recognizing handwritten letters in 32x32 images. I speculated it will also be useful in classifying 32x32 traffic signs. I used ReLu's as activation functions.
![alt text][image4]


The initial architecture performed at around 0.89 validation accuracy. To increase the validation accuracy I experimented with different activation functions and their combinations. I tried ReLu, ReLu6, Elu, SoftPlus, SoftSign, Sigmoid and TanH. Of these functions, SoftSign activation function has lead to the highest validation accuracy at around 0.93 when used for all convolutional and fully connected layers.

I have experimented with using dropout functions with probablility of keeping at 0.5 and 0.8 throughout the architecture. Their inclusion has led to slight drop in the validation accuracy, therefore I have not incuded them in a final model.

I have also tried removing max pooling and connecting it's neighboring layers. This decreased the validation accuracy.

As a final adjustment I tried elongating the layers. First convolutional layer was elongated from depth 6 to depth 10. Second convolutional layer from 16 to 20. Second fully connected layer from 84 to 86. The validation accuracy reach 0.960 where it stands now.

Elongating convolutional layers might increase the ability of the network to detect more variable features. Beacause traffic signs have more diversity of features than handwritten letters, longer layers might be more capable of detecting the variety of features.

The model has a final training accuracy of 1.000, validation accuracy of 0.960 and testing accuracy of 0.941. Although the network is overfitting to a small degree, it is still a good model to classify traffic signs.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5]

The first image ("Turn right at intersecion") might be difficult to classify because of it's similarity to other signs, such as turn left at intersection. On the other hand, it's distinct blue color might help diffrentiate it from other circular signs. The blue background might, in some conditions, obscure the sign.

The second image ("No overtaking") is very similar to "No overtaking - trucks only" at 32x3 resolution.

The third image ("Wrong way") appears to be easy to classify due to it's distinctive color and shape.

The fourth image ("Yield") also seems to be an easily classifiable sign.

The fifth image ("Right of way at intersection") can be confused with other signs of similar shape and color, yet different content.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Turn right            | Turn right                                    |
| No overtaking         | No overtaking  								|
| Wrong way     		| Wrong way 									|
| Yield					| Yield											|
| Right of way	   		| Right of way					 				|



The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.941. The high accuracy is not surprising because the accuracy of the testing set is also high. The traffic signs selected were also not underrepresented in the training set which should result in a higher classification accuracy for them.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For all images, the certainty of classification is close to 1. For the first sign, most signs that constitute the lower probabilities are signs that are similar in meaning and thus appearance to the correct sign. Those signs tell the driver that his choices on the intersection are limited and he or she must chose from the available options, as in Go straight or left. For the second image the second most probable sign (End of no passing) is one very similar to the correct sign (No passing). For other signs such similarities cannot be seen but their probabilities are also higher ()

First image

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.9987                | Turn right ahead                              |
| 5.054e-04             | Ahead only		                			|
| 3.400e-04          	| Go straight or left 							|
| 3.138e-04  			| Turn left ahead	    						|
| 5.304e-05    	   		| Roundabout					 				|

Second image

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.9968                | No passing                                    |
| 2.700e-03             | End of no passing		               			|
| 2.401e-04          	| Go straight or left 							|
| 1.100e-04  			| Dangerous curve to the left					|
| 8.620e-05    	   		| Dangerous curve to the right	 				|

Third image

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.9999                | No entry                                      |
| 6.949e-05             | Speed limit 20 km/h		          			|
| 1.489e-05          	| Stop              							|
| 8.941e-06  			| Double curve              					|
| 7.679e-06   	   		| Priority road                 				|

Fourth image

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.9999                | Yield                                         |
| 4.847e-06             | No passing        		          			|
| 1.459e-06          	| Speed limit 60 km/h  							|
| 2.318e-07  			| Speed limit 50 km/h          					|
| 1.123e-07   	   		| No vehicles                     				|

Fifth image

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.9999                | Right of way                                  |
| 3.781e-05             | Beware of ice and snow	          			|
| 1.652e-05          	| Pedastrians       							|
| 6.087e-06  			| Double curve               					|
| 2.153e-06   	   		| Speed limit 100 km/h                    		|
