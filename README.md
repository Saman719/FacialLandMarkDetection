# FacialLandMarkDetection
Facial Landmark Detection (and training) (68 face intresting points) Using Keras And Pure Tensorflow For Model Fitting

**USAGE**
You can use this model after finding a bounding box on the face and pass only the face in the bounding box to this model.
the result of this model is a (68 * 2) = 136 vector which every 2 number from begining is a pair of (x, y). so there are a total number of 68 points.

The data set which this model trained on is available on the project in /Data folder. To use your own data set you need to change data_handler.py file accourding to you needs.

You can alternatively access the dataset from this url : https://alirezakay.github.io/courses/cv/

If you only want to use the model, the only point is that resize your image to size of (75,75,3) and feed it to the model. the landmark_cnn.h5 model can be used directly on your python interpreter
