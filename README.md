# Merbli
Multiple event detection using minimum inputs

Implementation is based on the paper : G. Laput, Y. Zhang, and C. Harrison, “Synthetic Sensors: Towards General-Purpose Sensing", May 2017

Aimed at detecting events happening in a kitchen environment incorperating machine learning. Inputs the machine learning algorithms are sensor data collected using a sensor system planted at one corner of the kitchen.

Dataset_1.csv and Dataset_2.csv are such collected data sets used in this work.
Implimentations were done and tested when a single event is happening as well as when two events are happening simultaneously. Hence the Dataset_1.csv and Dataset_2.csv corresponds to these two cases respectively.

data_collection.py is the source code used when collecting data using the Raspberry Pi.

These collceted datasets are the input to machine learning system. Models were developed for five classifier algorithms. Namely :
   - Random Forest Algorithm
   - Decision Tree Algorithm
   - K- Nearest Neighbors Algorithm
   - Kernel SVM Algorithm
   - Naive Bayes Algorithm
   
for which the source code is available as other_classifiers.py. By uncommenting the relevant model's classifier parameters the program could be run for each othese models seperately.

An implemetation was even done with a deep learning algorithm; Artificial Neural Networks for which the code is available as ANN.py

The source code used in simultaneous event detection is available as simultaneous_event_detection.py which contains all the five classifier models and the deep learning algorithm. By uncommenting the relevant model the code can be run seperately for each of the classifier models needed.

A real time implementation was also done with all these models in the Raspberry Pi and the code used is available as inference.py. In case of real time testing however, the model learning cannot be done in the Raspberry Pi itself. Hence the learning is done in a machine with a high processing power and the models along with the data pre-prepeartion objects are saved to the disk and later these saved models are loaded to the Raspberry Pi to do the real time testing. This function of saving the models to the disk is facilitated in all three source codes for machine learning models; ANN.py, other_classifiers.py and simultaneous_event_detection.py.

An example of a such saved data pre-prepeartion model used for ANN is avaialble as scaleANN.pkl and an example of a saved model is given as best_model_ANN.h5


