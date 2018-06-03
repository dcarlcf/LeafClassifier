#this is a modified version of the classifier
#I just don't want to lose the 92% version

import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

sess = tf.InteractiveSession()

cwd = os.getcwd()

train = pd.read_csv(os.path.join(cwd, 'train.csv'))
test = pd.read_csv(os.path.join(cwd, 'test.csv'))

#grab list of all plant species
plantNamesDup = train['species'].to_dict()

#seperate out duplicates
plantNameDict = {}
index = 0

#assign plant names to a dict
for key,value in plantNamesDup.items():
    if value not in plantNameDict.values():
        plantNameDict[index] = value
        index = index+1

#invert the dict so a plant name can be assigned to a number
invPlantNameDict = {v: k for k, v in plantNameDict.items()}

#print(invPlantNameDict)

#now use the list to create a 990x99 matrix of one hot's for the training data
#initialize matrix
train_y_one_hot = np.zeros((990,99))

#iterate through the plant names from the train data panda
#grab the plant name
#find that name in the dictionary, grab the dictionary key
#and then turn train_y(value, key) from 0 to 1 
for value in (range(train.shape[0])):
	plantName = train['species'].iloc[value]
	keyToActivate = (invPlantNameDict.get(plantName))
	train_y_one_hot[value, keyToActivate] = 1

#now we have a one hot array of labels! sweet
#did some investigating and im pretty sure i did this correctly

#now lets format and normalize the x training data

#take out the id # and label
trainXNP = train.iloc[:, 2:].values

#split up the data
train_x, validate_x, test_x = np.vsplit(trainXNP, [600, 790])

#now split up the y data
train_y, validate_y, test_y = np.vsplit(train_y_one_hot, [600, 790])

#now lets construct the computational graph
#our x train is 850x192
#our y train is 850x99
#W is 192x99
#b is 99

#define numFeatures, num_nodes, numLabels
#num_nodes is the number of nodes in the hidden layer
numFeatures = 192
numLabels = 99
num_nodes = 150
num_nodes2 = 150

#X  
X = tf.placeholder(tf.float32, [None, 192])

#dropout var
keep_prob = tf.placeholder(tf.float32)

# hidden layer
weights_hidden = tf.Variable(tf.random_normal([numFeatures, num_nodes]))
bias_hidden = tf.Variable(tf.random_normal([num_nodes]))
preactivations_hidden = tf.add(tf.matmul(X, weights_hidden), bias_hidden)
activations_hidden = tf.nn.relu(preactivations_hidden)

#dropout layer
activations_hidden_drop = tf.nn.dropout(activations_hidden, keep_prob)

#second hidden
weights_hidden_2 = tf.Variable(tf.random_normal([num_nodes, num_nodes2]))
bias_hidden_2 = tf.Variable(tf.random_normal([num_nodes2]))
preactivations_hidden_2 = tf.add(tf.matmul(activations_hidden_drop, weights_hidden_2), bias_hidden_2)
activations_hidden_2 = tf.nn.sigmoid(preactivations_hidden_2)

#dropout 2
activations_hidden_drop_2 = tf.nn.dropout(activations_hidden, keep_prob)

# output layer
weights_output = tf.Variable(tf.random_normal([num_nodes2, numLabels]))
bias_output = tf.Variable(tf.random_normal([numLabels]))
preactivations_output = tf.add(tf.matmul(activations_hidden_drop_2, weights_output), bias_output)


#W1 = tf.Variable(tf.zeros([192, 99]))
#b1 = tf.Variable(tf.zeros([99]))
#W2 = tf.Variable(tf.zeros([99, 99]))
#b2 = tf.Variable(tf.zeros([99]))
#hiddenV = tf.matmul(x, W1) + b1
#y = tf.matmul(hiddenV, W2) + b2

#y hat prediction function
y_ = tf.placeholder(tf.float32, [None, 99])

#cross entropy
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=preactivations_output))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#now lets start the session and run it
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(preactivations_output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for _ in range(80000):
    sess.run(train_step, feed_dict={X: train_x, y_: train_y, keep_prob: 0.5})

    if _%1000 == 0:
    	print('After ', _, ' iterations:')
    	print('Training accuracy: ')
    	print(sess.run(accuracy, feed_dict={X: train_x,
                                      y_: train_y, keep_prob: 1}))
    	print('Validation accuracy: ')
    	print(sess.run(accuracy, feed_dict={X: validate_x,
    								  y_: validate_y, keep_prob: 1}))


  # Test trained model

print(sess.run(accuracy, feed_dict={X: test_x,
                                      y_: test_y, keep_prob: 1}))

#ran 80,000 iterations
#after 47,000 test accuracy is 1, validation accuracy oscillates between 93.6-94.7
#still only 91% test accuracy 
#conclusion: dropout does not improve this neural network at all

