import tensorflow as tf 
import pandas as  pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from tensorflow.python.framework import ops
from sklearn.utils import shuffle
import math

#defining helping functions
def one_hot_matrix(labels , C ):
	C = tf.constant(C , name = "C")
	one_hot_matrix = tf.one_hot(labels , C , axis = 0)
	sess = tf.Session()
	one_hot = sess.run(one_hot_matrix)
	sess.close()
	return one_hot

def initialize_parameters():
	W1 = tf.get_variable("W1" , [n_nodes_l1 , 784 ] , initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	b1 = tf.zeros((n_nodes_l1 , 1))
	W2 = tf.get_variable("W2" , [n_nodes_l2, n_nodes_l1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	b2 = tf.zeros((n_nodes_l2 ,1))
	W3 = tf.get_variable("W3" , [n_nodes_l3 ,n_nodes_l2] , initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	b3 = tf.zeros((n_nodes_l3,1))
	W4 = tf.get_variable("W4",[n_classes , n_nodes_l3] , initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	b4 = tf.zeros((n_classes , 1))

	parameters = {"W1":W1 , "b1":b1 ,"W2":W2 , "b2":b2 ,"W3":W3 , "b3":b3 ,"W4":W4 , "b4":b4 }
	return parameters


def create_placeholders(n_x , n_y):
	X = tf.placeholder(tf.float32 , [n_x , None] )
	Y = tf.placeholder(tf.float32 , [n_y , None] )
	return X , Y

def forward_propagation(X , parameters):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	W3 = parameters["W3"]
	b3 = parameters["b3"]
	W4 = parameters["W4"]
	b4 = parameters["b4"]

	Z1 = tf.add(tf.matmul(W1 , X) , b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2 , A1) , b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3 , A2) , b3)
	A3 = tf.nn.relu(Z3)
	Z4 = tf.add(tf.matmul(W4 , A3) , b4)

	return Z4

def compute_cost(Z4, Y):
	logits = tf.transpose(Z4)
	labels = tf.transpose(Y)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits , labels = labels))
	return cost


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
   
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


#CREATING DATA 
df = pd.read_csv("D:/AI/deeplearning.ai/tensorFlow/train.csv")
df = shuffle(df)

X = np.array(df.drop(['label'] ,axis =1 ))
Y = np.array(df['label'])

X_train_orig , X_test_orig , Y_train_orig , Y_test_orig = train_test_split(X , Y , test_size = 0.2)

#flattening data so as to pass into neural network layers
X_train_flatten = X_train_orig.T 
X_test_flatten = X_test_orig.T 

#preprocessing data
X_train = X_train_flatten / 255
X_test  = X_test_flatten / 255

#converting labels to one hot matrix
Y_train = one_hot_matrix(Y_train_orig , C = 10)
Y_test  = one_hot_matrix(Y_test_orig , C = 10)



# CREATING MODEL VARIABLES
n_nodes_l1 = 500
n_nodes_l2 = 500
n_nodes_l3 = 500
n_classes  = 10


def model(X_train , Y_train , X_test , Y_test , num_epochs , learning_rate , minibatch_size):

	ops.reset_default_graph()

	tf.set_random_seed(1)
	seed = 3
	costs = []
	(n_x , m) = X_train.shape
	n_y = Y_train.shape[0]

	parameters = initialize_parameters()
	X , Y = create_placeholders(n_x , n_y)
	Z4 = forward_propagation(X , parameters)
	cost = compute_cost(Z4 , Y)
	optimizer =  tf.train.AdamOptimizer(learning_rate = learning_rate ).minimize(cost)
	init = tf.global_variables_initializer()


	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			epoch_cost = 0
			num_minibatches = int(m / minibatch_size)
			seed = seed + 1
			minibatches = random_mini_batches(X_train , Y_train , minibatch_size , seed = 0)

			for minibatch in minibatches:
				(minibatch_X , minibatch_Y) = minibatch
				_ , minibatch_cost = sess.run([optimizer , cost] , feed_dict = {X : minibatch_X , Y : minibatch_Y})
				epoch_cost += minibatch_cost / num_minibatches

			if epoch % 5 == 0:
				costs.append(epoch_cost)
				print("cost after {} epoch:{}".format(epoch , epoch_cost))

		plt.plot(np.squeeze(costs))
		plt.xlabel('cost')
		plt.ylabel('epochs(per 100)')
		plt.title('learning rate='+str(learning_rate))
		plt.show()

		parameters = sess.run(parameters)
		print("parameters tuned")

		correct_prediction = tf.equal(tf.argmax(Z4) , tf.argmax(Y))
		accuracy  = tf.reduce_mean(tf.cast(correct_prediction , float) )

		print("Train accuracy :" , accuracy.eval({X:X_train , Y:Y_train}))
		print("Test accuracy :" , accuracy.eval({X: X_test , Y:Y_test}))


		return parameters




parameters = model(X_train , Y_train , X_test , Y_test , num_epochs = 20 , learning_rate = 0.0001 , minibatch_size = 100)

