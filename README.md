# a4
### Part-1 K-Nearest Neighbours:
From the provided pdf I can understand that k-nearest neighbors is a type of non-parametric supervised machine learning algorithm that is used for both classification and regression tasks. For classification, the principle behind k-nearest neighbors is to find k training samples that are closest in distance to a new sample in the test dataset, and then make a prediction based on those samples. 

To check for the nearest neighbours I have made use of the functions defined in the utils.py file that is the euclidean distance and the manhattan distance. The manhattan distance function is similar to the one defined for the raichu code. I have made use of the min-heap to store the neighbours and each time I'm calculating the length of the neighbours if it is less than the distance between then Imadding it to the queue. Each Element is stored as negative distance along with the corresponding distance from the node. 

### Part-2 Multi-Layer Perceptron:
From the provided pdf I can understand that individual neurons are arranged into multiple layers that connect to each other to create a network called a neural network (or multilayer perceptron). The first layer is always the input layer that represents the input of a sample from the dataset. The layers after the input layer are called hidden layers because they are not directly exposed to the dataset inputs. Multilayer perceptrons must have at least one hidden layer in their network.

One-hot encoding creates an array where each column represents a possible categorical value from the original data 

In the code, I have created a feedforward system, a backpropogation system for updating the weights in accordance to the error. I had made use of the gradient descent in the start a very basic algorithim which works with backtracking as well i.e my code was working in a way in which it took a point, got the output for that point and then it adjusted the weight for that point. Later on, since the code was taking long to be implemented I changed the approach to batch gradient descent where in the code was implemented within a few minutes unlike hours to run