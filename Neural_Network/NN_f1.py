import numpy as np
from sklearn import *
import matplotlib.pyplot as plt

# Generate a dataset and plot it
np.random.seed(0)
X, y = datasets.make_gaussian_quantiles(cov=3.,
                                 n_samples=400, n_features=2,
                                 n_classes=3, random_state=0)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)


num_examples = len(X) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 3 # output layer dimensionality
# Gradient descent parameters (I picked these by hand)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

def sig(x):
    return 1 / (1 - np.exp(x))
sig = np.vectorize(sig)

def relu(x):
    return max(0.0, x)
relu = np.vectorize(relu)

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    z3 = a2.dot(W3) + b3
    exp_scores = np.exp(z3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    return 1./num_examples * data_loss

# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    z3 = a2.dot(W3) + b3
    exp_scores = np.exp(z3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim1, nn_hdim2, num_passes=20000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim1) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim1))
    W2 = np.random.randn(nn_hdim1, nn_hdim2) / np.sqrt(nn_hdim1)
    b2 = np.zeros((1, nn_hdim2))
    W3 = np.random.randn(nn_hdim2, nn_output_dim) / np.sqrt(nn_hdim2)
    b3 = np.zeros((1, nn_output_dim))
    # This is what we return at the end
    model = {}  

    epsilon = 0.03
    # Gradient descent. For each batch...
    for i in range(0, num_passes):
        x = [X[:num_examples//2], X[num_examples//2:]]
        Y = [y[:num_examples//2], y[num_examples//2:]]
        for j in range(2):
            # Forward propagation
            z1 = x[j].dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            a2 = np.tanh(z2)
            z3 = a2.dot(W3) + b3
            exp_scores = np.exp(z3)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            # Backpropagation
            delta4 = probs
            delta4[range(num_examples // 2), Y[j]] -= 1
            dW3 = (a2.T).dot(delta4)
            db3 = np.sum(delta4, axis=0, keepdims=True)
            delta3 = delta4.dot(W3.T) * (1 - np.power(a2, 2))
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(x[j].T, delta2)
            db1 = np.sum(delta2, axis=0)
            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW3 += reg_lambda * W3
            dW2 += reg_lambda * W2
            dW1 += reg_lambda * W1
            # Gradient descent parameter update
            W1 += -epsilon * dW1
            b1 += -epsilon * db1
            W2 += -epsilon * dW2
            b2 += -epsilon * db2
            W3 += -epsilon * dW3
            b3 += -epsilon * db3
            # Assign new parameters to the model
            model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
            epsilon -= 0.01 / 20000
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i %.dth: %f"%(i, j + 1, calculate_loss(model)))
    return model

model = build_model(3, 2, print_loss=True)

 # Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                      np.arange(y_min, y_max, 0.1))

Z = predict(model, np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor='k')
plt.contourf(xx, yy, Z, alpha=0.5)

plt.show()
