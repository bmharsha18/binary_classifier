import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df1 = pd.read_csv("X_train.txt", header=None)
df2 = pd.read_csv("Y_train.txt",header=None)

X_train = np.array(df1)
Y_train = np.array(df2)
Y_train = Y_train.reshape(1,len(Y_train[0]))


def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z 
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W,A)+b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1/m*np.dot(dZ,np.transpose(A_prev))
    db = 1/m*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(np.transpose(W),dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 
    for l in range(1,L+1):
        parameters["W"+str(l)]-=learning_rate*grads["dW"+str(l)]
        parameters["b"+str(l)]-=learning_rate*grads["db"+str(l)]
        
    return parameters



def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    grads = {}
    costs = list()                                                      
    (n_x, n_h, n_y) = layers_dims
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        if i%10 == 0:
            print(i)

        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        
        # Compute cost
        cost = compute_cost(A2, Y)
        costs.append(cost)
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
    return parameters,costs

n_x = 12288     
n_h = 15
n_y = 1
layers_dims = (n_x, n_h, n_y)    


para,costs = two_layer_model(X_train, Y_train, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False)

tW1 = para["W1"]
tb1 = para["b1"]
tW2 = para["W2"]
tb2 = para["b2"]

learning_rate = 0.0075
xvalues = list()
yvalues = list()

for i in range(len(costs)):
    xvalues.append(i)
for i in costs:
    yvalues.append(float(i))
    
plt.plot(xvalues,yvalues,'b')
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(learning_rate))
plt.show()

df3 = pd.read_csv("/home/harshabm/Documents/Placements/Projects/Neural Network/Untitled Folder/X_test.txt", header=None)
df4 = pd.read_csv("/home/harshabm/Documents/Placements/Projects/Neural Network/Untitled Folder/Y_test.txt",header=None)

X_test = np.array(df3)
Y_test = np.array(df4)
Y_test = Y_test.reshape(1,len(Y_test[0]))

Out1, ca1 = linear_activation_forward(X_test, tW1, tb1, "relu")
Out2, ca2 = linear_activation_forward(Out1, tW2, tb2, "sigmoid")

Out2[Out2>=0.5] = 1
Out2[Out2 < 0.5] = 0

counter = 0
for j in range(len(Out2[0])):
    if int(Out2[0][j]) == int(Y_test[0][j]):
        #print(int(Out2[0][j])," ",Y_test[0][j])
        counter = counter + 1

print("\nNumber of training samples : ",len(X_train[0]))
print("\nNumber of testing samples : ",len(X_test[0]))
print("\nNumber of matches out of",len(Out2[0]),"are",counter)
val = counter/len(Out2[0]) * 100
print("\nAccuracy: ",val)


