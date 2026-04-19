import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk


def one_hot(y, num_classes=3): #One-hot encoding
    n = len(y)
    one_hot_matrix = np.zeros((n, num_classes))
    one_hot_matrix[np.arange(n), y] = 1
    return one_hot_matrix

def sigmoid(x): #Sigmoid Funktion
    positive_mask = x >= 0 #Fallunterscheidung um Overflow zu vermeiden
    result = np.zeros_like(x, dtype=np.float64)

    result[positive_mask] = 1 / (1 + np.exp(-x[positive_mask])) #Für x>0: 1 / (1 + exp(-x))

    exp_x = np.exp(x[~positive_mask]) #Für x<0: exp(x) / (1 + exp(x))
    result[~positive_mask] = exp_x / (1 + exp_x)
    return result

def sigmoid_derivative(x): #Ableitung der Sigmoid Funktion
    s = sigmoid(x)
    return s * (1 - s)

def softplus(z): #Softplus Funktion
    return  np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0) #Vermeidung von Overflow für große x

def softplus_derivative(x): #Ableitung der Softplus Funktion
    return sigmoid(x)

def gradient(params, grads, learning_rate): #Gradientenabstieg Aufgabe 4
    # Anzahl der Schichten
    num_layers = len(params) // 2  # Wir haben W und b pro Schicht

    for l in range(1, num_layers + 1):
        # W^(l) = W^(l) - η * dW^(l)
        params[f'W{l}'] = params[f'W{l}'] - learning_rate * grads[f'dW{l}']

        # b^(l) = b^(l) - η * db^(l)
        params[f'b{l}'] = params[f'b{l}'] - learning_rate * grads[f'db{l}']

    return params

