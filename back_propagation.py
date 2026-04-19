import numpy as np
from forward_propagation import *
from functions import *


def mse(y_true, y_pred): #MSE Loss Function
    n = y_true.shape[0]
    return np.sum((y_true - y_pred) ** 2) / n

def mse_derivative(y_true, y_pred): #Ableitung nach y_pred
    n = y_true.shape[0]
    return 2 * (y_pred - y_true) / n

def cross_entropy_loss(y_true, y_pred): #Cross-Entropy Loss Function
    n = y_true.shape[0]
    epsilon = 1e-15 #Wähle epsilon klein um log(0) zu vermeiden
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred_clipped)) / n

def cross_entropy_loss_derivative(y_true, y_pred): #Ableitung nach y_pred
    n = y_true.shape[0]
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y_true / (n * y_pred_clipped)


def backward(params, cache, y_true, loss_type='cross_entropy'): #Backpropagation Funktion
    # Extrahiere Zwischenwerte
    a0 = cache['a0']
    z1, a1 = cache['z1'], cache['a1']
    z2, a2 = cache['z2'], cache['a2']
    z3, a3 = cache['z3'], cache['a3']

    # Extrahiere Gewichte
    W1, W2, W3 = params['W1'], params['W2'], params['W3']

    n = y_true.shape[0]  # Anzahl Samples
    grads = {}

    #Output Layer
    if loss_type == 'cross_entropy': #Cross-Entropy + Softmax
        loss = cross_entropy_loss(y_true, a3)
        delta3 = (a3 - y_true) / n

    elif loss_type == 'mse': #MSE + Softmax
        loss = mse(y_true, a3)
        dL_da3 = mse_derivative(y_true, a3) #Ableitung MSE nach Softmax Output
        delta3 = np.zeros_like(a3)
        for i in range(n): #Jacobi Matrix von Softmax
            s = a3[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - s @ s.T
            delta3[i] = dL_da3[i] @ jacobian

    #Gradienten für W3 und b3
    grads['dW3'] = a2.T @ delta3
    grads['db3'] = np.sum(delta3, axis=0, keepdims=True)

    #Hidden Layer 2
    delta2 = (delta3 @ W3.T) * softplus_derivative(z2)

    grads['dW2'] = a1.T @ delta2
    grads['db2'] = np.sum(delta2, axis=0, keepdims=True)

    #Hidden Layer 1
    delta1 = (delta2 @ W2.T) * softplus_derivative(z1)

    grads['dW1'] = a0.T @ delta1
    grads['db1'] = np.sum(delta1, axis=0, keepdims=True)

    return grads, loss