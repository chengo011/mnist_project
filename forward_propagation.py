import numpy as np
from functions import *

def softmax(x): #Softmax Funktion
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) #Numerische Stabilität (Übung 2 Teil 4)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward(X, params): #Forward Propagation
    #Weights und Biases extrahieren
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']

    cache = {} #Zwischenspeicher

    a0 = X #Input Layer
    cache['a0'] = a0

    #Hidden Layer 1
    z1 = a0 @ W1 + b1
    a1 = softplus(z1)
    cache['z1'] = z1
    cache['a1'] = a1

    #Hidden Layer 2
    z2 = a1 @ W2 + b2
    a2 = softplus(z2)
    cache['z2'] = z2
    cache['a2'] = a2

    # Output Layer
    z3 = a2 @ W3 + b3
    a3 = softmax(z3)
    cache['z3'] = z3
    cache['a3'] = a3

    return cache

def initialize_params(layer_dims, seed=42): #Hilfsfunktion für Parameter Initialisierung
    #layer_dims = Lister der Schichtgrößen
    np.random.seed(seed)
    params = {}
    for l in range(1, len(layer_dims)):
        n_in = layer_dims[l - 1]
        n_out = layer_dims[l]

        params[f'W{l}'] = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        params[f'b{l}'] = np.zeros((1, n_out))
    return params #return dictionary für weights und biases