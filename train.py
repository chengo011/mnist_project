import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from functions import *
from forward_propagation import *
from back_propagation import *
from sklearn.metrics import accuracy_score

def train_full_batch(X_train, y_train, X_test, y_test, params, learning_rate=0.1, epochs=1000, loss_type='cross_entropy'):
    # One-Hot Encoding der Labels
    y_train_onehot = one_hot(y_train, num_classes=3)
    y_test_onehot = one_hot(y_test, num_classes=3)

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }

    for epoch in range(epochs):
        #Forward Propagation
        cache = forward(X_train, params)

        #Back Propagation
        grads, loss = backward(params, cache, y_train_onehot, loss_type)

        #Parameter Update
        params = gradient(params, grads, learning_rate)

        # Vorhersagen auf Trainingsdaten
        y_train_pred = np.argmax(cache['a3'], axis=1)
        train_acc = accuracy_score(y_train, y_train_pred)

        # Vorhersagen auf Testdaten
        cache_test = forward(X_test, params)
        y_test_pred = np.argmax(cache_test['a3'], axis=1)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Speichere Verlauf
        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        # 5. Fortschritt ausgeben
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoche {epoch:4d}/{epochs} | "
                  f"Loss: {loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f}")

    return params, history