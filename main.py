import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from functions import *
from forward_propagation import *
from back_propagation import *
from train import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


#Plotte Trainingsverlauf Aufgabe 6
def plot_history(history, loss_type='cross_entropy'):
    epochs = range(len(history['train_loss']))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].set_xlabel('Epoche')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Loss-Verlauf ({loss_type})')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Trainings Accuracy')
    axes[1].plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    axes[1].set_xlabel('Epoche')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Accuracy-Verlauf ({loss_type})')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'training_verlauf_{loss_type}.png', dpi=150)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=['1', '5', '7'], title='Confusionmatrix'): #Aufgabe 6
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='Wahre Klasse',
           xlabel='Vorhergesagte Klasse')
    th = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > th else 'black',
                    fontsize=14)

    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{title.replace(" ", "_")}.png', dpi=150)
    plt.show()

    return cm


if __name__ == "__main__":
    digits = load_digits()
    X, y = digits.data, digits.target

    # Filtern nach Klassen 1, 5, 7
    mask = np.isin(y, [1, 5, 7])
    X, y = X[mask], y[mask]

    # Labels ummappen: {1: 0, 5: 1, 7: 2}
    label_map = {1: 0, 5: 1, 7: 2}
    y = np.array([label_map[label] for label in y])

    print(f"Anzahl Samples: {len(y)}")
    print(f"Input-Dimensionen: {X.shape[1]}")

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training Samples: {len(y_train)}")
    print(f"Test Samples: {len(y_test)}")

    # Skalierung
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Training mit Cross-Entropy Loss
    print("Training mit Cross-Entropy Loss:")

    layer_dims = [64, 64, 32, 3]
    params_ce = initialize_params(layer_dims, seed=42)

    params_ce, history_ce = train_full_batch(
        X_train, y_train, X_test, y_test, params_ce,
        learning_rate=0.1, epochs=1000, loss_type='cross_entropy'
    )

    # Plots für Cross-Entropy
    plot_history(history_ce, loss_type='Cross-Entropy')

    # Finale Vorhersagen und Confusion Matrix
    cache_test = forward(X_test, params_ce)
    y_pred_ce = np.argmax(cache_test['a3'], axis=1)
    plot_confusion_matrix(y_test, y_pred_ce, title='Cross-Entropy')

    # Training mit MSE Loss
    print("Training mit MSE Loss:")

    params_mse = initialize_params(layer_dims, seed=42)

    params_mse, history_mse = train_full_batch(
        X_train, y_train, X_test, y_test, params_mse,
        learning_rate=0.5, epochs=1000, loss_type='mse'
    )

    # Plots für MSE
    plot_history(history_mse, loss_type='MSE')

    # Finale Vorhersagen und Confusion Matrix
    cache_test = forward(X_test, params_mse)
    y_pred_mse = np.argmax(cache_test['a3'], axis=1)
    plot_confusion_matrix(y_test, y_pred_mse, title='MSE')

    # Zusammenfassung
    print("Zusammenfassung:")
    print(f"Cross-Entropy - Finale Test Accuracy: {history_ce['test_acc'][-1]:.4f}")
    print(f"MSE           - Finale Test Accuracy: {history_mse['test_acc'][-1]:.4f}")
