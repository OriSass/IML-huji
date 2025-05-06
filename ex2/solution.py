import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


### HELPER FUNCTIONS ###
# Add here any helper functions which you think will be useful

def generate_data(m):
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    weights = np.array([-0.6, 0.4])
    X = np.random.multivariate_normal(mean, cov, m)
    y = np.sign(X @ weights)
    return X, y

def plot_decision_boundary(clf, X, y, f_weights, title, save_path=None):
    plt.figure()
    plt.title(title)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z_svm = clf.decision_function(grid).reshape(xx.shape)
    plt.contour(xx, yy, Z_svm, colors='k', levels=[0], linestyles=['-'])

    Z_f = grid @ f_weights
    plt.contour(xx, yy, Z_f.reshape(xx.shape), colors='g', levels=[0], linestyles=['--'])

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def generate_two_gaussians(n_samples):
    cov = np.array([[0.5, 0.2], [0.2, 0.5]])
    X1 = np.random.multivariate_normal([-1, -1], cov, n_samples // 2)
    X2 = np.random.multivariate_normal([1, 1], cov, n_samples // 2)
    X = np.vstack((X1, X2))
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    return X, y

def plot_classifier(X, y, clf, title, save_path=None):
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ['#FF0000', '#0000FF']

    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(cmap_bold), edgecolor='k')
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

### Exercise Solution ###

def pratical_1_runner(save_path=None):
    np.random.seed(0)
    C_values = [0.1, 1, 5, 10, 100]
    m_values = [5, 10, 20, 100]
    f_weights = np.array([-0.6, 0.4])

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    for m in m_values:
        X, y = generate_data(m)
        for C in C_values:
            clf = SVC(C=C, kernel='linear')
            clf.fit(X, y)
            title = f"SVM Decision (m={m}, C={C})"
            filename = f"svm_m{m}_C{C}.png" if save_path else None
            filepath = os.path.join(save_path, filename) if save_path else None
            plot_decision_boundary(clf, X, y, f_weights, title, filepath)



def practical_2_runner(save_path=None):
    np.random.seed(0)
    classifiers = {
        "SVM (λ=5)": SVC(C=1/5, kernel='rbf'),
        "Decision Tree (depth=7)": DecisionTreeClassifier(max_depth=7),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
    }

    datasets = {
        "Moons": make_moons(n_samples=200, noise=0.2),
        "Circles": make_circles(n_samples=200, noise=0.1),
        "Two Gaussians": generate_two_gaussians(200)
    }

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    for data_name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        for clf_name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            title = f"{clf_name} on {data_name} (Acc={acc:.3f})"
            filename = f"{data_name}_{clf_name}.png".replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
            filepath = os.path.join(save_path, filename) if save_path else None
            print(f"Saving: {filepath}")
            plot_classifier(X, y, clf, title, filepath)


if __name__ == "__main__":
    path = None
    pratical_1_runner(save_path="plots/runner_1/")
    practical_2_runner(save_path="plots/runner_2/")