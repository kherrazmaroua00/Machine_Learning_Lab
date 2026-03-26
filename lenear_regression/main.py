import numpy as np
import matplotlib.pyplot as plt

# -------- Step 1: Read dataset --------
def get_training_data(file_path):
    data = np.loadtxt(file_path, skiprows=1)
    X = data[:, 0].reshape(-1, 1) #
    y = data[:, 1].reshape(-1, 1)

    # add column of ones for intercept for theta_0
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    return X, y


# -------- Step 2: Initialize theta --------
def init_theta(n):
    return np.ones((n, 1))


# -------- Step 3: Gradient --------
def generate_gradient(X, theta, y):
    m = X.shape[0]
    return (1/m) * X.T.dot(X.dot(theta) - y)


# -------- Step 4: Gradient Descent --------
def gradient_descent(X, y, theta, alpha):
    J_history = []
    
    for i in range(1000):
        gradient = generate_gradient(X, theta, y)
        theta = theta - alpha * gradient
        
        # compute loss
        loss = (1/(2*X.shape[0])) * np.sum((X.dot(theta) - y)**2)
        J_history.append((i, loss))
        
    return theta, J_history


# -------- Step 5: Plot loss --------
def show_loss(J_history):
    x = [i[0] for i in J_history]
    y = [i[1] for i in J_history]
    
    plt.plot(x, y)
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.show()


# -------- Step 6: Plot regression --------
def show_regression(X, y, theta):
    plt.scatter(X[:,1], y)
    plt.plot(X[:,1], X.dot(theta))
    plt.xlabel("area")
    plt.ylabel("price")
    plt.title("Linear Regression")
    plt.show()


# -------- MAIN --------
X , y = get_training_data("Ir2_data.txt")

theta = init_theta(X.shape[1])
alpha = 0.01

theta, J_history = gradient_descent(X, y, theta, alpha)

print("Final theta:", theta)

show_loss(J_history)
show_regression(X, y, theta)