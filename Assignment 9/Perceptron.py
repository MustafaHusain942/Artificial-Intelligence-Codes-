import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression

# Sample dataset
X = np.array([
    [0, 0, 1, 0],  # free down
    [1, 0, 1, 0],  # free down
    [0, 1, 1, 0],  # free up
    [1, 1, 1, 1],  # blocked all sides
    [0, 0, 0, 1],  # free up and down
    [0, 1, 0, 0],  # free up, right
])

y_binary = np.array([1, 1, 1, 0, 1, 1])  # Move or not
y_multi = np.array([1, 1, 0, -1, 0, 3])   # Direction: Up = 0, Down = 1, Left = 2, Right = 3

# Train Perceptron (binary classification)
perceptron = Perceptron()
perceptron.fit(X, y_binary)

# Train multi-category model (direction prediction)
X_multi = X[y_multi != -1]        # Filter out rows with no direction
y_multi_filtered = y_multi[y_multi != -1]
multi_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
multi_model.fit(X_multi, y_multi_filtered)

# Test input (robot senses obstacles)
test_input = np.array([[0, 0, 1, 0]])  # Obstacle Left

# Predict movement (1 = move, 0 = stop)
move_decision = perceptron.predict(test_input)[0]

if move_decision == 1:
    direction = multi_model.predict(test_input)[0]
    direction_map = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
    print("Decision: MOVE")
    print("Direction:", direction_map[direction])
else:
    print("Decision: STOP")