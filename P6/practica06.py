import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from Utils import load_data_csv,one_hot_encoding,accuracy
from MLP_Complete import MLP_Complete

class Labels:
    NONE = 0
    ACCELERATE = 1
    BRAKE = 2
    LEFT_ACCELERATE = 3
    RIGHT_ACCELERATE = 4
    LEFT_BRAKE = 5
    RIGHT_BRAKE = 6

label_mapping = {
    0: "NONE",
    1: "ACCELERATE",
    2: "BRAKE",
    3: "LEFT_ACCELERATE",
    4: "RIGHT_ACCELERATE",
    5: "LEFT_BRAKE",
    6: "RIGHT_BRAKE"
}

# Ejercicio 2
x_columns = ["ray1", "ray2", "ray3", "ray4", "ray5", "kartx", "karty", "kartz", "time"]
gameData, X, y = load_data_csv("KartData.csv", x_columns, "action")
#display(gameData) # Para el notebook

df1 = pd.DataFrame(gameData, columns=x_columns)

scaling = StandardScaler()
scaling.fit(df1)
scaled_data = scaling.transform(df1)

principal = PCA(n_components=3)
principal.fit(scaled_data)
x = principal.transform(scaled_data)

feature_weights_mapping = {}
for i, component in enumerate(principal.components_):
    component_feature_weights = zip(x_columns, component)
    sorted_feature_weight = sorted(component_feature_weights, key=lambda x: abs(x[1]), reverse=True)
    feature_weights_mapping[i] = sorted_feature_weight

# En nuestro dataset, el número de clases es 3
# porque en ningún momento deceleramos (BRAKE)
label1 = feature_weights_mapping[0][0][0]
label2 = feature_weights_mapping[1][0][0]
label3 = feature_weights_mapping[2][0][0]

left_accelerate_pc1 = gameData[label1][y == "LEFT_ACCELERATE"]
right_accelerate_pc1 = gameData[label1][y == "RIGHT_ACCELERATE"]
accelerate_pc1 = gameData[label1][y == "ACCELERATE"]
left_accelerate_pc2 = gameData[label2][y == "LEFT_ACCELERATE"]
right_accelerate_pc2 = gameData[label2][y == "RIGHT_ACCELERATE"]
accelerate_pc2 = gameData[label2][y == "ACCELERATE"]
left_accelerate_pc3 = gameData[label3][y == "LEFT_ACCELERATE"]
right_accelerate_pc3 = gameData[label3][y == "RIGHT_ACCELERATE"]
accelerate_pc3 = gameData[label3][y == "ACCELERATE"]

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(projection='3d')
ax.scatter(left_accelerate_pc1, left_accelerate_pc2, left_accelerate_pc3, c='red', marker='X', s=15)
ax.scatter(right_accelerate_pc1, right_accelerate_pc2, right_accelerate_pc3, c='green', marker='o', s=15)
ax.scatter(accelerate_pc1, accelerate_pc2, accelerate_pc3, c='blue', marker='*', s=15)
ax.set_xlabel(label1)
ax.set_ylabel(label2)
ax.set_zlabel(label3)

plt.savefig("ejercicio2.png")
plt.show()

# Ejercicio 4 
x_columns = ["ray1", "ray2", "ray3", "ray4", "ray5", "kartx", "karty", "kartz", "time"]
kartData, X, y = load_data_csv("KartData.csv", x_columns, "action")

# Normalizar los datos
scaler = StandardScaler()
X = scaler.fit_transform(X).T

# Split de los datos 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.33)
yEnc = one_hot_encoding(np.array(y_train))

alpha_ = 1.0
lambda_ = 0.0

# Perceptron multicapa de implementacion propia 
mlpc = MLP_Complete(X_train.shape[1],[9],yEnc.shape[1])
Jhistory = mlpc.backpropagation(X_train,yEnc,alpha_,lambda_,2000)
a1,ai,zi = mlpc.feedforward(X_test)
y_pred_mlp = mlpc.predict(ai[len(ai) - 1])
y_pred_mlp_labels = [label_mapping[label] for label in y_pred_mlp]

# Perceptron multicapa de SKlearn
mlp = MLPClassifier(activation='logistic',alpha=lambda_,learning_rate='constant',learning_rate_init=alpha_,max_iter=2000,random_state=0,epsilon=0.12)
mlp.fit(X_train,y_train)
y_pred_sklearn = mlp.predict(X_test)

# Comprobacion de resultados
accuracy_mlpc = accuracy(y_test,y_pred_mlp_labels)
print("MLP accuracy: " + str(accuracy_mlpc))
accuracy_sklearn = accuracy(y_test,y_pred_sklearn)
print("SKlearn accuracy: " + str(accuracy_sklearn))
