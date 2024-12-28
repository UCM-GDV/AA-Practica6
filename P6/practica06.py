import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from Utils import load_data_csv,one_hot_encoding,accuracy,label_hot_encoding
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
    0 : "NONE",
    1: "ACCELERATE",
    2: "BRAKE",
    3: "LEFT_ACCELERATE",
    4: "RIGHT_ACCELERATE",
    5: "LEFT_BRAKE",
    6: "RIGHT_BRAKE",
    
}
label_array = ["NONE","ACCELERATE","BRAKE","LEFT_ACCELERATE","RIGHT_ACCELERATE","LEFT_BRAKE","RIGHT_BRAKE"]

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
#plt.show()

# Ejercicio 4 
x_columns = ["ray1", "ray2", "ray3", "ray4", "ray5", "kartx", "karty", "kartz", "time"]
kartData, X, y = load_data_csv("KartData.csv", x_columns, "action")

# Normalizar los datos
scaler = StandardScaler()
X = scaler.fit_transform(X).T

# Split de los datos 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
yEnc = one_hot_encoding(np.array(y_train),[label_array])


# Hiperparametros para 2 capas ocultas
alpha_2 = 0.665 #learning rate
lambda_2 = 0.056
numiters_2 = 2200
hidden_layers_sizes_2 = [9,7]
# Perceptron propio con mas de 3 capas
mlpc_2 = MLP_Complete(X_train.shape[1],hidden_layers_sizes_2,yEnc.shape[1])
Jhistory_2 = mlpc_2.backpropagation(X_train,yEnc,alpha_2,lambda_2,numiters_2)
a1_2,ai_2,zi_2 = mlpc_2.feedforward(X_test)
y_pred_mlp_2 = mlpc_2.predict(ai_2[len(ai_2) - 1])
y_pred_mlp_labels_2 = [label_mapping[label] for label in y_pred_mlp_2]

# Comprobacion de resultados
accuracy_mlpc_2 = accuracy(y_test,y_pred_mlp_labels_2)
print("MLP 2 hidden layers accuracy: " + str(accuracy_mlpc_2))


alpha_ = 0.665 #learning rate
lambda_ = 0.056
numiters_ = 2200
hidden_layers_sizes = 7

# Perceptron multicapa de implementacion propia 
mlpc = MLP_Complete(X_train.shape[1],[hidden_layers_sizes],yEnc.shape[1])
Jhistory = mlpc.backpropagation(X_train,yEnc,alpha_,lambda_,numiters_)
a1,ai,zi = mlpc.feedforward(X_test)
y_pred_mlp = mlpc.predict(ai[len(ai) - 1])
y_pred_mlp_labels = [label_mapping[label] for label in y_pred_mlp]

# Perceptron multicapa de SKlearn
mlp = MLPClassifier(hidden_layer_sizes=(hidden_layers_sizes,),activation='logistic',alpha=lambda_,learning_rate='constant',learning_rate_init=alpha_,max_iter=numiters_,random_state=0,epsilon=0.12)
mlp.fit(X_train,y_train)
y_pred_sklearn = mlp.predict(X_test)

# Comprobacion de resultados para una unica capa oculta 
accuracy_mlpc = accuracy(y_test,y_pred_mlp_labels)
print("MLP accuracy: " + str(accuracy_mlpc))
accuracy_sklearn = accuracy(y_test,y_pred_sklearn)
print("SKlearn accuracy: " + str(accuracy_sklearn))

# SKLearn con distintos parametros
skalpha = 1.0
sklearningrate=0.25
skiters = 1850
mlp_2 = MLPClassifier(hidden_layer_sizes=tuple(hidden_layers_sizes_2),activation='relu',alpha=skalpha,learning_rate='adaptive',solver='sgd',learning_rate_init=sklearningrate,max_iter=skiters,random_state=0,epsilon=0.12)
mlp_2.fit(X_train,y_train)
y_pred_sklearn_2 = mlp_2.predict(X_test)

# Comprobacion de resultados
accuracy_sklearn_2 = accuracy(y_test,y_pred_sklearn_2)
print("SKlearn modified accuracy: " + str(accuracy_sklearn_2))

# Modelo KNN
knn = KNeighborsClassifier(n_neighbors=7,leaf_size=9)
knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)

# Comprobacion de resultados
accuracy_knn = accuracy(y_test,y_pred_knn)
print("KNN accuracy: " + str(accuracy_knn))

# Modelo de arbol de decision 
decisiontree = DecisionTreeClassifier(criterion='gini',splitter='best',max_leaf_nodes=18,max_features=9)
decisiontree.fit(X_train,y_train)
y_pred_tree= decisiontree.predict(X_test)

# Comprobacion de resultados
accuracy_decisontree = accuracy(y_test,y_pred_tree)
print("Decision Tree accuracy: " + str(accuracy_decisontree))

# Modelo Random Forest
randomforest = RandomForestClassifier(n_estimators=80,max_depth=9,max_leaf_nodes=9)
randomforest.fit(X_train,y_train)
y_pred_forest = randomforest.predict(X_test)

# Comprobacion de resultados
accuracy_forest = accuracy(y_test,y_pred_forest)
print("Random Forest  accuracy: " + str(accuracy_forest))


# Matrices de Confusion
