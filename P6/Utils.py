import numpy as np
import pandas as pd
from skl2onnx import to_onnx
from onnx2json import convert
import pickle
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

def cleanData(data, x_columns, y_column):
    for x_column in x_columns:
        data[x_column] = data[x_column].astype(np.float64)
    data = data.drop(data[data[y_column] == "NONE"].index)
    return data

def load_data_csv(path,x_columns,y_column):
    data = pd.read_csv(path)
    data = cleanData(data, x_columns, y_column)
    xi = []
    for x_column in x_columns:
        xi.append(data[x_column].to_numpy())
    X = np.array(xi)
    y = data[y_column]
    return data, X, y

def one_hot_encoding(Y,categories_='auto'):
    oneHotEncoder=OneHotEncoder(categories=categories_)
    YEnc=oneHotEncoder.fit_transform(Y.reshape(-1,1)).toarray()
    return YEnc

def accuracy(P,Y):
    return accuracy_score(P,Y)

def drawConfusionMatrix(matrix,xclasses,yclasses,filename,title="Confusion Matrix",xlabel="Predictions",ylabel="True label"):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)
    plt.title(title)
    ax.set_xticks(np.arange(len(xclasses)))
    ax.set_yticks(np.arange(len(yclasses)))
    ax.set_xticklabels(xclasses,rotation=45,ha="right",rotation_mode="anchor")
    ax.set_yticklabels(yclasses)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _annotate_heatmap(im, matrix)
    ax.figure.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def _annotate_heatmap(im, data=None, valfmt="{x:.0f}", textcolors=("white", "black"), threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = tck.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def calculateConfusionMatrix(classes, predictions, y_true, displayClass=0):
    num_classes = len(classes[0])
    matrix = np.zeros((num_classes, num_classes))
    for i in range(len(predictions)):
        if (predictions[i] == y_true[i]):
            matrix[y_true[i]][y_true[i]] += 1
        else:
            matrix[y_true[i]][predictions[i]] += 1
    return matrix.reshape((num_classes,num_classes))

def ExportONNX_JSON_TO_Custom(onnx_json,mlp):
    graphDic = onnx_json["graph"]
    initializer = graphDic["initializer"]
    s= "num_layers:"+str(mlp.n_layers_)+"\n"
    index = 0
    parameterIndex = 0
    for parameter in initializer:
        s += "parameter:"+str(parameterIndex)+"\n"
        print(parameter["dims"])
        s += "dims:"+str(parameter["dims"])+"\n"
        print(parameter["name"])
        s += "name:"+str(parameter["name"])+"\n"
        print(parameter["doubleData"])
        s += "values:"+str(parameter["doubleData"])+"\n"
        index = index + 1
        parameterIndex = index // 2
    return s

def ExportAllformatsMLPSKlearn(mlp,X,picklefileName,onixFileName,jsonFileName,customFileName):
    with open(picklefileName,'wb') as f:
        pickle.dump(mlp,f)
    
    onx = to_onnx(mlp, X[:1])
    with open(onixFileName, "wb") as f:
        f.write(onx.SerializeToString())
    
    onnx_json = convert(input_onnx_file_path=onixFileName,output_json_path=jsonFileName,json_indent=2)
    
    customFormat = ExportONNX_JSON_TO_Custom(onnx_json,mlp)
    with open(customFileName, 'w') as f:
        f.write(customFormat)
        
def export_to_json(model, filename):
    model_dict = {"num_layers": len(model.coefs_)}
    parameters = []

    for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
        parameter = {
            "parameter": i,
            "coefficient": {
                "dims": list(coef.shape),
                "values": coef.flatten().tolist()
            },
            "intercepts": {
                "dims": [1, len(intercept)],
                "values": intercept.tolist()
            }
        }
        parameters.append(parameter)

    model_dict["parameters"] = parameters

    with open(filename, 'w') as f:
        json.dump(model_dict, f)

def export_to_txt(model, filename):
    with open(filename, 'w') as f:
        num_layers = len(model.coefs_) + 1
        f.write(f"num_layers:{num_layers}\n")

        parameter_num = 0
        for _, (coefs, intercepts) in enumerate(zip(model.coefs_, model.intercepts_)):
            for param_type, param_values in [('coefficient', coefs), ('intercepts', intercepts)]:
                dims = list(map(str, reversed(param_values.shape)))
                f.write(f"parameter:{parameter_num}\n")
                f.write(f"dims:{dims}\n")
                f.write(f"name:{param_type}\n")
                f.write(f"values:{param_values.flatten().tolist()}\n")
            parameter_num += 1

def export_to_txt_custom(model, filename):
    with open(filename, 'w') as f:
        num_layers = len(model.thetas) + 1
        f.write(f"num_layers:{num_layers}\n")

        parameter_num = 0
        for _, (coefs) in enumerate(zip(model.thetas)):
            for param_type, param_values in [('coefficient', coefs)]:
                dims = list(map(str, param_values[0].shape))
                f.write(f"parameter:{parameter_num}\n")
                f.write(f"dims:{dims}\n")
                f.write(f"name:{param_type}\n")
                f.write(f"values:{param_values[0].flatten().tolist()}\n")
            parameter_num += 1

def WriteStandardScaler(file,mean,var):
    line = ""
    for i in range(0,len(mean)-1):
        line = line + str(mean[i]) + ","
    line = line + str(mean[len(mean)-1])+ "\n"
    for i in range(0,len(var)-1):
        line = line + str(var[i]) + ","
    line = line + str(var[len(var)-1])+ "\n"
    with open(file, 'w') as f:
        f.write(line)
        f.close()