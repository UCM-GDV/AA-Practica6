import numpy as np
import pandas as pd
from skl2onnx import to_onnx
from onnx2json import convert
import pickle
import json
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.metrics import accuracy_score

def cleanData(data, x_columns, y_column):
    for x_column in x_columns:
        data[x_column] = data[x_column].apply(lambda x:  str(x).replace('.', '', x.count('.') - 1))
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

def one_hot_encoding(Y,cat='auto'):
    oneHotEncoder=OneHotEncoder(categories=cat)
    YEnc=oneHotEncoder.fit_transform(Y.reshape(-1,1)).toarray()
    return YEnc

def label_hot_encoding(Y, cat):
    labelEncoder=LabelEncoder()
    labelEncoder.fit(cat)
    YEnc = labelEncoder.transform(Y)
    return YEnc


def accuracy(P,Y):
    return accuracy_score(P,Y)

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
