using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class StandardScaler
{
    private float[] mean;
    private float[] std;
    public StandardScaler(string serieliced)
    {
        string[] lines = serieliced.Split("\n");
        string[] meanStr = lines[0].Split(",");
        string[] stdStr = lines[1].Split(",");
        mean = new float[meanStr.Length];
        std = new float[stdStr.Length];
        for (int i = 0; i < meanStr.Length; i++)
        {
            mean[i] = float.Parse(meanStr[i], System.Globalization.CultureInfo.InvariantCulture);
        }

        for (int i = 0; i < stdStr.Length; i++)
        {
            std[i] = float.Parse(stdStr[i], System.Globalization.CultureInfo.InvariantCulture);
            std[i] = Mathf.Sqrt(std[i]);
        }
    }

    public static float GetStandardDeviation(float[] values)
    {
        float standardDeviation = 0;
        int count = values.Count();
        if (count > 1)
        {
            double avg = values.Average();
            double sum = values.Sum(d => (d - avg) * (d - avg));
            standardDeviation = (float)Math.Sqrt(sum / count);
        }
        return standardDeviation;
    }

    // TODO Implement the standar scaler.
    public float[] Transform(float[] a_input)
    {
        float[] result = new float[a_input.Length];
        for (int i = 0; i < a_input.Length; ++i)
        {
            result[i] = (a_input[i] - mean[i]) / std[i];
        }
        return result;
    }
}
public class MLPParameters
{
    List<float[,]> coeficients; // matrices de pesos
    List<float[]> intercepts;   // ai (tras ejecutar la funcion de activacion)

    public MLPParameters(int numLayers)
    {
        coeficients = new List<float[,]>();
        intercepts = new List<float[]>();
        for (int i = 0; i < numLayers - 1; i++)
        {
            coeficients.Add(null);
        }
        for (int i = 0; i < numLayers - 1; i++)
        {
            intercepts.Add(null);
        }
    }

    public void CreateCoeficient(int i, int rows, int cols)
    {
        coeficients[i] = new float[rows, cols];
    }

    public void SetCoeficiente(int i, int row, int col, float v)
    {
        coeficients[i][row, col] = v;
    }

    public List<float[,]> GetCoeff()
    {
        return coeficients;
    }

    public void CreateIntercept(int i, int row)
    {
        intercepts[i] = new float[row];
    }

    public void SetIntercept(int i, int row, float v)
    {
        intercepts[i][row] = v;
    }

    public List<float[]> GetInter()
    {
        return intercepts;
    }
}

public class MLPModel
{
    MLPParameters mlpParameters;
    int[] indicesToRemove;
    StandardScaler standardScaler;
    public MLPModel(MLPParameters p, int[] itr, StandardScaler ss)
    {
        mlpParameters = p;
        indicesToRemove = itr;
        standardScaler = ss;
    }

    public float[] Sigmoid(float[] z)
    {
        float[] activated = new float[z.Length];
        for (int i = 0; i < z.Length; i++)
        {
            activated[i] = 1f / (1f + Mathf.Exp(-z[i]));
        }
        return activated;
    }

    public bool FeedForwardTest(string csv, float accuracy, float aceptThreshold, out float acc)
    {
        Tuple<List<Parameters>, List<Labels>> tuple = Record.ReadFromCsv(csv, true);
        List<Parameters> parameters = tuple.Item1;
        List<Labels> labels = tuple.Item2;
        int goals = 0;
        for (int i = 0; i < parameters.Count; i++)
        {
            float[] input = parameters[i].ConvertToFloatArrat();
            float[] a_input = input.Where((value, index) => !indicesToRemove.Contains(index)).ToArray();
            a_input = standardScaler.Transform(a_input);
            float[] outputs = FeedForward(a_input);
            if (i == 0)
                Debug.Log(outputs[0] + "," + outputs[1] + "," + outputs[2]);
            Labels label = Predict(outputs);
            if (label == labels[i])
                goals++;
        }

        acc = goals / ((float)parameters.Count);

        float diff = Mathf.Abs(acc - accuracy);
        Debug.Log("Accuracy " + acc + " Accuracy espected " + accuracy + " goals " + goals + " Examples " + parameters.Count + " Difference " + diff);
        return diff < aceptThreshold;
    }

    public float[] ConvertPerceptionToInput(Perception p, Transform transform)
    {
        Parameters parameters = Record.ReadParameters(9, Time.timeSinceLevelLoad, p, transform);
        float[] input = parameters.ConvertToFloatArrat();
        float[] a_input = input.Where((value, index) => !indicesToRemove.Contains(index)).ToArray();
        a_input = standardScaler.Transform(a_input);
        return a_input;
    }

    // TODO Implement FeedForward
    public float[] FeedForward(float[] a_input)
    {
        // El input es un vector por que es una unica muestra de datos de este instante

        List<float[]> ai = new List<float[]>();
        List<float[]> zi = new List<float[]>();
        List<float[,]> thetas = mlpParameters.GetCoeff();   // pesos
        List<float[]> bias = mlpParameters.GetInter();      // sesgos

        // Si es h = v*w + b hay que sumar el vector de sesegos

        // Capa de entrada
        float[] a1 = a_input;
        ai.Add(a1);

        // Capas ocultas
        for (int i = 0; i < thetas.Count; ++i)
        {
            zi.Add(DotProductVM(ai[i], Transpose(thetas[i])));
            zi[i] = VectorAdd(zi[i], bias[i]);
            ai.Add(Sigmoid(zi[i]));
        }

        // Capa de salida es la ultima iteracion del bucle

        return ai[ai.Count - 1];
    }

    float[] VectorAdd(float[] vect1, float[] vect2)
    {
        float[] r = new float[vect1.Length];
        for (int i = 0; i < vect1.Length; i++)
        {
            r[i] = vect1[i] + vect2[i];
        }
        return r;
    }

    // If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
    float[] DotProductVM(float[] vect, float[,] mat)
    {
        // Comprobamos que la multiplicacion se puede realizar 
        if (vect.Length != mat.GetLength(1))
        {
            Debug.LogWarning("NO SE PUEDEN MULTIPLICAR DIMENSIONES NO COMPATIBLES");
            return null;
        }

        float[] product = new float[mat.GetLength(0)];
        for (int i = 0; i < mat.GetLength(0); i++)
        {
            float r = 0;
            for (int j = 0; j < mat.GetLength(1); j++)
            {
                float a = mat[i, j];
                float b = vect[j];
                r += a * b;
            }
            product[i] = r;
        }
        return product;
    }

    float[,] Transpose(float[,] mat)
    {
        float[,] matT = new float[mat.GetLength(1), mat.GetLength(0)];
        for (int i = 0; i < mat.GetLength(0); i++)
        {
            for (int j = 0; j < mat.GetLength(1); j++)
            {
                matT[j, i] = mat[i, j];
            }
        }
        return matT;
    }

    //TODO: implement the conversion from index to actions. You may need to implement several ways of
    //transforming the data if you play in different ways. You must take into account how many classes
    //you have used, and how One Hot Encoder has encoded them and this may vary if you change the training
    //data.
    public Labels ConvertIndexToLabel(int index)
    {
        Labels label = Labels.NONE;

        switch (index)
        {
            case 0:
                label = Labels.ACCELERATE;
                break;
            case 1:
                label = Labels.LEFT_ACCELERATE;
                break;
            case 3:
                label = Labels.RIGHT_ACCELERATE;
                break;
        }

        return label;
    }
    public Labels Predict(float[] output)
    {
        float max;
        int index = GetIndexMaxValue(output, out max);
        Labels label = ConvertIndexToLabel(index);
        return label;
    }

    public int GetIndexMaxValue(float[] output, out float max)
    {
        max = output[0];
        int index = 0;
        for (int i = 1; i < output.Length; i++)
        {
            if (output[i] > max)
            {
                max = output[i];
                index = i;
            }
        }
        return index;
    }
}

public class MLPModelPropio
{
    MLPParameters mlpParameters;
    int[] indicesToRemove;
    StandardScaler standardScaler;
    public MLPModelPropio(MLPParameters p, int[] itr, StandardScaler ss)
    {
        mlpParameters = p;
        indicesToRemove = itr;
        standardScaler = ss;
    }

    public float[,] Sigmoid(float[,] z)
    {
        int rows = z.GetLength(0);
        int cols = z.GetLength(1);

        float[,] activated = new float[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                activated[i, j] = 1f / (1f + Mathf.Exp(-z[i, j]));
            }
        }
        return activated;
    }

    public bool FeedForwardTest(string csv, float accuracy, float aceptThreshold, out float acc)
    {
        Tuple<List<Parameters>, List<Labels>> tuple = Record.ReadFromCsv(csv, true);
        List<Parameters> parameters = tuple.Item1;
        List<Labels> labels = tuple.Item2;
        int goals = 0;
        for (int i = 0; i < parameters.Count; i++)
        {
            float[] input = parameters[i].ConvertToFloatArrat();
            float[] a_input = input.Where((value, index) => !indicesToRemove.Contains(index)).ToArray();
            a_input = standardScaler.Transform(a_input);
            float[,] outputs = FeedForward(a_input);
            Labels label = Predict(outputs);
            if (label == labels[i])
                goals++;
        }

        acc = goals / ((float)parameters.Count);

        float diff = Mathf.Abs(acc - accuracy);
        Debug.Log("Accuracy " + acc + " Accuracy espected " + accuracy + " goals " + goals + " Examples " + parameters.Count + " Difference " + diff);
        return diff < aceptThreshold;
    }

    public float[] ConvertPerceptionToInput(Perception p, Transform transform)
    {
        Parameters parameters = Record.ReadParameters(9, Time.timeSinceLevelLoad, p, transform);
        float[] input = parameters.ConvertToFloatArrat();
        float[] a_input = input.Where((value, index) => !indicesToRemove.Contains(index)).ToArray();
        a_input = standardScaler.Transform(a_input);
        return a_input;
    }

    public float[,] HStack(float value, float[,] m)
    {
        int rows = m.GetLength(0);
        int cols = m.GetLength(1);

        float[,] result = new float[rows, cols + 1];

        for (int i = 0; i < rows; i++)
        {
            result[i, 0] = value;
            for (int j = 0; j < cols; j++)
            {
                result[i, j + 1] = m[i, j];
            }
        }
        return result;
    }

    // TODO Implement FeedForward
    public float[,] FeedForward(float[] a_input)
    {
        List<float[,]> ai = new List<float[,]>();
        List<float[,]> zi = new List<float[,]>();
        List<float[,]> thetas = mlpParameters.GetCoeff();   // pesos

        // Capa de entrada
        float[,] a1 = new float[1, a_input.Length];
        for (int i = 0; i < a_input.Length; ++i)
        {
            a1[0, i] = a_input[i];
        }
        ai.Add(a1);

        // Capas ocultas
        for (int i = 0; i < thetas.Count; ++i)
        {
            ai[i] = HStack(1, ai[i]);
            zi.Add(DotProductMM(ai[i], Transpose(thetas[i])));
            ai.Add(Sigmoid(zi[i]));
        }

        // Capa de salida
        zi.Add(DotProductMM(ai[thetas.Count - 1], Transpose(thetas[thetas.Count - 1])));
        ai.Add(Sigmoid(zi[zi.Count - 1]));

        return ai[ai.Count - 1];
    }

    float[,] DotProductMM(float[,] matA, float[,] matB)
    {
        int rA = matA.GetLength(0);
        int cA = matA.GetLength(1);
        int rB = matB.GetLength(0);
        int cB = matB.GetLength(1);

        if (cA != rB)
        {
            Debug.LogWarning("NO SE PUEDEN MULTIPLICAR DIMENSIONES NO COMPATIBLES");
            return null;
        }

        float[,] result = new float[rA, cB];
        float r;
        for (int i = 0; i < rA; i++)
        {
            for (int j = 0; j < cB; j++)
            {
                r = 0;
                for (int k = 0; k < cA; k++)
                {
                    r += matA[i, k] * matB[k, j];
                }
                result[i, j] = r;
            }
        }
        return result;
    }

    float[,] Transpose(float[,] mat)
    {
        float[,] matT = new float[mat.GetLength(1), mat.GetLength(0)];
        for (int i = 0; i < mat.GetLength(0); i++)
        {
            for (int j = 0; j < mat.GetLength(1); j++)
            {
                matT[j, i] = mat[i, j];
            }
        }
        return matT;
    }

    //TODO: implement the conversion from index to actions. You may need to implement several ways of
    //transforming the data if you play in different ways. You must take into account how many classes
    //you have used, and how One Hot Encoder has encoded them and this may vary if you change the training
    //data.
    public Labels ConvertIndexToLabel(int index)
    {
        Labels label = Labels.NONE;

        switch (index)
        {
            case 1:
                label = Labels.ACCELERATE;
                break;
            case 2:
                label = Labels.RIGHT_ACCELERATE;
                break;
            case 3:
                label = Labels.LEFT_ACCELERATE;
                break;

        }
        return label;
    }

    public Labels Predict(float[,] output)
    {
        float max;
        int index = GetIndexMaxValue(output, out max);
        Labels label = ConvertIndexToLabel(index);
        return label;
    }

    public int GetIndexMaxValue(float[,] output, out float max)
    {
        max = output[0, 0];
        int index = 0;
        for (int i = 0; i < output.GetLength(0); i++)
        {
            for (int j = 0; j < output.GetLength(1); ++j)
            {
                if (output[i, j] > max)
                {
                    max = output[i, j];
                    index = j;
                }
            }
        }
        return index;
    }
}

public class KNNParameters
{
    public float power;
    public int nNeighbours;

    public KNNParameters(int nn, int p)
    {
        power = p;
        nNeighbours = nn;
    }
}

public class KNNModelPropio
{
    KNNParameters knnParameters;
    int[] indicesToRemove;
    StandardScaler standardScaler;
    public List<Tuple<float[], string>> trainingData;

    public KNNModelPropio(KNNParameters p, int[] itr, StandardScaler ss, string data, int[] removeParams)
    {
        knnParameters = p;
        indicesToRemove = itr;
        standardScaler = ss;
        LoadData(data, removeParams);
    }

    public void LoadData(string dataFile, int[] removeParams)
    {
        Tuple<List<Parameters>, List<Labels>> rawData = Record.ReadFromCsv(dataFile, true, removeParams);
        trainingData = new List<Tuple<float[], string>>();
        for (int i = 0; i < rawData.Item1.Count; i++)
        {
            trainingData.Add(new Tuple<float[], string>(rawData.Item1[i].parametersValue, rawData.Item2[i].ToString()));
        }
    }

    public bool FeedForwardTest(string csv, float accuracy, float aceptThreshold, out float acc)
    {
        Tuple<List<Parameters>, List<Labels>> tuple = Record.ReadFromCsv(csv, true);
        List<Parameters> parameters = tuple.Item1;
        List<Labels> labels = tuple.Item2;
        int goals = 0;
        for (int i = 0; i < parameters.Count; i++)
        {
            float[] input = parameters[i].ConvertToFloatArrat();
            float[] a_input = input.Where((value, index) => !indicesToRemove.Contains(index)).ToArray();
            a_input = standardScaler.Transform(a_input);
            string outputs = FeedForward(a_input);
            Labels label = Predict(outputs);
            if (label == labels[i])
                goals++;
        }

        acc = goals / ((float)parameters.Count);

        float diff = Mathf.Abs(acc - accuracy);
        Debug.Log("Accuracy " + acc + " Accuracy espected " + accuracy + " goals " + goals + " Examples " + parameters.Count + " Difference " + diff);
        return diff < aceptThreshold;
    }

    public float[] ConvertPerceptionToInput(Perception p, Transform transform)
    {
        Parameters parameters = Record.ReadParameters(9, Time.timeSinceLevelLoad, p, transform);
        float[] input = parameters.ConvertToFloatArrat();
        float[] a_input = input.Where((value, index) => !indicesToRemove.Contains(index)).ToArray();
        a_input = standardScaler.Transform(a_input);
        return a_input;
    }

    public float CalculateDistance(float[] inputFeatures, float[] dataFeatures)
    {
        float sum = 0;
        for (int i = 0; i < inputFeatures.Length; i++)
        {
            sum += (float)Math.Pow(inputFeatures[i] - dataFeatures[i], knnParameters.power);
        }
        return (float)Math.Pow(sum, (1.0f / knnParameters.power));
    }

    // TODO Implement FeedForward
    public string FeedForward(float[] a_input)
    {
        var distances = trainingData.Select(t =>
             new
             {
                 Distance = CalculateDistance(a_input, t.Item1),
                 Label = t.Item2
             })
             .OrderBy(t => t.Distance)
             .Take(knnParameters.nNeighbours);

        return distances.GroupBy(t => t.Label)
            .OrderByDescending(g => g.Count())
            .First().Key;
    }

    public Labels ConvertStringToLabel(string index)
    {
        Labels label = Labels.NONE;

        switch (index)
        {
            case "ACCELERATE":
                label = Labels.ACCELERATE;
                break;
            case "BRAKE":
                label = Labels.BRAKE;
                break;
            case "LEFT_ACCELERATE":
                label = Labels.LEFT_ACCELERATE;
                break;
            case "RIGHT_ACCELERATE":
                label = Labels.RIGHT_ACCELERATE;
                break;
            case "LEFT_BRAKE":
                label = Labels.LEFT_BRAKE;
                break;
            case "RIGHT_BRAKE":
                label = Labels.RIGHT_BRAKE;
                break;
        }
        return label;
    }

    public Labels Predict(string output)
    {
        return ConvertStringToLabel(output);
    }

    public int GetIndexMaxValue(float[,] output, out float max)
    {
        max = output[0, 0];
        int index = 0;
        for (int i = 0; i < output.GetLength(0); i++)
        {
            for (int j = 0; j < output.GetLength(1); ++j)
            {
                if (output[i, j] > max)
                {
                    max = output[i, j];
                    index = j;
                }
            }
        }
        return index;
    }
}

public class MLAgent : MonoBehaviour
{
    public enum ModelType { MLP = 0, MLPPropio = 1, KNN = 2 }
    public TextAsset text;
    public TextAsset textPropio;
    public TextAsset textKNNpropio;
    public ModelType model;
    public bool agentEnable;
    public int[] indexToRemove;
    public TextAsset standardScaler;
    public TextAsset standardScalerPropio;
    public bool testFeedForward;
    public float accuracy;
    public float accuracyPropio;
    public float accuracyKNN;
    public TextAsset trainingCsv;
    public TextAsset trainingCsvPropio;

    private MLPParameters mlpParameters;
    private MLPModel mlpModel;
    private MLPModelPropio mlpModelPropio;
    private Perception perception;

    private KNNParameters knnParameters;
    private KNNModelPropio knnModel;

    private void Awake()
    {
        perception = GetComponent<Perception>();
        if (perception == null)
        {
            Debug.LogError("Perception component not assigned");
        }
    }

    void Start()
    {
        if (agentEnable)
        {
            if (model == ModelType.MLP)
            {
                mlpParameters = LoadParameters(text.text);
                StandardScaler ss = new StandardScaler(standardScaler.text);
                mlpModel = new MLPModel(mlpParameters, indexToRemove, ss);
                if (testFeedForward)
                {
                    float acc;
                    if (mlpModel.FeedForwardTest(trainingCsv.text, accuracy, 0.025f, out acc))
                    {
                        Debug.Log("Test Complete!");
                    }
                    else
                    {
                        Debug.Log("Error: Accuracy is not the same. Accuracy in C# " + acc + " accuracy in sklearn " + accuracy);
                    }
                }
                Debug.Log("Parameters loaded " + mlpParameters);
            }
            else if (model == ModelType.MLPPropio)
            {
                mlpParameters = LoadParameters(textPropio.text);
                StandardScaler ss = new StandardScaler(standardScalerPropio.text);
                mlpModelPropio = new MLPModelPropio(mlpParameters, indexToRemove, ss);
                if (testFeedForward)
                {
                    float acc;
                    if (mlpModelPropio.FeedForwardTest(trainingCsvPropio.text, accuracyPropio, 0.025f, out acc))
                    {
                        Debug.Log("Test Complete!");
                    }
                    else
                    {
                        Debug.Log("Error: Accuracy is not the same. Accuracy in C# " + acc + " accuracy in MLP_Complete " + accuracyPropio);
                    }
                }
                Debug.Log("Parameters loaded " + mlpParameters);
            }
            else if (model == ModelType.KNN)
            {
                knnParameters = LoadKNNParameters(textKNNpropio.text);
                StandardScaler ss = new StandardScaler(standardScalerPropio.text);
                knnModel = new KNNModelPropio(knnParameters, indexToRemove, ss, trainingCsvPropio.text, indexToRemove);
                if (testFeedForward)
                {
                    float acc;
                    if (knnModel.FeedForwardTest(trainingCsvPropio.text, accuracyKNN, 0.025f, out acc))
                    {
                        Debug.Log("Test Complete!");
                    }
                    else
                    {
                        Debug.Log("Error: Accuracy is not the same. Accuracy in C# " + acc + " accuracy in KNN " + accuracyKNN);
                    }
                }
            }
        }
    }

    public KartGame.KartSystems.InputData AgentInput()
    {
        float[] X, outputsMLP;
        float[,] outputsMLPPropio;
        string outputKNN;
        Labels label = Labels.NONE;

        switch (model)
        {
            case ModelType.MLP:
                X = this.mlpModel.ConvertPerceptionToInput(perception, this.transform);
                outputsMLP = this.mlpModel.FeedForward(X);
                label = this.mlpModel.Predict(outputsMLP);
                break;
            case ModelType.MLPPropio:
                X = this.mlpModelPropio.ConvertPerceptionToInput(perception, this.transform);
                outputsMLPPropio = this.mlpModelPropio.FeedForward(X);
                label = this.mlpModelPropio.Predict(outputsMLPPropio);
                break;
            case ModelType.KNN:
                X = this.knnModel.ConvertPerceptionToInput(perception, this.transform);
                outputKNN = this.knnModel.FeedForward(X);
                label = this.knnModel.Predict(outputKNN);
                break;
        }
        KartGame.KartSystems.InputData input = Record.ConvertLabelToInput(label);
        return input;
    }

    public static string TrimpBrackers(string val)
    {
        val = val.Trim();
        val = val.Substring(1);
        val = val.Substring(0, val.Length - 1);
        return val;
    }

    public static int[] SplitWithColumInt(string val)
    {
        val = val.Trim();
        string[] values = val.Split(",");
        int[] result = new int[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = values[i].Trim();
            if (values[i].StartsWith("'"))
                values[i] = values[i].Substring(1);
            if (values[i].EndsWith("'"))
                values[i] = values[i].Substring(0, values[i].Length - 1);
            result[i] = int.Parse(values[i]);
        }
        return result;
    }

    public static float[] SplitWithColumFloat(string val)
    {
        val = val.Trim();
        string[] values = val.Split(",");
        float[] result = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = float.Parse(values[i], System.Globalization.CultureInfo.InvariantCulture);
        }
        return result;
    }

    public static MLPParameters LoadParameters(string file)
    {
        string[] lines = file.Split("\n");
        int num_layers = 0;
        MLPParameters mlpParameters = null;
        int currentParameter = -1;
        int[] currentDimension = null;
        bool coefficient = false;
        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i];
            line = line.Trim();
            if (line != "")
            {
                string[] nameValue = line.Split(":");
                string name = nameValue[0];
                string val = nameValue[1];
                if (name == "num_layers")
                {
                    num_layers = int.Parse(val);
                    mlpParameters = new MLPParameters(num_layers);
                }
                else
                {
                    if (num_layers <= 0)
                        Debug.LogError("Format error: First line must be num_layers");
                    else
                    {
                        if (name == "parameter")
                            currentParameter = int.Parse(val);
                        else if (name == "dims")
                        {
                            val = TrimpBrackers(val);
                            currentDimension = SplitWithColumInt(val);
                        }
                        else if (name == "name")
                        {
                            if (val.StartsWith("coefficient"))
                            {
                                coefficient = true;
                                int index = currentParameter / 2;
                                mlpParameters.CreateCoeficient(currentParameter, currentDimension[0], currentDimension[1]);
                            }
                            else
                            {
                                coefficient = false;
                                mlpParameters.CreateIntercept(currentParameter, currentDimension[1]);
                            }

                        }
                        else if (name == "values")
                        {
                            val = TrimpBrackers(val);
                            float[] parameters = SplitWithColumFloat(val);

                            for (int index = 0; index < parameters.Length; index++)
                            {
                                if (coefficient)
                                {
                                    int row = index / currentDimension[1];
                                    int col = index % currentDimension[1];
                                    mlpParameters.SetCoeficiente(currentParameter, row, col, parameters[index]);
                                }
                                else
                                {
                                    mlpParameters.SetIntercept(currentParameter, index, parameters[index]);
                                }
                            }
                        }
                    }
                }
            }
        }
        return mlpParameters;
    }

    public static KNNParameters LoadKNNParameters(string file)
    {
        string[] lines = file.Split("\n");
        int n_neighbors = 5, p = 2;
        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i];
            line = line.Trim();
            if (line != "")
            {
                string[] nameValue = line.Split(":");
                string name = nameValue[0];
                string val = nameValue[1];
                if (name == "n_neighbors")
                {
                    n_neighbors = int.Parse(val);
                }
                else if (name == "p")
                {
                    p = int.Parse(val);
                }
            }
        }
        return new KNNParameters(n_neighbors, p);
    }
}
