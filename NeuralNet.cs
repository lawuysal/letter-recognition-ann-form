using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace letter_recognition_ann_form
{
    public class NeuralNet
    {
        public static double[,] testInput = new double[7, 5];
        public static List<Panel> panels = new List<Panel>();
        static private double[,] weightsInputHidden; 
        static private double[,] weightsHiddenOutput; 
        static private double[] biasesHidden; 
        static private double[] biasOutput; 
        private double learningRate = 0.1; 

        public NeuralNet(int inputNodes, int hiddenNodes, int outputNodes)
        {
            weightsInputHidden = new double[inputNodes, hiddenNodes];
            weightsHiddenOutput = new double[hiddenNodes, outputNodes];
            biasesHidden = new double[hiddenNodes];
            biasOutput = new double[outputNodes];
            Random rnd = new Random();

            for (int i = 0; i < inputNodes; i++)
            {
                for (int j = 0; j < hiddenNodes; j++)
                {
                    weightsInputHidden[i, j] = rnd.NextDouble() - 0.5;
                }
            }

            for (int i = 0; i < hiddenNodes; i++)
            {
                for (int j = 0; j < outputNodes; j++)
                {
                    weightsHiddenOutput[i, j] = rnd.NextDouble() - 0.5;
                }
                biasesHidden[i] = rnd.NextDouble() - 0.5;
            }

            for (int i = 0; i < outputNodes; i++)
            {
                biasOutput[i] = rnd.NextDouble() - 0.5;
            }
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double SigmoidDerivative(double x)
        {
            double sigmoid = Sigmoid(x);
            return sigmoid * (1.0 - sigmoid);
        }

        public double[] FeedForward(double[,] inputs)
        {
            int inputNodes = inputs.GetLength(0);
            int cols = inputs.GetLength(1);
            int outputNodes = biasOutput.Length;
            double[] hiddenInputs = new double[biasesHidden.Length];
            double[] hiddenOutputs = new double[biasesHidden.Length];
            double[] output = new double[outputNodes];

            // Hidden layer calculations
            for (int i = 0; i < biasesHidden.Length; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < inputNodes; j++)
                {
                    for (int k = 0; k < cols; k++)
                    {
                        sum += inputs[j, k] * weightsInputHidden[j, i];
                    }
                }
                sum += biasesHidden[i];
                hiddenInputs[i] = sum;
                hiddenOutputs[i] = Sigmoid(sum);
            }

            // Output layer calculations
            for (int i = 0; i < outputNodes; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < biasesHidden.Length; j++)
                {
                    sum += hiddenOutputs[j] * weightsHiddenOutput[j, i];
                }
                sum += biasOutput[i];
                output[i] = Sigmoid(sum);
            }

            return output;
        }

        public void BackPropagation(double[,] inputs, double[] targets)
        {
            int inputNodes = inputs.GetLength(0);
            int cols = inputs.GetLength(1);
            int outputNodes = biasOutput.Length;

            // FORWARD PROPAGATION
            double[] hiddenInputs = new double[biasesHidden.Length];
            double[] hiddenOutputs = new double[biasesHidden.Length];
            double[] output = new double[outputNodes];

            // FORWARD PROPAGATION - Hidden layer calculations
            for (int i = 0; i < biasesHidden.Length; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < inputNodes; j++)
                {
                    for (int k = 0; k < cols; k++)
                    {
                        sum += inputs[j, k] * weightsInputHidden[j, i];
                    }
                }
                sum += biasesHidden[i];
                hiddenInputs[i] = sum;
                hiddenOutputs[i] = Sigmoid(sum);
            }

            // FORWARD PROPAGATION - Output layer calculations
            for (int i = 0; i < outputNodes; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < biasesHidden.Length; j++)
                {
                    sum += hiddenOutputs[j] * weightsHiddenOutput[j, i];
                }
                sum += biasOutput[i];
                output[i] = Sigmoid(sum);
            }

            // Error calculation
            double[] errors = new double[outputNodes];
            for (int i = 0; i < outputNodes; i++)
            {
                errors[i] = (targets[i] - output[i]);
            }

            // Updating output layer weights and biases
            for (int i = 0; i < outputNodes; i++)
            {
                double deltaOutput = errors[i] * SigmoidDerivative(output[i]);
                for (int j = 0; j < biasesHidden.Length; j++)
                {
                    weightsHiddenOutput[j, i] += hiddenOutputs[j] * deltaOutput * learningRate;
                }
                biasOutput[i] += deltaOutput * learningRate;
            }

            // Updating hidden layer weights and biases
            for (int i = 0; i < biasesHidden.Length; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < outputNodes; j++)
                {
                    sum += errors[j] * weightsHiddenOutput[i, j];
                }
                double deltaHidden = sum * SigmoidDerivative(hiddenInputs[i]);
                for (int j = 0; j < inputNodes; j++)
                {
                    for (int k = 0; k < cols; k++)
                    {
                        weightsInputHidden[j, i] += inputs[j, k] * deltaHidden * learningRate;
                    }
                }
                biasesHidden[i] += deltaHidden * learningRate;
            }
        }

        public static void SerializeWeights()
        {
            string jsonWeightsInputHidden = JsonConvert.SerializeObject(weightsInputHidden);
            string jsonWeightsHiddenOutput = JsonConvert.SerializeObject(weightsHiddenOutput);
            string jsonBiasesHidden = JsonConvert.SerializeObject(biasesHidden);
            string jsonBiasOutput = JsonConvert.SerializeObject(biasOutput);

            File.WriteAllText("weightsInputHidden.txt", jsonWeightsInputHidden);
            File.WriteAllText("weightsHiddenOutput.txt", jsonWeightsHiddenOutput);
            File.WriteAllText("biasesHidden.txt", jsonBiasesHidden);
            File.WriteAllText("biasOutput.txt", jsonBiasOutput);
        }

        public static void DeserializeWeights()
        {
            
            string jsonWeightsInputHidden = File.ReadAllText("weightsInputHidden.txt");
            string jsonWeightsHiddenOutput = File.ReadAllText("weightsHiddenOutput.txt");
            string jsonBiasesHidden = File.ReadAllText("biasesHidden.txt");
            string jsonBiasOutput = File.ReadAllText("biasOutput.txt");

            weightsInputHidden = JsonConvert.DeserializeObject<double[,]>(jsonWeightsInputHidden);
            weightsHiddenOutput = JsonConvert.DeserializeObject<double[,]>(jsonWeightsHiddenOutput);
            biasesHidden = JsonConvert.DeserializeObject<double[]>(jsonBiasesHidden);
            biasOutput = JsonConvert.DeserializeObject<double[]>(jsonBiasOutput);
            
            
        }
    }
}
