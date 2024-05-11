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
        public static double error = 1;
        public static double[,] testInput = new double[7, 5];
        public static List<Panel> panels = new List<Panel>();
        static private double[,] weightsInputToHidden; 
        static private double[,] weightsHiddenToOutput; 
        static private double[] hiddenBiases; 
        static private double[] outputBiases; 
        private double learningRate = 0.1; 

        public NeuralNet(int inputNodes, int hiddenNodes, int outputNodes)
        {
            weightsInputToHidden = new double[inputNodes, hiddenNodes];
            weightsHiddenToOutput = new double[hiddenNodes, outputNodes];
            hiddenBiases = new double[hiddenNodes];
            outputBiases = new double[outputNodes];
            Random random = new Random();

            for (int i = 0; i < inputNodes; i++)
            {
                for (int j = 0; j < hiddenNodes; j++)
                {
                    weightsInputToHidden[i, j] = random.NextDouble() - 0.5;
                }
            }

            for (int i = 0; i < hiddenNodes; i++)
            {
                for (int j = 0; j < outputNodes; j++)
                {
                    weightsHiddenToOutput[i, j] = random.NextDouble() - 0.5;
                }
                hiddenBiases[i] = random.NextDouble() - 0.5;
            }

            for (int i = 0; i < outputNodes; i++)
            {
                outputBiases[i] = random.NextDouble() - 0.5;
            }
        }

        private double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        private double SigmoidDerivative(double x)
        {
            double sigmoid = Sigmoid(x);
            return sigmoid * (1 - sigmoid);
        }

        public double[] FeedForward(double[,] inputs)
        {
            int inputNodes = inputs.GetLength(0);
            int cols = inputs.GetLength(1);
            int outputNodes = outputBiases.Length;
            double[] hiddenInputs = new double[hiddenBiases.Length];
            double[] hiddenOutputs = new double[hiddenBiases.Length];
            double[] output = new double[outputNodes];

            // Hidden layer calculations
            for (int i = 0; i < hiddenBiases.Length; i++)
            {
                double sum = 0;
                for (int j = 0; j < inputNodes; j++)
                {
                    for (int k = 0; k < cols; k++)
                    {
                        sum += inputs[j, k] * weightsInputToHidden[j, i];
                    }
                }
                sum += hiddenBiases[i];
                hiddenInputs[i] = sum;
                hiddenOutputs[i] = Sigmoid(sum);
            }

            // Output layer calculations
            for (int i = 0; i < outputNodes; i++)
            {
                double sum = 0;
                for (int j = 0; j < hiddenBiases.Length; j++)
                {
                    sum += hiddenOutputs[j] * weightsHiddenToOutput[j, i];
                }
                sum += outputBiases[i];
                output[i] = Sigmoid(sum);
            }

            return output;
        }

        public void BackPropagation(double[,] inputs, double[] targets)
        {
            int inputNodes = inputs.GetLength(0);
            int cols = inputs.GetLength(1);
            int outputNodes = outputBiases.Length;

            // FORWARD PROPAGATION
            double[] hiddenInputs = new double[hiddenBiases.Length];
            double[] hiddenOutputs = new double[hiddenBiases.Length];
            double[] output = new double[outputNodes];

            // FORWARD PROPAGATION - Hidden layer calculations
            for (int i = 0; i < hiddenBiases.Length; i++)
            {
                double sum = 0;
                for (int j = 0; j < inputNodes; j++)
                {
                    for (int k = 0; k < cols; k++)
                    {
                        sum += inputs[j, k] * weightsInputToHidden[j, i];
                    }
                }
                sum += hiddenBiases[i];
                hiddenInputs[i] = sum;
                hiddenOutputs[i] = Sigmoid(sum);
            }

            // FORWARD PROPAGATION - Output layer calculations
            for (int i = 0; i < outputNodes; i++)
            {
                double sum = 0;
                for (int j = 0; j < hiddenBiases.Length; j++)
                {
                    sum += hiddenOutputs[j] * weightsHiddenToOutput[j, i];
                }
                sum += outputBiases[i];
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
                for (int j = 0; j < hiddenBiases.Length; j++)
                {
                    weightsHiddenToOutput[j, i] += hiddenOutputs[j] * deltaOutput * learningRate;
                }
                outputBiases[i] += deltaOutput * learningRate;
            }

            // Updating hidden layer weights and biases
            for (int i = 0; i < hiddenBiases.Length; i++)
            {
                double sum = 0;
                for (int j = 0; j < outputNodes; j++)
                {
                    sum += errors[j] * weightsHiddenToOutput[i, j];
                }
                double deltaHidden = sum * SigmoidDerivative(hiddenInputs[i]);
                for (int j = 0; j < inputNodes; j++)
                {
                    for (int k = 0; k < cols; k++)
                    {
                        weightsInputToHidden[j, i] += inputs[j, k] * deltaHidden * learningRate;
                    }
                }
                hiddenBiases[i] += deltaHidden * learningRate;
            }
            NeuralNet.error = errors.Max(num => Math.Abs(num));
        }

        public static void SerializeWeights()
        {
            try
            {
                string jsonweightsInputToHidden = JsonConvert.SerializeObject(weightsInputToHidden);
                string jsonweightsHiddenToOutput = JsonConvert.SerializeObject(weightsHiddenToOutput);
                string jsonhiddenBiases = JsonConvert.SerializeObject(hiddenBiases);
                string jsonoutputBiases = JsonConvert.SerializeObject(outputBiases);

                File.WriteAllText("weightsInputToHidden.txt", jsonweightsInputToHidden);
                File.WriteAllText("weightsHiddenToOutput.txt", jsonweightsHiddenToOutput);
                File.WriteAllText("hiddenBiases.txt", jsonhiddenBiases);
                File.WriteAllText("outputBiases.txt", jsonoutputBiases);

                MessageBox.Show("Weights and biases are saved to the file.");

            } catch (Exception e)
            {
                MessageBox.Show("A problem occured when writing files.");
            }
        }

        public static void DeserializeWeights()
        {
            try
            {
                string jsonweightsInputToHidden = File.ReadAllText("weightsInputToHidden.txt");
                string jsonweightsHiddenToOutput = File.ReadAllText("weightsHiddenToOutput.txt");
                string jsonhiddenBiases = File.ReadAllText("hiddenBiases.txt");
                string jsonoutputBiases = File.ReadAllText("outputBiases.txt");

                weightsInputToHidden = JsonConvert.DeserializeObject<double[,]>(jsonweightsInputToHidden);
                weightsHiddenToOutput = JsonConvert.DeserializeObject<double[,]>(jsonweightsHiddenToOutput);
                hiddenBiases = JsonConvert.DeserializeObject<double[]>(jsonhiddenBiases);
                outputBiases = JsonConvert.DeserializeObject<double[]>(jsonoutputBiases);

                MessageBox.Show("Weights and biases are loaded from the file.");
            }
            catch (Exception e)
            {
                MessageBox.Show("A problem occured when reading files.");
            }
        }
    }
}
