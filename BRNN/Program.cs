using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BRNN
{
    class Program
    {
        private static void run(string[] args)
        {
            int epochCount = 3;
            int recurrentWindowSize = 1;
            var activationFunction = new Func<double, double>(d => d);
            var inputNeuron = new InputNeuron(epochCount, activationFunction);
            var inputNeurons = new List<InputNeuron>();
            inputNeurons.Add(inputNeuron);

            var forwardNeuron = new RecurrentNeuron(recurrentWindowSize, epochCount, RecurrentMode.Forward);
            var forwardNeurons = new List<RecurrentNeuron>();
            forwardNeurons.Add(forwardNeuron);

            var backwardNeuron = new RecurrentNeuron(recurrentWindowSize, epochCount, RecurrentMode.Backward);
            var backwardNeurons = new List<RecurrentNeuron>();
            backwardNeurons.Add(backwardNeuron);

            var outputNeuron = new OutputNeuron(epochCount, activationFunction);
            var outputNeurons = new List<OutputNeuron>();
            outputNeurons.Add(outputNeuron);

            forwardNeuron.SetInputs(inputNeurons);
            backwardNeuron.SetInputs(inputNeurons);
            outputNeuron.SetInputs(forwardNeurons);
            outputNeuron.SetInputs(backwardNeurons);

            var network = new BiRNN(recurrentWindowSize, new double[] { 1, 2, 3 }, epochCount);
            network.InputNeurons = inputNeurons;
            network.OutputNeurons = outputNeurons;
            network.ForwardNeurons = forwardNeurons;
            network.BackwardNeurons = backwardNeurons;

            network.ActivateNetwork();
        }
    }
}
