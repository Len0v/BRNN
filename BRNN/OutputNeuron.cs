using System;

namespace BRNN
{
    public class OutputNeuron : Neuron
    {
        public OutputNeuron(int epochCount, Func<double, double> activationFunc):base(epochCount, activationFunc){ }
    }
}