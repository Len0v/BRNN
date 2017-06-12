using System;

namespace BRNN
{
    public class InputNeuron : Neuron
    {
        private double[] value;

        public InputNeuron(int epochCount, Func<double, double> activationFunc)
            : base(epochCount, activationFunc)
        { }

        public override double GetOutput(int epoch)
        {
            return value[epoch];
        }

        public void SetOutput(double[] value)
        {
            this.value = value;
        }
    }
}