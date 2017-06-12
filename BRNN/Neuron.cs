using System;
using System.Collections.Generic;

namespace BRNN
{
    public class Neuron
    {
        protected IEnumerable<Connection> inputNeuronConnections;
        protected double[] neuronValue;
        private Random random;

        public Func<double, double> ActivationFunction { get; set; }

        public Neuron()
        {

        }

        public Neuron(int epochCount, Func<double, double> activationFunc) : this()
        {
            neuronValue = new double[epochCount];
            ActivationFunction = activationFunc;
            random = new Random();
        }

        public void Activate(int timeFrame)
        {
            AggregateValues(timeFrame);
            neuronValue[timeFrame] = ActivationFunction(neuronValue[timeFrame]);
        }

        public virtual double GetOutput(int timeFrame)
        {
            return neuronValue[timeFrame];
        }

        public void SetInputs(IEnumerable<Neuron> neurons)
        {
            var connections = new List<Connection>();
            foreach (var neuron in neurons)
            {
                var connection = new Connection { Neuron = neuron, Weight = random.NextDouble() };
                connections.Add(connection);
            }
            inputNeuronConnections = connections;
        }

        protected virtual void AggregateValues(int timeFrame)
        {
            foreach (var connection in inputNeuronConnections)
            {
                neuronValue[timeFrame] += connection.Neuron.GetOutput(timeFrame) * connection.Weight;
            }
        }
    }
}