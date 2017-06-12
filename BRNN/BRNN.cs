using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BRNN
{
    class BiRNN
    {
        public int windowSize { get; set; }
        public double[] inputValues { get; set;}
        public int epochCount { get; set; }
        public List<InputNeuron> InputNeurons { get; set; }
        public List<RecurrentNeuron> ForwardNeurons { get; set; }
        public List<RecurrentNeuron> BackwardNeurons { get; set; }
        public List<OutputNeuron> OutputNeurons { get; set; }

        public BiRNN(int windowSize, double[] inputValues, int epochCount)
        {
            this.windowSize = windowSize;
            this.inputValues = inputValues;
            this.epochCount = epochCount;
        }

        public void ActivateNetwork()
        {
            foreach (var neuron in InputNeurons)
            {
                neuron.SetOutput(inputValues);
            }

            //Step 1 - Input and forward pass
            for (var i = 0; i < epochCount; i++)
            {
                //ActivateNeurons(InputNeurons, i);
                ActivateNeurons(ForwardNeurons, i);
            }

            //Step 2 - Backward and output pass
            for (var i = epochCount - 1; i >= 0; i--)
            {
                ActivateNeurons(BackwardNeurons, i);
                ActivateNeurons(OutputNeurons, i);
            }
        }

        public void ActivateNeurons(IEnumerable<Neuron> neurons, int epochNumber)
        {
            foreach (var neuron in neurons)
            {
                neuron.Activate(epochNumber);
            }
        }
    }
}
