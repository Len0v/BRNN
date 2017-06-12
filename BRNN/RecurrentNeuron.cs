using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BRNN
{
    class RecurrentNeuron : Neuron
    {
        public RecurrentMode Mode { get; set; }
        protected double recurrentWeight;
        protected int recurrentWindowSize;
        protected int epochNumber;

        public RecurrentNeuron(int recurrentWindowSize, int epochNumber, RecurrentMode mode) : base()
        {
            this.recurrentWindowSize = recurrentWindowSize;
            this.epochNumber = epochNumber;
            Mode = mode;
        }

        protected override void AggregateValues(int timeFrame)
        {
            base.AggregateValues(timeFrame);
            neuronValue[timeFrame] += GetRecurrentValue(timeFrame);
        }

        protected double GetRecurrentValue(int timeFrame)
        {
            double value = 0.0;

            if (Mode == RecurrentMode.Forward)
            {
                for (var i = recurrentWindowSize; i > 0; i--)
                {
                    int index = timeFrame - i;
                    if (index < 0) continue;
                    value += neuronValue[index] * recurrentWeight;
                }
            }
            else
            {
                for (int i = 1; i <= recurrentWindowSize; i++)
                {
                    int index = timeFrame + i;
                    if (index == epochNumber) return value;
                    value += neuronValue[index] * recurrentWeight;
                }
            }
            return value;
        }
    }

    public enum RecurrentMode
    {
        Forward,
        Backward
    }
}
