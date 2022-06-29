using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CostSensitivePOMDP
{
    class SimpleBelief
    {
        private static int IDs = 0;
        public int ID = IDs++;

        private Dictionary<int, float> m_dValues;
        //private float[] values;
        public double this[int i]
        {
            get {
                if (m_dValues.ContainsKey(i))
                    return m_dValues[i];
                return 0.0;
                //return values[i];
            }
        }

        public SimpleBelief Previous { get; private set; }
        private POMDP POMDP;

        public Action GeneratingAction;
        public bool GeneratingObservation;


        public bool[] ObservedAttributes;
        public int ObservedCount { get; private set; }
        public Dictionary<int, int>[] AttributeValueDistribution;
        public Dictionary<int, int> ClassDistribution { get; private set; }
        public bool SingleValueAttributes { get; private set; }
        public List<int> Examples { get; private set; }


        //initialize first belief
        public SimpleBelief(POMDP p, int cExamples, int cAttributes) : this(p, cExamples * (int)Math.Pow(2, cAttributes))
        {
            bool[] aAttributes = new bool[cAttributes];
            for (int i = 0; i < cExamples; i++)
            {
                int s = POMDP.ToStateIdx(i, aAttributes);
                SetValue(s, 1.0 / cExamples);
            }
        }

        public SimpleBelief(POMDP p, int cStates)
        {
            //values = new float[cStates];
            m_dValues = new Dictionary<int, float>();
            POMDP = p;
            AttributeValueDistribution = new Dictionary<int, int>[POMDP.Attributes];
            for (int i = 0; i < POMDP.Attributes; i++)
                AttributeValueDistribution[i] = new Dictionary<int, int>();
            ClassDistribution = new Dictionary<int, int>();
            Examples = new List<int>();
            SingleValueAttributes = true;
        }

        private void SetValue(int iState, double dValue)
        {
            //values[iState] = (float)dValue;
            m_dValues[iState] = (float)dValue;
            if (dValue != 0.0)
            {
                POMDP.ToState(iState, out int iExample, out bool[] aObserved);
                if (ObservedAttributes == null)
                {
                    ObservedCount = 0;
                    foreach (bool observed in aObserved)
                        if (observed)
                            ObservedCount++;
                    ObservedAttributes = aObserved;
                }
                for (int iAttribute = 0; iAttribute < POMDP.Attributes; iAttribute++)
                {
                    int iValue = POMDP.ExamplesAttributes[iExample][iAttribute];
                    if (!AttributeValueDistribution[iAttribute].ContainsKey(iValue))
                    {
                        AttributeValueDistribution[iAttribute][iValue] = 0;
                        if (AttributeValueDistribution[iAttribute].Keys.Count > 1)
                            SingleValueAttributes = false;
                    }
                    AttributeValueDistribution[iAttribute][iValue]++;
                }
                int iClass = POMDP.ExamplesClasses[iExample];
                if (!ClassDistribution.ContainsKey(iClass))
                    ClassDistribution[iClass] = 0;
                ClassDistribution[iClass]++;
                Examples.Add(iExample);
            }
        }

        public SimpleBelief Next(Action a, bool bObservation)
        {
            SimpleBelief b = new SimpleBelief(POMDP, POMDP.States);

            double dSum = 0.0;
            foreach (int start in m_dValues.Keys)
            {
                if (this[start] != 0)
                {
                    int end = POMDP.Tr(start, a);
                    if (POMDP.O(end, a, bObservation) != 0.0)
                    {
                        b.SetValue(end, this[start]);
                        dSum += b[end];
                    }
                }
            }
            List<int> lNonZero = new List<int>(b.m_dValues.Keys);
            foreach (int i in lNonZero)
            {
                //b.values[i] /= (float)dSum;
                //if(b.m_dValues.ContainsKey(i))
                b.m_dValues[i] = b.m_dValues[i] / (float)dSum;
            }
            b.Previous = this;
            b.GeneratingAction = a;
            b.GeneratingObservation = bObservation;
            return b;
        }

        public override string ToString()
        {
            return "B" + ID + ": " + Examples.Count;
        }
        public override int GetHashCode()
        {
            return Examples.Count;
        }
        public override bool Equals(object obj)
        {
            if (obj is SimpleBelief b)
            {
                if (b.Examples.Count != Examples.Count)
                    return false;
                foreach (int iExample in Examples)
                    if (!b.Examples.Contains(iExample))
                        return false;
                return true;
            }
            return false;
        }

        public double Entropy()
        {
            if (ClassDistribution.Keys.Count == 0)
                return -100.0;
            double dSum = 0.0;
            foreach(int iClass in ClassDistribution.Keys)
            {
                double p = (1.0 * ClassDistribution[iClass]) / Examples.Count;
                dSum += p * Math.Log(p);
            }
            return dSum;
        }
    }
}
