using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CostSensitivePOMDP
{
    class Belief
    {
        private POMDP POMDP;
        public List<int> Examples;
        public bool[] ObservedAttributes;
        public int ObservedCount { get; private set; }
        public Dictionary<int, int>[] AttributeValueDistribution;
        public Dictionary<int,int> ClassDistribution { get; private set; }
        public bool SingleValueAttributes { get; private set; }

        public int GeneratingTestAction { get; private set; }
        public int GeneratingSplitAttribute { get; private set; }
        public int GeneratingSplitValue { get; private set; }
        public Belief Previous { get; private set; }

        public Belief(POMDP p)
        {
            POMDP = p;
            Examples = new List<int>();
            ObservedAttributes = new bool[p.Attributes];
            AttributeValueDistribution = new Dictionary<int, int>[p.Attributes];
            for (int i = 0; i < p.Attributes; i++)
                AttributeValueDistribution[i] = new Dictionary<int, int>();
            GeneratingTestAction = -1;
            GeneratingSplitAttribute = -1;
            GeneratingSplitValue = -1;
            ObservedCount = 0;
            ClassDistribution = new Dictionary<int, int>();
            SingleValueAttributes = true;
            CopyOf = null;
        }

        private Belief CopyOf;

        //shallow copy because example set is not expected to change
        public Belief(Belief b)
        {
            POMDP = b.POMDP;
            Examples = b.Examples;
            AttributeValueDistribution = b.AttributeValueDistribution;
            ObservedAttributes = new bool[b.ObservedAttributes.Length];
            for (int i = 0; i < ObservedAttributes.Length; i++)
                ObservedAttributes[i] = b.ObservedAttributes[i];
            ClassDistribution = b.ClassDistribution;
            SingleValueAttributes = b.SingleValueAttributes;
            ObservedCount = b.ObservedCount;
            CopyOf = b;
        }

        public void AddExample(int iExample)
        {
            if (CopyOf != null)
                Console.Write("BUGBUG");
            Examples.Add(iExample);
            for (int i = 0; i < POMDP.Attributes; i++)
            {
                int iValue = POMDP.ExamplesAttributes[iExample][i];
                if (!AttributeValueDistribution[i].ContainsKey(iValue))
                {
                    AttributeValueDistribution[i][iValue] = 0;
                    if (AttributeValueDistribution[i].Count > 1)
                        SingleValueAttributes = false;
                }
                AttributeValueDistribution[i][iValue]++;
                
            }
            int iClass = POMDP.ExamplesClasses[iExample];
            if (!ClassDistribution.ContainsKey(iClass))
                ClassDistribution[iClass] = 0;
            ClassDistribution[iClass]++;
        }

        public Belief ObserveAttribute(int iAttribute)
        {
            Belief b = new Belief(this);
            b.GeneratingTestAction = iAttribute;
            if (!b.ObservedAttributes[iAttribute])
                b.ObservedCount++;
            b.ObservedAttributes[iAttribute] = true;
            
            return b;
        }
        public void Split(int iAttribute, int iValue, out Belief bTrue, out Belief bFalse)
        {
            bTrue = new Belief(POMDP);
            bTrue.Previous = this;
            bTrue.GeneratingSplitAttribute = iAttribute;
            bTrue.GeneratingSplitValue = iValue;
            bFalse = new Belief(POMDP);
            bFalse.Previous = this;
            bFalse.GeneratingSplitAttribute = iAttribute;
            bFalse.GeneratingSplitValue = iValue;

            for (int i = 0; i < ObservedAttributes.Length; i++)
            {
                bTrue.ObservedAttributes[i] = ObservedAttributes[i];
                bFalse.ObservedAttributes[i] = ObservedAttributes[i];
            }
            bTrue.ObservedCount = ObservedCount;
            bFalse.ObservedCount = ObservedCount;

            foreach (int iExample in Examples)
            {
                if (POMDP.ExamplesAttributes[iExample][iAttribute] >= iValue)
                    bTrue.AddExample(iExample);
                else
                    bFalse.AddExample(iExample);
            }

        }

        public Belief Split(int iAttribute, int iValue, bool bGreaterThan)
        {
            Belief b = new Belief(POMDP);
            b.Previous = this;
            b.GeneratingSplitAttribute = iAttribute;
            b.GeneratingSplitValue = iValue;

            for (int i = 0; i < ObservedAttributes.Length; i++)
                b.ObservedAttributes[i] = ObservedAttributes[i];
            b.ObservedCount = ObservedCount;

            foreach (int iExample in Examples)
            {
                if ((POMDP.ExamplesAttributes[iExample][iAttribute] >= iValue) == bGreaterThan)
                    b.AddExample(iExample);
            }

            return b;
        }
    }
}
