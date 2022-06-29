using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CostSensitivePOMDP
{
    class AlphaVector
    {
        public static int COUNT = 0;
        public int ID { get; private set; }

        public List<AlphaVector> Successors { get; private set; }
        public double[] Values { get; private set; }
        public POMDP POMDP;
        /*
        public int ClassificationAction { get;  set; }
        public int AttributeTest { get;  set; }
        public int AttributeSplit { get;  set; }
        public int SplitValue { get;  set; }
        */
        public Action Action { get; set; }

        public AlphaVector(POMDP p)
        {
            POMDP = p;
            Values = new double[p.States];
            ID = COUNT++;
            Successors = new List<AlphaVector>();
            /*
            ClassificationAction = -1;
            AttributeSplit = -1;
            AttributeTest = -1;
            SplitValue = -1;
            */
        }


        public static AlphaVector operator+(AlphaVector av1, AlphaVector av2)
        {
            AlphaVector avSum = new AlphaVector(av1.POMDP);
            for(int i = 0; i < av1.POMDP.States; i++)
            {
                avSum.Values[i] = av1.Values[i] + av2.Values[i];
            }
            return avSum;
        }

        public static AlphaVector operator-(AlphaVector av, double dCost)
        {
            AlphaVector avSum = new AlphaVector(av.POMDP);
            for (int i = 0; i < av.POMDP.States; i++)
            {
                avSum.Values[i] = av.Values[i] - dCost;
            }
            return avSum;
        }

        public static double operator*(AlphaVector av, Belief b)
        {
            
            if (av.Action is SplitOnAttribtueAction && !b.ObservedAttributes[((SplitOnAttribtueAction)av.Action).Attribute])
                return double.NegativeInfinity;
            //this is for overfitting
            if (b.Examples.Count < 20 && !(av.Action is ClassificationAction))
                return double.NegativeInfinity;
                
            double dSum = 0;
            foreach (int iExample in b.Examples)
                dSum += av.Values[iExample];
            dSum /= b.Examples.Count;
            return dSum;
        }

        public override string ToString()
        {
            return "AV" + ID + " a=" + Action;
        }
    }
}
