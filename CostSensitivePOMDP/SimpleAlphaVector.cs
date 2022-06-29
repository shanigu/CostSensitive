using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CostSensitivePOMDP
{
    class SimpleAlphaVector
    {
        private static int IDs = 0;
        public int ID;

        public static int CacheHits = 0;

        public float[] values;
        public double this[int i]
        {
            get { return values[i]; }
            set {
                if (value == 0)
                    Console.Write("*");
                values[i] = (float)value;
            }
        }

        private SimpleAlphaVector[] Next;
        private POMDP POMDP;

        public Action Action { get; set; }

        public Dictionary<string,SimpleAlphaVector> CachedG;


        public SimpleAlphaVector(POMDP p, int cStates)
        {
            values = new float[cStates];
            POMDP = p;
            CachedG = new Dictionary<string, SimpleAlphaVector>();
            ID = IDs++;
            if (ID == 1916)
                Console.Write("*");
        }

        public SimpleAlphaVector G(SplitOnAttribtueAction a, bool bObservation)
        {
            string sName = a.Attribute + "," + a.Split + "," + bObservation;
            if (CachedG.ContainsKey(sName))
            {
                CacheHits++;
                return CachedG[sName];
            }

            SimpleAlphaVector av = new SimpleAlphaVector(POMDP, values.Length);
            for (int s = 0; s < values.Length; s++)
            {
                //if (s == 129)
                //    Console.Write("*");
                    
                float dO = (float)POMDP.O(s, a, bObservation);
                if (dO > 0)
                {
                    int end = POMDP.Tr(s, a);
                    av.values[s] = values[end] * dO;
                }
            }
            CachedG[sName] = av;

            return av;
        }

        public static double operator *(SimpleAlphaVector av, SimpleBelief b)
        {
            //if (av == null || b == null)
            //   return double.NegativeInfinity;
            /*
           double dSum = 0.0;
           for (int i = 0; i < av.values.Length; i++)
           {
               if(b[i] > 0)
                   dSum += av[i] * b[i];
           }
           return dSum;
           */
            return av.product(b);
        }
        public static bool operator >(SimpleAlphaVector av1, SimpleAlphaVector av2)
        {
            if (av1 == null || av2 == null)
                return false;
            for (int i = 0; i < av1.values.Length; i++)
            {
                if (av1[i] < av2[i])
                    return false;
            }
            return true;
        }
        public static bool operator <(SimpleAlphaVector av1, SimpleAlphaVector av2)
        {
            if (av1 == null || av2 == null)
                return false;
            for (int i = 0; i < av1.values.Length; i++)
            {
                if (av1[i] > av2[i])
                    return false;
            }
            return true;
        }
        public double product(SimpleBelief b)
        {
            double dSum = 0;
            foreach(int iExample in b.Examples)
            {
                int iState = POMDP.ToStateIdx(iExample, b.ObservedAttributes);
                dSum += b[iState] * this[iState];
            }
            return dSum;
        }

        public static SimpleAlphaVector operator +(SimpleAlphaVector av1, SimpleAlphaVector av2)
        {
            SimpleAlphaVector avSum = new SimpleAlphaVector(av1.POMDP, av1.values.Length);
            for(int iState = 0; iState < av1.values.Length; iState++)
            {
                avSum[iState] = av1[iState] + av2[iState];
            }
            return avSum;
        }

        public static SimpleAlphaVector operator +(SimpleAlphaVector av1, double d)
        {
            SimpleAlphaVector avSum = new SimpleAlphaVector(av1.POMDP, av1.values.Length);
            for (int iState = 0; iState < av1.values.Length; iState++)
            {
                avSum[iState] = av1[iState] + d;
            }
            return avSum;
        }
        public static SimpleAlphaVector operator -(SimpleAlphaVector av1, double d)
        {
            SimpleAlphaVector avSum = new SimpleAlphaVector(av1.POMDP, av1.values.Length);
            for (int iState = 0; iState < av1.values.Length; iState++)
            {
                avSum[iState] = av1[iState] - d;
            }
            return avSum;
        }

        public override string ToString()
        {
            return "AV" + ID + ": " + Action;
        }

        public override int GetHashCode()
        {
            return Action.ToString().GetHashCode();
        }
        public override bool Equals(object obj)
        {
            if(obj is SimpleAlphaVector av)
            {
                if (Action.ToString() != av.Action.ToString())
                    return false;
                for(int i = 0; i < POMDP.States; i++)
                {
                    if (Math.Abs(values[i] - av.values[i]) > 0.01)
                        return false;

                }
                return true;
            }
            return base.Equals(obj);
        }

        public void AddRewardAndDiscount(Action a)
        {
            for (int iState = 0; iState < POMDP.States; iState++)
            {
                float dR = (float)POMDP.R(iState, a);
                values[iState] = dR + values[iState] * (float)POMDP.Discount;
            }
        }
    }
}
