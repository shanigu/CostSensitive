using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatasetGeneration
{
    class DecisionTree : GenerationModel
    {

        
        public List<Node> Leaves;

        private Node Root;

        public DecisionTree(int cClasses, int cAttributes, int cValues, int cLevels, int iRandomSeed = 0) : base(cClasses, cAttributes, cValues, cLevels, iRandomSeed)
        {
            int[,] aRanges = new int[Attributes, 2];
            for (int i = 0; i < Attributes; i++)
            {
                aRanges[i, 0] = 0;
                aRanges[i, 1] = Values - 1;
            }

            List<int> PrimaryClasses = new List<int>();
            for (int i = 0; i < Classes; i++)
                PrimaryClasses.Add(i);

            int cParticles = 10000;
            int[] aClassParticles = new int[Classes];
            for(int i = 0; i < Classes - 1; i++)
            {
                aClassParticles[i] = Rnd.Next(cParticles / 2);
                cParticles -= aClassParticles[i];
            }
            aClassParticles[Classes - 1] = cParticles;

            Leaves = new List<Node>();
            //Root = GenerateTree(Levels, aRanges, PrimaryClasses);
            Root = GenerateTree(Levels, aRanges, aClassParticles);
            SetPaths(Root, null);
        }

        private void SetPaths(Node n, Node nPrevious)
        {
            n.SetPath(nPrevious);
            if(n.Classes == null)
            {
                SetPaths(n.GT, n);
                SetPaths(n.LTE, n);
            }
        }

        private Node GenerateTreeII(int cLevels, int[,] aRanges, int[] aClassParticles)
        {
            Node n = new Node();


            if (cLevels > 0 && Rnd.NextDouble() < 0.9)
            {
                int iAttribute = Rnd.Next(Attributes);
                while (aRanges[iAttribute, 1] - aRanges[iAttribute, 0] < 1)
                    iAttribute = Rnd.Next(Attributes);
                n.Attribute = iAttribute;
                n.Threashold = Rnd.Next(aRanges[iAttribute, 1] - aRanges[iAttribute, 0]) + aRanges[iAttribute, 0];

                int iSplitClass = Rnd.Next(Classes);
                int[] aGTParticles = new int[Classes];
                int[] aLTEParticles = new int[Classes];
                bool bGTParticles = Rnd.NextDouble() > 0.5;

                for (int i = 0; i < Classes; i++)
                {
                    if (i != iSplitClass)
                    {
                        aGTParticles[i] = aLTEParticles[i] = aClassParticles[i] / 2;
                    }
                }
                if (AttributeUsefulness[iAttribute] == 0)
                {
                    aLTEParticles[iSplitClass] = (int)(0.3 * aClassParticles[iSplitClass]);
                    aGTParticles[iSplitClass] = aClassParticles[iSplitClass] - aLTEParticles[iSplitClass];
                }
                if (AttributeUsefulness[iAttribute] == 1)
                {
                    aLTEParticles[iSplitClass] = (int)(0.15 * aClassParticles[iSplitClass]);
                    aGTParticles[iSplitClass] = aClassParticles[iSplitClass] - aLTEParticles[iSplitClass];
                }
                if (AttributeUsefulness[iAttribute] == 2)
                {
                    aLTEParticles[iSplitClass] = (int)(0.05 * aClassParticles[iSplitClass]);
                    aGTParticles[iSplitClass] = aClassParticles[iSplitClass] - aLTEParticles[iSplitClass];
                }
                if (!bGTParticles)
                {
                    int aux = aGTParticles[iSplitClass];
                    aGTParticles[iSplitClass] = aLTEParticles[iSplitClass];
                    aLTEParticles[iSplitClass] = aux;
                }



                int[,] aRangesGT = (int[,])aRanges.Clone();
                aRangesGT[iAttribute, 0] = n.Threashold + 1;
                n.GT = GenerateTree(cLevels - 1, aRangesGT, aGTParticles);
                int[,] aRangesLTE = (int[,])aRanges.Clone();
                aRangesLTE[iAttribute, 1] = n.Threashold;
                n.LTE = GenerateTree(cLevels - 1, aRangesLTE, aLTEParticles);
            }
            else
            {
                n.Classes = new Dictionary<int, double>();

                int iMainClass = 0;
                double cParticles = 0;
                for (int i = 0; i < Classes; i++)
                {
                    if (aClassParticles[i] > aClassParticles[iMainClass])
                        iMainClass = i;
                    cParticles += aClassParticles[i];
                }
                n.PrimaryClass = iMainClass;

                for (int i = 0; i < Classes; i++)
                {
                    n.Classes[i] = aClassParticles[i] / cParticles;

                }


                Leaves.Add(n);
            }


            return n;
        }


        private Node GenerateTree(int cLevels, int[,] aRanges, int[] aClassParticles)
        {
            Node n = new Node();

            int iMainClass = 0;
            double cParticles = 0;
            for (int i = 0; i < Classes; i++)
            {
                if (aClassParticles[i] > aClassParticles[iMainClass])
                    iMainClass = i;
                cParticles += aClassParticles[i];
            }
            double dProbMainClass = aClassParticles[iMainClass] / cParticles;

            if (cLevels > 0 && dProbMainClass < 0.8)
            {
                int iAttribute = Rnd.Next(Attributes);
                while (aRanges[iAttribute, 1] - aRanges[iAttribute, 0] < 1)
                    iAttribute = Rnd.Next(Attributes);
                n.Attribute = iAttribute;
                n.Threashold = Rnd.Next(aRanges[iAttribute, 1] - aRanges[iAttribute, 0]) + aRanges[iAttribute, 0];

                int iSplitClass = Rnd.Next(Classes);
                int[] aGTParticles = new int[Classes];
                int[] aLTEParticles = new int[Classes];
                bool bGTParticles = Rnd.NextDouble() > 0.5;

                double dPortion = 0.95;
                if (AttributeUsefulness[iAttribute] == 0)
                    dPortion = 0.7;
                if (AttributeUsefulness[iAttribute] == 1)
                    dPortion = 0.85;
                  
                for (int i = 0; i < Classes; i++)
                {
                    if (i == iSplitClass)
                    {
                        aGTParticles[i] = (int)(aClassParticles[i] * dPortion);
                    }
                    else
                    {
                        aGTParticles[i] = (int)(aClassParticles[i] * (1.0 - dPortion));
                    }
                    aLTEParticles[i] = aClassParticles[i] - aGTParticles[i];
                }
                

                int[,] aRangesGT = (int[,])aRanges.Clone();
                aRangesGT[iAttribute, 0] = n.Threashold + 1;
                n.GT = GenerateTree(cLevels - 1, aRangesGT, aGTParticles);
                int[,] aRangesLTE = (int[,])aRanges.Clone();
                aRangesLTE[iAttribute, 1] = n.Threashold;
                n.LTE = GenerateTree(cLevels - 1, aRangesLTE, aLTEParticles);
            }
            else
            {
                n.Classes = new Dictionary<int, double>();

                
                n.PrimaryClass = iMainClass;

                for (int i = 0; i < Classes; i++)
                {
                    n.Classes[i] = aClassParticles[i] / cParticles;

                }


                Leaves.Add(n);
            }


            return n;
        }


        private Node GenerateTree(int cLevels, int[,] aRanges, List<int> PrimaryClasses)
        {
            Node n = new Node();


            if (cLevels > 0 && Rnd.NextDouble() < 0.9)
            {
                int iAttribute = Rnd.Next(Attributes);
                while(aRanges[iAttribute, 1] - aRanges[iAttribute, 0] < 1)
                    iAttribute = Rnd.Next(Attributes);
                n.Attribute = iAttribute;
                n.Threashold = Rnd.Next(aRanges[iAttribute, 1] - aRanges[iAttribute, 0]) + aRanges[iAttribute, 0];





                int[,] aRangesGT = (int[,])aRanges.Clone();
                aRangesGT[iAttribute, 0] = n.Threashold + 1;
                n.GT = GenerateTree(cLevels - 1, aRangesGT, PrimaryClasses);
                int[,] aRangesLTE = (int[,])aRanges.Clone();
                aRangesLTE[iAttribute, 1] = n.Threashold;
                n.LTE = GenerateTree( cLevels - 1, aRangesLTE, PrimaryClasses);
            }
            else
            {
                n.Classes = new Dictionary<int, double>();

                int iMainClass = Rnd.Next(Classes);
                if (PrimaryClasses.Count > 0)
                {
                    int iMainClassIdx = Rnd.Next(PrimaryClasses.Count);
                    iMainClass = PrimaryClasses[iMainClassIdx];
                    PrimaryClasses.Remove(iMainClass);
                }
                n.PrimaryClass = iMainClass;

                int iSecondClass = Rnd.Next(Classes);
                int iThirdClass = Rnd.Next(Classes);

                n.Classes[iMainClass] = 0.8;
                if (!n.Classes.ContainsKey(iSecondClass))
                    n.Classes[iSecondClass] = 0.15;
                else
                    n.Classes[iSecondClass] += 0.15;
                if (!n.Classes.ContainsKey(iThirdClass))
                    n.Classes[iThirdClass] = 0.05;
                else
                    n.Classes[iThirdClass] += 0.05;

                Leaves.Add(n);
            }


            return n;
        }

        int cSources = 0;
        public override bool Classify(Example e)
        {
            Node n = Root;
            while (n.Classes == null)
            {
                if (e.Attributes[n.Attribute] <= n.Threashold)
                    n = n.LTE;
                else
                    n = n.GT;
            }

            //if (n == e.Source)
            //    cSources++;

            return (n.PrimaryClass == e.Class);
                
        }

        public override Example Sample()
        {
            Example e = new Example();
            e.Attributes = new Dictionary<int, int>();
            Node n = Root;
            while(n.Classes == null)
            {
                if(!e.Attributes.ContainsKey(n.Attribute))
                {
                    e.Attributes[n.Attribute] = Rnd.Next(Values);
                }
                bool bGT = false;

                if (e.Attributes[n.Attribute] > n.Threashold)
                    bGT = true;
                             
                if (AttributeUsefulness[n.Attribute] == 0)
                {
                    if (Rnd.NextDouble() < 0.3)
                        bGT = !bGT;
                }
                if (AttributeUsefulness[n.Attribute] == 1)
                {
                    if (Rnd.NextDouble() < 0.15)
                        bGT = !bGT;
                }
                if (AttributeUsefulness[n.Attribute] == 2)
                {
                    if (Rnd.NextDouble() < 0.05)
                        bGT = !bGT;
                }
                
                if (bGT)
                    n = n.GT;
                else
                    n = n.LTE;
            }

            double dProb = Rnd.NextDouble();
            int iClass = -1;
            foreach (KeyValuePair<int, double> p in n.Classes)
            {
                dProb -= p.Value;
                if (dProb < 0)
                {
                    iClass = p.Key;
                    break;
                }
            }
            e.Class = iClass;

            //e.Source = n;

            for (int i = 0; i < Attributes; i++)
            {
                if (!e.Attributes.ContainsKey(i))
                {
                    e.Attributes[i] = Rnd.Next(Values);
                }
            }

            return e;
        }

        public int Normal(Random rnd, int mean, int std)
        {
            double u1 = 1.0 - rnd.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - rnd.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal =
                         mean + std * randStdNormal; //random normal(mean,stdDev^2)
            int iValue = (int)Math.Round(randNormal);
            if (iValue >= Values)
                iValue = Values - 1;
            if (iValue < 0)
                iValue = 0;
            return iValue;
        }


        public override string ToString()
        {
            return Root.ToString(0);
        }

        public string GetLeaves()
        {
            string s = "";
            foreach (Node n in Leaves)
                s += n.ToString(0) + "\n";
            return s;
        }

        public class Node
        {
            public int Attribute;
            public int Threashold;
            public Node GT, LTE;
            public int PrimaryClass;
            public Dictionary<int, double> Classes;
            public int ID;
            public string Path;

            public Dictionary<int, int> Means;

            public static int Nodes = 0;

            public Node()
            {
                ID = Nodes++;
                Path = "";
            }


            public Node(Node nPrevious, int iAttribute, int Threashold )
            {
                ID = Nodes++;
                Path = "";
                if (nPrevious == null)
                    Means = new Dictionary<int, int>();
                else
                    Means = new Dictionary<int, int>(nPrevious.Means);
                if (!Means.ContainsKey(iAttribute))
                    Means[iAttribute] = 0;
            }

            public void SetPath(Node nPrevious)
            {
                Path = "";
                if (nPrevious != null)
                {
                    Path = nPrevious.Path + nPrevious.Attribute;
                    if (this == nPrevious.GT)
                        Path += ">";
                    else
                        Path += "<";
                    Path += nPrevious.Threashold + ", ";
                }
            }

            public string ToString(int cTabs)
            {
                string s = "";
                for (int i = 0; i < cTabs; i++)
                    s += "\t";
                if(Classes == null)
                {
                    return s + "a=" + Attribute + ",t=" + Threashold + "\n" + GT.ToString(cTabs + 1) + "\n" + LTE.ToString(cTabs + 1);
                }
                else
                {
                    string sClasses = Path + ": ";
                    foreach (KeyValuePair<int, double> p in Classes)
                        sClasses += "c" + p.Key + "=" + p.Value + ", ";
                    return s + sClasses;
                }
            }
        }
    }
}
