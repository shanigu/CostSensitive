using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatasetGeneration
{
    public abstract class GenerationModel
    {
        public int Classes;
        public int Attributes;
        public int Values;
        public int Levels;
        public Dictionary<int, int> AttributeUsefulness;

        public Random Rnd;

        public GenerationModel(int cClasses, int cAttributes, int cValues, int cLevels, int iRandomSeed = 0)
        {
            Classes = cClasses;
            Attributes = cAttributes;
            Values = cValues;
            Levels = cLevels;

            Rnd = new Random(iRandomSeed);
            AttributeUsefulness = new Dictionary<int, int>();
            for (int i = 0; i < Attributes; i++)
            {
                //AttributeUsefulness[i] = Rnd.Next(3);
                AttributeUsefulness[i] = (int)((i * 3.0) / cAttributes);
            }
        }

        public void WriteCostMatrix(string sPath, int iIdx, bool bExponential, int iCostFactor)
        {
            if (!Directory.Exists(sPath))
                Directory.CreateDirectory(sPath);
            StreamWriter sw = new StreamWriter(sPath + "CM." + iIdx);

            for (int i = 0; i < Classes; i++)
            {
                for (int j = 0; j < Classes; j++)
                {
                    int iCost = 0;
                    if (bExponential)
                    {
                        if (i == j)
                            iCost = -1 * (int)Math.Pow(iCostFactor, i + Classes);
                        else
                            iCost = (int)Math.Pow(iCostFactor, i + Math.Abs(i - j));
                    }
                    else
                    {
                        if (i == j)
                            iCost = -1 * (i + Classes) * iCostFactor * iCostFactor;
                        else
                            iCost = (i + 1) * iCostFactor * Math.Abs(i - j);
                    }
                    sw.Write(iCost);
                    if (j < Classes - 1)
                        sw.Write("\t");
                }
                sw.WriteLine();
            }
            sw.Close();
        }

        public void WriteExamples(string sPath, int iIdx, int cTrainSamples, int cTestSamples)
        {
            if (!Directory.Exists(sPath))
                Directory.CreateDirectory(sPath);
            StreamWriter swTrain = new StreamWriter(sPath + "data" + iIdx + ".rr");
            for (int iSample = 0; iSample < cTrainSamples; iSample++)
            {
                Example e = Sample();
                e.Write(swTrain);
            }
            swTrain.Close();
            StreamWriter swTest = new StreamWriter(sPath + "data" + iIdx + ".ss");
            for (int iSample = 0; iSample < cTestSamples; iSample++)
            {
                Example e = Sample();
                e.Write(swTest);
            }
            swTest.Close();
        }

        public abstract Example Sample();
        public abstract bool Classify(Example e);

        public void WriteDataNames(string sPath, int iIdx, bool bExponentialCosts, int iFactor)
        {
            if (!Directory.Exists(sPath))
                Directory.CreateDirectory(sPath);
            StreamWriter sw = new StreamWriter(sPath + "data.names." + iIdx);
            /*
            for (int iClass = 0; iClass < Classes - 1; iClass++)
                sw.Write("C" + iClass + ",");
            sw.WriteLine("C" + (Classes - 1) + ".");
            */
            for (int iClass = 0; iClass < Classes - 1; iClass++)
                sw.Write((iClass + 1) + ",");
            sw.WriteLine(Classes + ".");

            for (int iAttribute = 0; iAttribute < Attributes; iAttribute++)
            {
                sw.Write("a" + iAttribute + ";");
                if (iAttribute % 2 == 0)
                {
                    sw.Write("no:no:");
                    if (bExponentialCosts)
                        sw.Write(Math.Pow(iFactor, iAttribute + 1));
                    else
                        sw.Write(iAttribute + 1);
                }
                else
                {
                    sw.Write("yes:yes:");
                    if (bExponentialCosts)
                        sw.Write(iAttribute + ":" + Math.Pow(iFactor, iAttribute + 1));
                    else
                        sw.Write(iAttribute + ":" + iFactor * (iAttribute + 1));
                }
                sw.WriteLine(":discrete.");
            }

            sw.Close();
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

    }

}

