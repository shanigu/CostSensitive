using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatasetGeneration
{
    class Program
    {
        static void Main(string[] args)
        {
            //int cTreeLevels = 5;
            int iIdx = 651;


            StreamWriter swProperties = new StreamWriter(@"G:\Data\CostSensitive\Synthetic\Properties.txt", true);

            // iIdx = ScaleComplexity2(10, iIdx, swProperties);

            //iIdx = ScaleClasses(20, iIdx, swProperties);

            //iIdx = SclaeComplexity(4, iIdx, swProperties);

            iIdx = SclaeComplexity3(10, iIdx, swProperties);

            iIdx = ScaleAttributes(15, iIdx, swProperties);

            iIdx = ScaleSamples(10000, iIdx, swProperties);

            swProperties.Close();

            /*
            List<Example> l = new List<Example>();
            Dictionary<int, int> dClasseCounts = new Dictionary<int, int>();
            for (int i = 0; i < cClasses; i++)
                dClasseCounts[i] = 0;
            int cSuccess = 0;
            for (int i = 0; i < cSamples; i++)
            {
                Example e = c.Sample();
                dClasseCounts[e.Class]++;
                l.Add(e);
                if (c.Classify(e))
                    cSuccess++;
            }
            */

        }

        private static int ScaleSamples(int cMaxSamples, int iIdx, StreamWriter swProperties)
        {
            for (int iSamples = 500; iSamples <= cMaxSamples; iSamples += 500)
            {
                int cClasses = 4;
                int cAttributes = 5;
                int cValues = 10;
                int iRandomSeed = 11;
                int cSamples = iSamples;
                int iCostFactor = 2;
                bool bExponential = true;
                int cCentroids = cClasses * 2;



                for (int iRandom = 0; iRandom < 10; iRandom++)
                {
                    iRandomSeed = DateTime.Now.Millisecond;
                    Clustering c = new Clustering(cClasses, cCentroids, cAttributes, cValues, false, iRandomSeed);

                    /*
                    DecisionTree dt = new DecisionTree(cClasses, cAttributes, cValues, cTreeLevels, iRandomSeed);
                    Console.WriteLine(dt);
                    Console.WriteLine(dt.GetLeaves());
                    */

                    c.WriteDataNames(@"G:\Data\CostSensitive\Synthetic\", iIdx, bExponential, iCostFactor);
                    c.WriteCostMatrix(@"G:\Data\CostSensitive\Synthetic\", iIdx, bExponential, iCostFactor);
                    c.WriteExamples(@"G:\Data\CostSensitive\Synthetic\", iIdx, cSamples, cSamples / 2);

                    swProperties.WriteLine(iIdx + " ScaleSamples " + cClasses + " " + cCentroids + " " + cAttributes + " " + cValues + " " + iRandomSeed + " " + cSamples);
                    iIdx++;
                }
            }

            return iIdx;
        }

        private static int ScaleAttributes(int cMaxAttrbiutes, int iIdx, StreamWriter swProperties)
        {
            for (int iAtributes = 3; iAtributes < cMaxAttrbiutes; iAtributes++)
            {
                int cClasses = 4;
                int cAttributes = iAtributes;
                int cValues = 10;
                int iRandomSeed = 11;
                int cSamples = 1000;
                int iCostFactor = 2;
                bool bExponential = true;
                int cCentroids = cClasses * 2;



                for (int iRandom = 0; iRandom < 10; iRandom++)
                {
                    iRandomSeed = DateTime.Now.Millisecond;
                    Clustering c = new Clustering(cClasses, cCentroids, cAttributes, cValues, false, iRandomSeed);

                    /*
                    DecisionTree dt = new DecisionTree(cClasses, cAttributes, cValues, cTreeLevels, iRandomSeed);
                    Console.WriteLine(dt);
                    Console.WriteLine(dt.GetLeaves());
                    */

                    c.WriteDataNames(@"G:\Data\CostSensitive\Synthetic\", iIdx, bExponential, iCostFactor);
                    c.WriteCostMatrix(@"G:\Data\CostSensitive\Synthetic\", iIdx, bExponential, iCostFactor);
                    c.WriteExamples(@"G:\Data\CostSensitive\Synthetic\", iIdx, cSamples, cSamples / 2);

                    swProperties.WriteLine(iIdx + " ScaleAttributes " + cClasses + " " + cCentroids + " " + cAttributes + " " + cValues + " " + iRandomSeed + " " + cSamples);
                    iIdx++;
                }
            }

            return iIdx;
        }

        private static int SclaeComplexity(int cMaxComplexity, int iIdx, StreamWriter swProperties)
        {
            for (int iComplexity = 1; iComplexity < cMaxComplexity; iComplexity++)
            {
                int cClasses = 4;
                int cAttributes = 5;
                int cValues = 10;
                int iRandomSeed = 11;
                int cSamples = 1000;
                int iCostFactor = 2;
                bool bExponential = true;
                int cCentroids = cClasses * iComplexity;

                for (int iRandom = 0; iRandom < 10; iRandom++)
                {
                    iRandomSeed = DateTime.Now.Millisecond;
                    Clustering c = new Clustering(cClasses, cCentroids, cAttributes, cValues, false, iRandomSeed);

                    /*
                    DecisionTree dt = new DecisionTree(cClasses, cAttributes, cValues, cTreeLevels, iRandomSeed);
                    Console.WriteLine(dt);
                    Console.WriteLine(dt.GetLeaves());
                    */

                    c.WriteDataNames(@"G:\Data\CostSensitive\Synthetic\", iIdx, bExponential, iCostFactor);
                    c.WriteCostMatrix(@"G:\Data\CostSensitive\Synthetic\", iIdx, bExponential, iCostFactor);
                    c.WriteExamples(@"G:\Data\CostSensitive\Synthetic\", iIdx, cSamples, cSamples / 2);

                    swProperties.WriteLine(iIdx + " " + cClasses + " " + cCentroids + " " + cAttributes + " " + cValues + " " + iRandomSeed + " " + cSamples);
                    iIdx++;
                }
            }

            return iIdx;
        }

        private static int SclaeComplexity3(int cClasses, int iIdx, StreamWriter swProperties)
        {
            int cAttributes = 8;
            for (int iDifficultyLevel = 1; iDifficultyLevel <= cAttributes; iDifficultyLevel++)
            {
                int cValues = 10;
                int iRandomSeed = 11;
                int cSamples = 1000;
                int iCostFactor = 2;
                bool bExponential = true;

                for (int iRandom = 0; iRandom < 10; iRandom++)
                {
                    iRandomSeed = DateTime.Now.Millisecond;
                    Clustering c = new Clustering(cClasses, cClasses, cAttributes, cValues, false, iDifficultyLevel, iRandomSeed);

                    /*
                    DecisionTree dt = new DecisionTree(cClasses, cAttributes, cValues, cTreeLevels, iRandomSeed);
                    Console.WriteLine(dt);
                    Console.WriteLine(dt.GetLeaves());
                    */

                    c.WriteDataNames(@"G:\Data\CostSensitive\Synthetic\", iIdx, bExponential, iCostFactor);
                    c.WriteCostMatrix(@"G:\Data\CostSensitive\Synthetic\", iIdx, bExponential, iCostFactor);
                    c.WriteExamples(@"G:\Data\CostSensitive\Synthetic\", iIdx, cSamples, cSamples / 2);

                    swProperties.WriteLine(iIdx + " SclaeComplexity3 " + cClasses + " " + iDifficultyLevel + " " + cAttributes + " " + cValues + " " + iRandomSeed + " " + cSamples);
                    iIdx++;
                }
            }

            return iIdx;
        }

        

        private static int ScaleClasses(int cMaxClasses, int iIdx, StreamWriter swProperties)
        {
            for (int iClasses = 3; iClasses < cMaxClasses; iClasses++)
            {
                int cClasses = iClasses;
                int cAttributes = 5;
                int cValues = 10;
                int iRandomSeed = 11;
                int cSamples = 1000;
                int iCostFactor = 2;
                bool bExponential = true;
                int cCentroids = cClasses * 2;


                for (int iRandom = 0; iRandom < 10; iRandom++)
                {
                    iRandomSeed = DateTime.Now.Millisecond;
                    Clustering c = new Clustering(cClasses, cCentroids, cAttributes, cValues, false, iRandomSeed);

                    /*
                    DecisionTree dt = new DecisionTree(cClasses, cAttributes, cValues, cTreeLevels, iRandomSeed);
                    Console.WriteLine(dt);
                    Console.WriteLine(dt.GetLeaves());
                    */

                    c.WriteDataNames(@"G:\Data\CostSensitive\Synthetic\", iIdx, bExponential, iCostFactor);
                    c.WriteCostMatrix(@"G:\Data\CostSensitive\Synthetic\", iIdx, bExponential, iCostFactor);
                    c.WriteExamples(@"G:\Data\CostSensitive\Synthetic\", iIdx, cSamples, cSamples / 2);

                    swProperties.WriteLine(iIdx + " ScaleClasses " + cClasses + " " + cCentroids + " " + cAttributes + " " + cValues + " " + iRandomSeed + " " + cSamples);
                    iIdx++;
                }
            }

            return iIdx;
        }

        private static int ScaleComplexity2(int cMaxClasses, int iIdx, StreamWriter swProperties)
        {
            for (int iClasses = 3; iClasses < cMaxClasses; iClasses++)
            {
                int cClasses = iClasses;
                int cAttributes = 5;
                int cValues = 10;
                int iRandomSeed = 11;
                int cSamples = 1000;
                int iCostFactor = 2;
                bool bExponential = true;
                int cCentroids = cClasses * 2;


                for (int iRandom = 0; iRandom < 10; iRandom++)
                {
                    iRandomSeed = DateTime.Now.Millisecond;
                    Clustering c = new Clustering(cClasses, cCentroids, cAttributes, cValues, false, iRandomSeed);

                    /*
                    DecisionTree dt = new DecisionTree(cClasses, cAttributes, cValues, cTreeLevels, iRandomSeed);
                    Console.WriteLine(dt);
                    Console.WriteLine(dt.GetLeaves());
                    */

                    c.WriteDataNames(@"G:\Data\CostSensitive\Synthetic\", iIdx, bExponential, iCostFactor);
                    c.WriteCostMatrix(@"G:\Data\CostSensitive\Synthetic\", iIdx, bExponential, iCostFactor);
                    c.WriteExamples(@"G:\Data\CostSensitive\Synthetic\", iIdx, cSamples, cSamples / 2);

                    swProperties.WriteLine(iIdx + " " + cClasses + " " + cCentroids + " " + cAttributes + " " + cValues + " " + iRandomSeed + " " + cSamples);
                    iIdx++;
                }
            }
            for (int iClasses = 3; iClasses < cMaxClasses; iClasses++)
            {
                int cClasses = iClasses;
                int cAttributes = 5;
                int cValues = 10;
                int iRandomSeed = 11;
                int cSamples = 1000;
                int iCostFactor = 2;
                bool bExponential = true;
                int cCentroids = cClasses * 2;


                

                for (int iRandom = 0; iRandom < 10; iRandom++)
                {
                    iRandomSeed = DateTime.Now.Millisecond;
                    Clustering c = new Clustering(cClasses, cCentroids, cAttributes, cValues, true, iRandomSeed);

                    /*
                    DecisionTree dt = new DecisionTree(cClasses, cAttributes, cValues, cTreeLevels, iRandomSeed);
                    Console.WriteLine(dt);
                    Console.WriteLine(dt.GetLeaves());
                    */

                    c.WriteDataNames(@"G:\Data\CostSensitive\Synthetic\", iIdx, bExponential, iCostFactor);
                    c.WriteCostMatrix(@"G:\Data\CostSensitive\Synthetic\", iIdx, bExponential, iCostFactor);
                    c.WriteExamples(@"G:\Data\CostSensitive\Synthetic\", iIdx, cSamples, cSamples / 2);

                    swProperties.WriteLine(iIdx + " " + cClasses + " " + cCentroids + " " + cAttributes + " " + cValues + " " + iRandomSeed + " " + cSamples);
                    iIdx++;
                }
            }

            return iIdx;
        }
    }
}
