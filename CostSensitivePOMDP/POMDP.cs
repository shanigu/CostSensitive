using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Accord;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.MachineLearning.DecisionTrees.Rules;

namespace CostSensitivePOMDP
{
    class POMDP
    {
        public List<int[]> ExamplesAttributes { get; private set; }
        public List<int> ExamplesClasses { get; private set; }
        public int Attributes { get; private set; }
        public int Examples { get; private set; }
        public int Classes { get; private set; }
        public List<string> AttributeNames { get; private set; }
        public int[,] AttributeValuesDistribution { get; private set; }
        public int States { get; private set; }
        public double Discount { get; private set; }

        public Dictionary<string, int> ClassNameToIndex;

        public double[,] ClassificationCosts;
        public double[,] TestCosts;
        public List<bool> GroupCosts;

        private double TestExample(List<SimpleBelief> lBeliefs, IEnumerable<SimpleAlphaVector> lValueFunction, HashSet<SimpleAlphaVector> lUsed, int[] values, int iClass, ref int cProblems, ref int cSplits, ref int cTest, ref double dTestCosts, ref bool bCorrect)
        {
            SimpleBelief b = new SimpleBelief(this, ExamplesAttributes.Count, Attributes);
            lBeliefs.Add(b);
            int iPredictedClass = -1;
            double dCost = 0.0;
            dTestCosts = 0.0;
            List<SimpleAlphaVector> lVectors = new List<SimpleAlphaVector>();
            cSplits = 0;
            cTest = 0;
            double dDiscountFactor = 1.0;
            while (iPredictedClass == -1)
            {
                //if (b.Examples.Count < 150)
                //    Console.Write("*");
                SimpleAlphaVector av = GetBestVector(b, lValueFunction, 0);

                if (lVectors.Count > 0 && av == lVectors.Last())
                {
                    cProblems++;
                    iPredictedClass = 0;
                    break;
                }
                //we don't get to a classification action... consider breaking when the number of samples is below
                lVectors.Add(av);
                if (av.Action is ClassificationAction)
                {
                    iPredictedClass = ((ClassificationAction)av.Action).Class;

                    Dictionary<int, double> d = new Dictionary<int, double>();
                    foreach (SimpleAlphaVector av1 in lValueFunction)
                    {
                        if (av1.Action is ClassificationAction act1)
                        {
                            double d1 = av1 * b;
                            d[act1.Class] = d1;
                        }
                    }
                    Dictionary<int, double> d2 = new Dictionary<int, double>();
                    for (int i = 0; i < Classes; i++)
                        d2[i] = 0.0;
                    double count = 0.0;
                    double cost = 0.0;
                    foreach (KeyValuePair<int, int> p in b.ClassDistribution)
                    {
                        count += p.Value;
                        cost += ClassificationCosts[iPredictedClass, p.Key] * p.Value;
                        for (int i = 0; i < Classes; i++)
                            d2[i] += ClassificationCosts[i, p.Key] * p.Value;


                    }
                    for (int i = 0; i < Classes; i++)
                        d2[i] /= count;

                    cost /= count;
                }
                else
                {
                    if (av.Action is SplitOnAttribtueAction act)
                    {
                        cSplits++;
                        double dTestCost = TestCost(b, act.Attribute);
                        if (dTestCost != 0)
                            cTest++;
                        //dTestCosts += dDiscountFactor * dTestCost;
                        dTestCosts += dTestCost;

                        /*
                        if (b.GeneratingAction is SplitOnAttribtueAction aGenerating)
                        {
                            if (act.Attribute == aGenerating.Attribute)
                                Console.Write("*");
                        }
                        */
                        b = b.Next(act, (values[act.Attribute] >= act.Split));
                        lBeliefs.Add(b);
                    }
                }
                dDiscountFactor *= Discount;
            }
            lUsed.UnionWith(lVectors);
            bCorrect = (iPredictedClass == iClass);
            dCost -= ClassificationCosts[iPredictedClass, iClass];
            return dCost;
        }

        private double TestExample(List<SimpleBelief> lBeliefs, IEnumerable<SimpleAlphaVector> lValueFunction, HashSet<SimpleAlphaVector> lUsed, string sLine, ref int cProblems, ref int cSplits, ref int cTest, ref double dTestCosts, ref bool bCorrect)
        {
            string[] a = sLine.Split(' ');
            int[] values = new int[a.Length - 1];
            for (int i = 0; i < a.Length - 1; i++)
            {
                values[i] = int.Parse(a[i]);
            }
            string sClass = a[a.Length - 1];
            int iClass = ClassNameToIndex[sClass];

            return TestExample(lBeliefs, lValueFunction, lUsed, values, iClass, ref cProblems, ref cSplits, ref cTest, ref dTestCosts, ref bCorrect);
        }


        public void TestPolicy(string sPath, string sTestFileName, IEnumerable<SimpleAlphaVector> lValueFunction)
        {
            double dSumClassificationCosts = 0.0, dSumTestCosts = 0.0;
            int cExamples = 0;
            int cProblems = 0;
            double cCorrect = 0;
            double dTotalSplits = 0, dTotalTest = 0;
            double dADR = 0.0;
            HashSet<SimpleAlphaVector> lUsed = new HashSet<SimpleAlphaVector>();
            Dictionary<string, int[]> dWrong = new Dictionary<string, int[]>();
            using (StreamReader sr = new StreamReader(sPath + sTestFileName))
            {
                while (!sr.EndOfStream)
                {
                    List<SimpleBelief> lBeliefs = new List<SimpleBelief>();

                    string sLine = sr.ReadLine();
                    
                    int cSplits = 0, cTest = 0;
                    double dTestCosts = 0.0;
                    bool bCorrect = false;

                    double dCost = TestExample(lBeliefs, lValueFunction, lUsed, sLine, ref cProblems, ref cSplits, ref cTest, ref dTestCosts, ref bCorrect);

                    if (MDPLines.Count() > cExamples)
                    {
                        string[] aLine = MDPLines[cExamples].Split('|');
                        string sSplitPath = aLine[3].Trim();
                        if (!dWrong.ContainsKey(sSplitPath))
                            dWrong[sSplitPath] = new int[4];
                        if (MDPLines[cExamples].Contains("True") && bCorrect)
                            dWrong[sSplitPath][0]++;
                        if (MDPLines[cExamples].Contains("True") && !bCorrect)
                            dWrong[sSplitPath][1]++;
                        if (!MDPLines[cExamples].Contains("True") && bCorrect)
                            dWrong[sSplitPath][2]++;
                        if (!MDPLines[cExamples].Contains("True") && !bCorrect)
                            dWrong[sSplitPath][3]++;

                        if (MDPLines[cExamples].Contains("True") != bCorrect)
                            Console.Write("*");
                    }

                    if (bCorrect)
                        cCorrect++;

                    /*
                    SimpleBelief b = new SimpleBelief(this, ExamplesAttributes.Count, Attributes);
                    int iPredictedClass = -1;
                    double dCost = 0.0, dTestCosts = 0.0;
                    List<SimpleAlphaVector> lVectors = new List<SimpleAlphaVector>();
                    int cSplits = 0, cTest = 0;
                    while (iPredictedClass == -1)
                    {
                        //if (b.Examples.Count < 150)
                        //    Console.Write("*");
                        SimpleAlphaVector av = GetBestVector(b, lValueFunction, 20);
                        if (lVectors.Count > 0 && av == lVectors.Last())
                        {
                            cProblems++;
                            iPredictedClass = 0;
                            break;
                        }
                        //we don't get to a classification action... consider breaking when the number of samples is below
                        lVectors.Add(av);
                        if (av.Action is ClassificationAction)
                            iPredictedClass = ((ClassificationAction)av.Action).Class;
                        else
                        {
                            if (av.Action is SplitOnAttribtueAction act)
                            {
                                cSplits++;
                                double dTestCost = TestCost(b, act.Attribute);
                                if (dTestCost != 0)
                                    cTest++;
                                dTestCosts += dTestCost;
                                b = b.Next(act, (values[act.Attribute] >= act.Split));
                            }
                        }
                    }
                    */
                    dTotalSplits += cSplits;
                    dTotalTest += cTest;
                    dSumTestCosts += dTestCosts;
                    dSumClassificationCosts += dCost;
                    cExamples++;
                    dADR += (dCost - dTestCosts);
                    if (cExamples % 10 == 0)
                        Console.Write("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
                            + cExamples + " ( " + cProblems + "), " + Math.Round(dSumClassificationCosts / cExamples, 2) + ", " +
                            Math.Round((dSumClassificationCosts - dSumTestCosts) / cExamples, 2) + ", " + Math.Round(dADR / cExamples, 2) +
                            "," + dTotalSplits / cExamples + "," + dTotalTest / cExamples + ", " + cCorrect / cExamples + "            ");
                }
            }
            Console.Write("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
                        + cExamples + " ( " + cProblems + "), " + Math.Round(dSumClassificationCosts / cExamples, 2) + ", " +
                        Math.Round((dSumClassificationCosts - dSumTestCosts) / cExamples, 2) + ", " + Math.Round(dADR / cExamples, 2) +
                        "," + dTotalSplits / cExamples + "," + dTotalTest / cExamples + ", " + cCorrect / cExamples + "            ");

            Console.WriteLine();
            StreamWriter sw = new StreamWriter(sPath + "Results.txt", true);
            sw.WriteLine();
            sw.WriteLine(cExamples + " ( " + cProblems + "), " + Math.Round(dSumClassificationCosts / cExamples, 2) + ", " +
               Math.Round((dSumClassificationCosts - dSumTestCosts) / cExamples, 2) + "," + dTotalSplits / cExamples + "," + dTotalTest / cExamples + ", " + cCorrect / cExamples + ", " + DateTime.Now);
            sw.Close();
        }

        public List<SimpleAlphaVector> DecisionTreePointBasedValueIterationIII(int cIterations, int cMinExamples)
        {
            HashSet<SimpleAlphaVector> lValueFunction = new HashSet<SimpleAlphaVector>();
            List<SimpleAlphaVector> lClassificationActions = new List<SimpleAlphaVector>();
            InitClassificationVectors(lClassificationActions);
            HashSet<SimpleBelief> lBeliefs = new HashSet<SimpleBelief>();
            //HashSet<SimpleAlphaVector> lV = new HashSet<SimpleAlphaVector>();


            double[][] inputs = new double[Examples][];
            int[] outputs = new int[Examples];

            for (int iExample = 0; iExample < Examples; iExample++)
            {
                inputs[iExample] = new double[Attributes];
                for (int i = 0; i < Attributes; i++)
                    inputs[iExample][i] = ExamplesAttributes[iExample][i];
                outputs[iExample] = ExamplesClasses[iExample];
            }
            //ID3Learning teacher = new ID3Learning();
            C45Learning teacher = new C45Learning();
            DecisionTree t = teacher.Learn(inputs, outputs);
            DecisionSet rules = t.ToRules();
            int cCorrect = 0;
            for (int iExample = 0; iExample < Examples; iExample++)
            {

                SimpleBelief b = new SimpleBelief(this, ExamplesAttributes.Count, Attributes);
                List<SimpleBelief> lTrial = new List<SimpleBelief>();
                lTrial.Add(b);

                if (rnd.NextDouble() < 0.5)
                {
                    while (b.Examples.Count > cMinExamples && b.ClassDistribution.Keys.Count() > 1 && !b.SingleValueAttributes)
                    {
                        int iAttribute = rnd.Next(Attributes);
                        List<int> lValues = new List<int>(b.AttributeValueDistribution[iAttribute].Keys);
                        if (lValues.Count < 2)
                            continue;
                        lValues.Sort();
                        int iValue = rnd.Next(1, lValues.Count);
                        bool bGreaterThan = rnd.NextDouble() < 0.5;
                        b = b.Next(new SplitOnAttribtueAction(iAttribute, lValues[iValue]), bGreaterThan);
                        lTrial.Add(b);
                    }
                }
                else
                {
                    int iClass = ExamplesClasses[iExample];

                    DecisionRule rChosen = null;
                    foreach (DecisionRule r in rules)
                    {
                        if (r.Match(inputs[iExample]))
                            rChosen = r;
                    }

                    int[] splits = new int[rChosen.Antecedents.Count];
                    for (int i = 0; i < splits.Length; i++)
                    {
                        splits[i] = rChosen.Antecedents[i].Index;
                    }
                    Permute(splits);

                    for (int iSplit = 0; iSplit < splits.Length; iSplit++)
                    {
                        double dValue = -1.0;
                        ComparisonKind ck;
                        foreach (Antecedent a in rChosen.Antecedents)
                        {
                            if (a.Index == splits[iSplit])
                            {
                                dValue = a.Value;
                                ck = a.Comparison;
                            }
                        }
                        SplitOnAttribtueAction action = new SplitOnAttribtueAction(splits[iSplit], dValue);
                        b = b.Next(action, ExamplesAttributes[iExample][splits[iSplit]] >= dValue);
                        lTrial.Add(b);
                    }

                }
                for (int iBelief = lTrial.Count - 1; iBelief >= 0; iBelief--)
                {
                    SimpleAlphaVector av = Backup(lTrial[iBelief], lValueFunction, lClassificationActions);
                    if (av != null)
                    {
                        lValueFunction.Add(av);
                    }
                }
                lBeliefs.UnionWith(lTrial);
                Console.Write("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b" +
                    iExample + ") |V|=" + lValueFunction.Count + " |B| = " + lTrial.Count + ", |B| all = " + lBeliefs.Count + "             ");

            }

            Console.WriteLine();
            for (int i = 0; i < 1; i++)
            {
                int cProblems = 0;
                cCorrect = 0;
                List<SimpleAlphaVector> lAll = new List<SimpleAlphaVector>(lValueFunction);
                lAll.AddRange(lClassificationActions);
                for (int iExample = 0; iExample < Examples; iExample++)
                {
                    /*
                    int[] values = ExamplesAttributes[iExample];
                    int iClass = ExamplesClasses[iExample];
                    List<SimpleBelief> lExampleBeliefs = new List<SimpleBelief>();
                    TestExample(lExampleBeliefs, lAll, new HashSet<SimpleAlphaVector>(), values, iClass, ref cProblems, out int cSplits, out int cTest, out double dCost, out bool bCorrect);
                    if (!bCorrect)
                    {
                        for (int iBelief = lExampleBeliefs.Count - 1; iBelief >= 0; iBelief--)
                        {
                            lBeliefs.Add(lExampleBeliefs[iBelief]);
                            SimpleAlphaVector av = Backup(lExampleBeliefs[iBelief], lValueFunction, lClassificationActions);
                            if (av != null)
                            {
                                lValueFunction.Add(av);
                            }
                        }
                    }
                    else
                        cCorrect++;
                      
                    if (bCorrect)
                        cCorrect++;
                      */
                    Console.Write("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b" +
                        i + "." + iExample + ") |V|=" + lValueFunction.Count + ", " + cCorrect + ", |B| all = " + lBeliefs.Count + "             ");
                }
                Console.WriteLine();
            }

            Console.WriteLine();
            lValueFunction.UnionWith(lClassificationActions);
            return new List<SimpleAlphaVector>(lValueFunction);
        }

        private void Permute(int[] splits)
        {
            if (splits.Length == 1)
                return;
            for(int i = 0; i < splits.Length * 2; i++)
            {
                int j = rnd.Next(splits.Length);
                int k = rnd.Next(splits.Length);
                int aux = splits[j];
                splits[j] = splits[k];
                splits[k] = aux;
            }
        }

        public List<SimpleAlphaVector> DecisionTreePointBasedValueIteration(int cIterations, int cMinExamples)
        {
            HashSet<SimpleAlphaVector> lValueFunction = new HashSet<SimpleAlphaVector>();
            List<SimpleAlphaVector> lClassificationActions = new List<SimpleAlphaVector>();
            InitClassificationVectors(lClassificationActions);
            HashSet<SimpleBelief> lBeliefs = new HashSet<SimpleBelief>();
            //HashSet<SimpleAlphaVector> lV = new HashSet<SimpleAlphaVector>();

            double[][] inputs = new double[Examples][];
            int[] outputs = new int[Examples];

            for (int iExample = 0; iExample < Examples; iExample++)
            {
                inputs[iExample] = new double[Attributes];
                for (int i = 0; i < Attributes; i++)
                    inputs[iExample][i] = ExamplesAttributes[iExample][i];
                outputs[iExample] = ExamplesClasses[iExample];
            }
            //ID3Learning teacher = new ID3Learning();
            C45Learning teacher = new C45Learning();
            DecisionTree t = teacher.Learn(inputs, outputs);
            DecisionSet rules = t.ToRules();

            for (int i = 0; i < cIterations; i++)
            {

                int iExample = rnd.Next(Examples);
                int[] aAttributes = ExamplesAttributes[iExample];
                SimpleBelief b = new SimpleBelief(this, ExamplesAttributes.Count, Attributes);
                List<SimpleBelief> lTrial = new List<SimpleBelief>();
                lTrial.Add(b);

                while (b.Examples.Count > cMinExamples && b.ClassDistribution.Keys.Count() > 1 && !b.SingleValueAttributes)
                {
                    int iAttribute = rnd.Next(Attributes);
                    List<int> lValues = new List<int>(b.AttributeValueDistribution[iAttribute].Keys);
                    if (lValues.Count < 2)
                        continue;
                    lValues.Sort();
                    int iValue = rnd.Next(1, lValues.Count);
                    bool bGreaterThan = rnd.NextDouble() < 0.5;
                    b = b.Next(new SplitOnAttribtueAction(iAttribute, lValues[iValue]), bGreaterThan);
                    lTrial.Add(b);
                    lBeliefs.Add(b);
                }

                for (int iBelief = lTrial.Count - 1; iBelief >= 0; iBelief--)
                {
                    SimpleAlphaVector av = Backup(lTrial[iBelief], lValueFunction, lClassificationActions);
                    if (av != null)
                    {
                        lValueFunction.Add(av);
                    }
                }

                if (i % 10 == 0)
                {

                    foreach (DecisionRule r in rules)
                    {
                        b = new SimpleBelief(this, ExamplesAttributes.Count, Attributes);
                        lTrial = new List<SimpleBelief>();
                        lTrial.Add(b);
                        foreach (Antecedent a in r)
                        {
                            SplitOnAttribtueAction action = new SplitOnAttribtueAction(a.Index, a.Value);
                            b = b.Next(action, a.Comparison == ComparisonKind.GreaterThan);
                            lTrial.Add(b);
                        }
                        for (int iBelief = lTrial.Count - 1; iBelief >= 0; iBelief--)
                        {
                            SimpleAlphaVector av = Backup(lTrial[iBelief], lValueFunction, lClassificationActions);
                            if (av != null)
                            {
                                SimpleBelief bTrue = lTrial[iBelief].Next(av.Action, true);
                                SimpleBelief bFalse = lTrial[iBelief].Next(av.Action, false);
                                lValueFunction.Add(av);
                            }
                        }
                    }

                }
                Console.Write("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b" + i + ") |V|=" + lValueFunction.Count + " |B| = " + lTrial.Count + ", |B| all = " + lBeliefs.Count + "             ");
            }
            Console.WriteLine();
            for (int i = 0; i < 0; i++)
            {
                int cProblems = 0, cCorrect = 0;
                List<SimpleAlphaVector> lAll = new List<SimpleAlphaVector>(lValueFunction);
                lAll.AddRange(lClassificationActions);
                for (int iExample = 0; iExample < Examples; iExample++)
                {
                    /*
                    int[] values = ExamplesAttributes[iExample];
                    int iClass = ExamplesClasses[iExample];
                    List<SimpleBelief> lExampleBeliefs = new List<SimpleBelief>();
                    TestExample(lExampleBeliefs, lAll, new HashSet<SimpleAlphaVector>(), values, iClass, ref cProblems, out int cSplits, out int cTest, out double dCost, out bool bCorrect);
                    if (!bCorrect)
                    {
                        for (int iBelief = lExampleBeliefs.Count - 1; iBelief >= 0; iBelief--)
                        {
                            lBeliefs.Add(lExampleBeliefs[iBelief]);
                            SimpleAlphaVector av = Backup(lExampleBeliefs[iBelief], lValueFunction, lClassificationActions);
                            if (av != null)
                            {
                                lValueFunction.Add(av);
                            }
                        }
                    }
                    else
                        cCorrect++;
                    Console.Write("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b" +
                        i + "." + iExample + ") |V|=" + lValueFunction.Count + ", " + cCorrect + ", |B| all = " + lBeliefs.Count + "             ");
                        */
                }
                Console.WriteLine();
            }

            Console.WriteLine();
            lValueFunction.UnionWith(lClassificationActions);
            return new List<SimpleAlphaVector>(lValueFunction);
        }

        public HashSet<SimpleAlphaVector> BFSPointBasedValueIteration(int cIterations, int cMinExamples, HashSet<SimpleAlphaVector> lValueFunction, HashSet<SimpleBelief> lBeliefs, string sPath, string sTestFileName)
        {
            List<SimpleAlphaVector> lClassificationActions = new List<SimpleAlphaVector>();
            InitClassificationVectors(lClassificationActions);
            //HashSet<SimpleAlphaVector> lV = new HashSet<SimpleAlphaVector>();
            int cUsed = 0;
            double dPrecision = 0.0, dADR = 0.0;
            for (int i = 0; i < cIterations; i++)
            {

                //int iExample = rnd.Next(Examples);
                //int[] aAttributes = ExamplesAttributes[iExample];
                SimpleBelief b = new SimpleBelief(this, ExamplesAttributes.Count, Attributes);
                List<SimpleBelief> lTrial = new List<SimpleBelief>();
                Queue<SimpleBelief> qOpen = new Queue<SimpleBelief>();
                qOpen.Enqueue(b);

                while (qOpen.Count > 0)
                {
                    b = qOpen.Dequeue();
                    lTrial.Add(b);
                    if (b.Examples.Count > cMinExamples && b.ClassDistribution.Keys.Count() > 1 && !b.SingleValueAttributes)
                    {
                        SplitOnAttribtueAction aBest = null;

                        if (lValueFunction.Count < 20 || rnd.NextDouble() < 0.8)
                        {

                            int iAttribute = rnd.Next(Attributes);

                            List<int> lValues = new List<int>(b.AttributeValueDistribution[iAttribute].Keys);
                            if (lValues.Count < 2)
                                continue;

                            double dMaxDiff = 0.0;
                            double dEntropy = b.Entropy();
                            int iBestValue = -1;
                            if (rnd.NextDouble() < 0.5)
                            {
                                foreach (int iCheckValue in lValues)
                                {
                                    SplitOnAttribtueAction a = new SplitOnAttribtueAction(iAttribute, iCheckValue);
                                    SimpleBelief b1 = b.Next(a, true);
                                    SimpleBelief b2 = b.Next(a, false);
                                    double dEntropy1 = b1.Entropy();
                                    double dEntropy2 = b2.Entropy();
                                    if (dEntropy1 - dEntropy > dMaxDiff)
                                    {
                                        dMaxDiff = dEntropy1 - dEntropy;
                                        iBestValue = iCheckValue;
                                    }
                                    if (dEntropy2 - dEntropy > dMaxDiff)
                                    {
                                        dMaxDiff = dEntropy2 - dEntropy;
                                        iBestValue = iCheckValue;
                                    }
                                }
                            }
                            else
                                iBestValue = lValues[rnd.Next(lValues.Count)];
                            aBest = new SplitOnAttribtueAction(iAttribute, iBestValue);
                        }
                        else
                        {
                            SimpleAlphaVector av = GetBestVector(b, lValueFunction, 20);
                            if (av == null || av.Action is ClassificationAction)
                                continue;
                            aBest = (SplitOnAttribtueAction)av.Action;
                        }
                         SimpleBelief bTrue = b.Next(aBest, true);
                        SimpleBelief bFalse = b.Next(aBest, false);
                        qOpen.Enqueue(bTrue);
                        qOpen.Enqueue(bFalse);
                    }
                }
                lBeliefs.UnionWith(lTrial);
                for (int iBelief = lTrial.Count - 1; iBelief >= 0; iBelief--)
                {
                    b = lTrial[iBelief];
                    SimpleAlphaVector avPrevious = GetBestVector(b, lValueFunction);
                    SimpleAlphaVector avNew = Backup(b, lValueFunction, lClassificationActions);
                    double dPreviousValue = double.NegativeInfinity, dNewValue = double.NegativeInfinity;
                    if (avPrevious != null)
                        dPreviousValue = avPrevious * b;
                    if (avNew != null)
                        dNewValue = avNew * b;

                    //SimpleAlphaVector av1 = GetBestVector(b, lClassificationActions);
                    if (dNewValue > dPreviousValue)
                    {
                        lValueFunction.Add(avNew);
                    }
                }
                
                SimpleAlphaVector avBestFirst = GetBestVector(lTrial[0], lValueFunction);
                double dVFirst = avBestFirst * lTrial[0];



                //HashSet<SimpleAlphaVector> lFilteredVectors = FilterDominated(lValueFunction);
                //lValueFunction.IntersectWith(lFilteredVectors);

                if (true)
                {
                    List<SimpleAlphaVector> lAll = new List<SimpleAlphaVector>(lValueFunction);
                    lAll.AddRange(lClassificationActions);
                    /*
                    int cProblems = 0;
                    double cCorrect = 0;
                    HashSet<SimpleAlphaVector> lUsed = new HashSet<SimpleAlphaVector>();
                    dADR = 0.0;
                    dPrecision = 0.0;
                    foreach (string sLine in MDPLines)
                    {
                        List<SimpleBelief> lExampleBeliefs = new List<SimpleBelief>();
                        int cSplits = 0, cTest = 0;
                        double dTestCosts = 0.0;
                        bool bCorrect = false;


                        string[] a1 = sLine.Split('|');
                        string[] a2 = a1[1].Split(',');
                        int[] values = new int[a2.Length - 1];
                        for (int j = 0; j < a2.Length - 1; j++)
                        {
                            values[j] = int.Parse(a2[j].Split('=')[1]);
                        }
                        string sClass = a1[4].Split(new char[] { ' ', '/' }, StringSplitOptions.RemoveEmptyEntries)[1];
                        int iClass = ClassNameToIndex[sClass];

                        double dCost = TestExample(lExampleBeliefs, lValueFunction, lUsed, values, iClass, ref cProblems, ref cSplits, ref cTest, ref dTestCosts, ref bCorrect);
                        if (bCorrect)
                            cCorrect++;
                        dADR += (dCost - dTestCosts);
                    }
                    dPrecision = cCorrect / Examples;
                    dADR /= Examples;
                    */
                    TestPolicy(sPath, sTestFileName, lAll);
                }



                Console.WriteLine("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b" +
                    i + ") |V|=" + lValueFunction.Count + " |B| = " + lTrial.Count + ", |B| all = " + lBeliefs.Count + ", p = " + Math.Round(dPrecision,2) + ", adr=" + Math.Round(dADR,2) +
                    ", used= " + cUsed + ", av=" + avBestFirst + "=" + Math.Round(dVFirst, 3) +  "             ");
            }
            Console.WriteLine();
            

            Console.WriteLine();
            lValueFunction.UnionWith(lClassificationActions);
            return lValueFunction;
        }

        class AlphaVectorComparer : IComparer<SimpleAlphaVector>
        {
            public int Compare(SimpleAlphaVector x, SimpleAlphaVector y)
            {
                for(int i = 0; i < x.values.Length; i++)
                {
                    if (x[i] > y[i])
                        return -1;
                    if (x[i] < y[i])
                        return 1;

                }
                return 0;
            }
        }

        private HashSet<SimpleAlphaVector> FilterDominated(HashSet<SimpleAlphaVector> lValueFunction)
        {
            HashSet<SimpleAlphaVector> lFiltered = new HashSet<SimpleAlphaVector>();

            /*
            List<SimpleAlphaVector> lSorted = new List<SimpleAlphaVector>(lValueFunction);
            lSorted.Sort(new AlphaVectorComparer());
            for(int i = 0; i < lSorted.Count; i++)
            {
                bool bDominated = false;
                SimpleAlphaVector av1 = lSorted[i];
                foreach(SimpleAlphaVector av2 in lFiltered)
                {
                    if(av2 > av1)
                    {
                        bDominated = true;
                        break;
                    }
                }
                if (!bDominated)
                    lFiltered.Add(av1);
            }
            */
            foreach(SimpleAlphaVector av1 in lValueFunction)
            {
                bool bDominated = false;
                foreach (SimpleAlphaVector av2 in lValueFunction)
                {
                    if (av1 != av2 && av2 > av1)
                    {
                        bDominated = true;
                        break;
                    }
                }
                if (!bDominated)
                    lFiltered.Add(av1);
            }
            return lFiltered;
        }

        Random rnd = new Random(10);
        public List<SimpleAlphaVector> PointBasedValueIteration(int cIterations, int cMinExamples)
        {
            HashSet<SimpleAlphaVector> lValueFunction = new HashSet<SimpleAlphaVector>();
            List<SimpleAlphaVector> lClassificationActions = new List<SimpleAlphaVector>();
            InitClassificationVectors(lClassificationActions);
            HashSet<SimpleBelief> lBeliefs = new HashSet<SimpleBelief>();
            //HashSet<SimpleAlphaVector> lV = new HashSet<SimpleAlphaVector>();

            double dPrecision = 0.0, dADR = 0.0;
            for (int i = 0; i < cIterations; i++)
            {

                //int iExample = rnd.Next(Examples);
                //int[] aAttributes = ExamplesAttributes[iExample];
                SimpleBelief b = new SimpleBelief(this, ExamplesAttributes.Count, Attributes);
                List<SimpleBelief> lTrial = new List<SimpleBelief>();
                lTrial.Add(b);

                while (b.Examples.Count > cMinExamples && b.ClassDistribution.Keys.Count() > 1 && !b.SingleValueAttributes)
                {
                    int iAttribute = rnd.Next(Attributes);
                    List<int> lValues = new List<int>(b.AttributeValueDistribution[iAttribute].Keys);
                    if (lValues.Count < 2)
                        continue;
                    lValues.Sort();
                    int iValue = rnd.Next(1, lValues.Count);
                    bool bGreaterThan = rnd.NextDouble() < 0.5;
                    SimpleBelief bTrue = b.Next(new SplitOnAttribtueAction(iAttribute, lValues[iValue]), true);
                    SimpleBelief bFalse = b.Next(new SplitOnAttribtueAction(iAttribute, lValues[iValue]), false);
                    lTrial.Add(bFalse);
                    lBeliefs.Add(bFalse);
                    lTrial.Add(bTrue);
                    lBeliefs.Add(bTrue);
                    if (bGreaterThan)
                        b = bTrue;
                    else
                        b = bFalse;
                }

                for (int iBelief = lTrial.Count - 1; iBelief >= 0; iBelief--)
                {
                    SimpleAlphaVector av = Backup(lTrial[iBelief], lValueFunction, lClassificationActions);
                    if (av != null)
                    {
                        lValueFunction.Add(av);
                    }
                }
                
                Console.Write("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b" +
                    i + ") |V|=" + lValueFunction.Count + " |B| = " + lTrial.Count + ", |B| all = " + lBeliefs.Count + ", p = " + dPrecision + ", adr=" + dADR + "             ");
            }
            Console.WriteLine();
            
            lValueFunction.UnionWith(lClassificationActions);
            return new List<SimpleAlphaVector>(lValueFunction);
        }

        public List<string> MDPLines = new List<string>();

        public HashSet<string> ReadMDPPolicy(string sInputFile)
        {
            StreamReader sr = new StreamReader(sInputFile);

            HashSet<string> hsPaths = new HashSet<string>();
            List<SimpleBelief> lBeliefs = new List<SimpleBelief>();

            double dPrecision = 0.0, dADR = 0.0;
            int i = 0, cSkipped = 0;
            string sSpecialPath = "a6 <= 3,  a6 > 2, a0 > 1,  a6 <= 3,  a0 > 2,  a6 > 1,  a6 > 2, a3 <= 2,  a6 <= 3,  a3 > 1,  a0 > 1,  a0 > 2, a3 <= 2,  a5 > 1,  a6 <= 3,  a3 > 1,  a0 > 1,  a0 > 2,";
            double dSumSpecial = 0.0;
            int cSpecial = 0;
            while (!sr.EndOfStream)
            {
                string sLine = sr.ReadLine().Trim();
                if (sLine == "")
                    continue;

                MDPLines.Add(sLine);


                string[] aLine = sLine.Split('|');

                string sPath = aLine[3].Trim();
                if (sPath != "Empty")
                    hsPaths.Add(sPath);

                if (MDPLines.Count == 30)
                    Console.Write("*");

                string[] path = sPath.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                SimpleBelief b = new SimpleBelief(this, ExamplesAttributes.Count, Attributes);
                List<SimpleBelief> lTrial = new List<SimpleBelief>();
                lTrial.Add(b);
                string[] aAttributes = aLine[1].Split(',');

                //int iExample = rnd.Next(Examples);
                int[] values = new int[Attributes];
                foreach (string sAttribute in aAttributes)
                {
                    if (sAttribute.Trim() == "")
                        continue;
                    string[] pair = sAttribute.Split('=');
                    int idx = AttributeNames.IndexOf(pair[0].Trim());
                    int iValue = int.Parse(pair[1].Trim());
                    values[idx] = iValue;
                }
                double dCost = 0.0;
                bool[] aObserved = new bool[Attributes];
                if (sPath != "Empty")
                {
                    foreach (string sAction in path)
                    {
                        string[] rule = sAction.Trim().Split(' ');

                        int iAttribute = AttributeNames.IndexOf(rule[0].Trim());
                        int iValue = int.Parse(rule[2].Trim());
                        double dValue = iValue + 0.5;

                        bool bGreaterThan = rule[1].Contains('>');


                        SimpleBelief bNext = b.Next(new SplitOnAttribtueAction(iAttribute, dValue), bGreaterThan);
                        dCost += R(aObserved, bNext.GeneratingAction);
                        aObserved[iAttribute] = true;
                        lTrial.Add(bNext);
                        lBeliefs.Add(bNext);
                        b = bNext;
                    }
                }
                string sClassAndClassify = aLine[4].Trim().Split(' ')[0];
                int iClassify = ClassNameToIndex[sClassAndClassify.Split('/')[0]];
                int iClass = ClassNameToIndex[sClassAndClassify.Split('/')[1]];

                double dTotal = dCost - ClassificationCosts[iClassify, iClass];
                dADR += dTotal;

                if (sPath == sSpecialPath)
                {
                    dSumSpecial += dTotal;
                    cSpecial++;
                }

                i++;
            }

            dADR /= i;
            return hsPaths;
        }


        public HashSet<SimpleAlphaVector> MDPPointBasedValueIteration(string sInputFile, HashSet<SimpleAlphaVector> lValueFunction)
        {
            List<SimpleAlphaVector> lClassificationActions = new List<SimpleAlphaVector>();
            InitClassificationVectors(lClassificationActions);
            HashSet<SimpleBelief> lBeliefs = new HashSet<SimpleBelief>();
            //HashSet<SimpleAlphaVector> lV = new HashSet<SimpleAlphaVector>();

            double dPrecision = 0.0, dADR = 0.0;
            int i = 0;
            HashSet<string> hsPaths = ReadMDPPolicy(sInputFile);
            /*
             * if (sPath == "Empty")
            {
                cSkipped++;
                continue;
            }

            if (!hsPaths.Add(sPath) || sPath == "Empty")
            {
                cSkipped++;
                continue;
            }
            */

            /*

            string[] aAttributes = aLine[1].Split(',');

            //int iExample = rnd.Next(Examples);
            int[] values = new int[Attributes];
            foreach(string sAttribute in aAttributes)
            {
                if (sAttribute.Trim() == "")
                    continue;
                string[] pair = sAttribute.Split('=');
                int idx = AttributeNames.IndexOf(pair[0].Trim());
                int iValue = int.Parse(pair[1].Trim());
                values[idx] = iValue;
            }
            */
            for (int iIteration = 0; iIteration < 200; iIteration++)
            {
                HashSet<SimpleAlphaVector> lNewV = new HashSet<SimpleAlphaVector>();
                double dDeltaV = 0.0, dNewValue = 0.0;
                int cImprovements = 0;
                i = 0;
                for (int j = 0; j < hsPaths.Count; j++)
                {
                    string sPath = hsPaths.ElementAt(j);
                    string[] path = sPath.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                    SimpleBelief b = new SimpleBelief(this, ExamplesAttributes.Count, Attributes);
                    List<SimpleBelief> lTrial = new List<SimpleBelief>();
                    lTrial.Add(b);
                    List<Action> lActions = new List<Action>();

                    List<string> lFilteredPath = new List<string>();
                    foreach(string sAction in path)
                    {
                    
                        if (!lFilteredPath.Contains(sAction.Trim()))
                            lFilteredPath.Add(sAction.Trim());
                    }

                    foreach (string sAction in lFilteredPath)
                    {
                        string[] rule = sAction.Trim().Split(' ');

                        int iAttribute = AttributeNames.IndexOf(rule[0].Trim());
                        int iValue = int.Parse(rule[2].Trim());
                        double dValue = iValue + 0.5;

                        bool bGreaterThan = rule[1].Contains('>');

                        Action a = new SplitOnAttribtueAction(iAttribute, dValue);
                        lActions.Add(a);
                        SimpleBelief bNext = b.Next(a, bGreaterThan);
                        lTrial.Add(bNext);
                        lBeliefs.Add(bNext);
                        b = bNext;
                    }
                    if (j == 6)
                        Console.Write("*");
                    for (int iBelief = lTrial.Count - 1; iBelief >= 0; iBelief--)
                    {
                        b = lTrial[iBelief];
                        SimpleAlphaVector avPrevious = GetBestVector(b, lValueFunction);
                        SimpleAlphaVector avClassify = GetBestVector(b, lClassificationActions);
                        double dPrevious = double.NegativeInfinity, dClassify = 0.0;
                        if (avPrevious != null)
                            dPrevious = avPrevious * b;
                        dClassify = avClassify * b;
                        SimpleAlphaVector av = Backup(b, lValueFunction, lClassificationActions);

                        if (iBelief == 0 && av.Action.ToString() != "Split 0 >= 4")
                            Console.Write("*");

                        //SplitOnAttribtueAction a = (SplitOnAttribtueAction)lActions[iBelief];
                        //SimpleAlphaVector av = ComputeActionVector(b, a.Attribute, a.Split, lValueFunction, lClassificationActions);
                        if (av != null)
                        {
                            double dNew = av * b;
                            if (dNew <= dPrevious && avPrevious != null)
                                lNewV.Add(avPrevious);
                            if (dNew > dPrevious)
                            {
                                if (dNew - dPrevious > dDeltaV)
                                {
                                    dDeltaV = dNew - dPrevious;
                                    dNewValue = dNew;
                                }
                                cImprovements++;
                                lNewV.Add(av);
                                lValueFunction.Add(av);
                            }
                        }
                    }
                    for (int iBelief = 0; iBelief >= 0; iBelief--)
                    {
                        b = lTrial[iBelief];
                        SimpleAlphaVector avPrevious = GetBestVector(b, lValueFunction);
                        SimpleAlphaVector avClassify = GetBestVector(b, lClassificationActions);
                        double dPrevious = double.NegativeInfinity, dClassify = 0.0;
                        if (avPrevious != null)
                            dPrevious = avPrevious * b;
                        dClassify = avClassify * b;
                        SimpleAlphaVector av = Backup(b, lValueFunction, lClassificationActions);

                        if (iBelief == 0 && av.Action.ToString() != "Split 0 >= 4")
                            Console.Write("*");

                        //SplitOnAttribtueAction a = (SplitOnAttribtueAction)lActions[iBelief];
                        //SimpleAlphaVector av = ComputeActionVector(b, a.Attribute, a.Split, lValueFunction, lClassificationActions);
                        if (av != null)
                        {
                            double dNew = av * b;
                            if (dNew <= dPrevious && avPrevious != null)
                                lNewV.Add(avPrevious);
                            if (dNew > dPrevious)
                            {
                                if (dNew - dPrevious > dDeltaV)
                                {
                                    dDeltaV = dNew - dPrevious;
                                    dNewValue = dNew;
                                }
                                cImprovements++;
                                lNewV.Add(av);
                                lValueFunction.Add(av);
                            }
                        }
                    }
                    Console.Write("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b" +
                       iIteration + ")" + i + "/" + hsPaths.Count + ": |V|=" + lValueFunction.Count + ", |V'|=" + lNewV.Count + 
                       " |B| = " + lTrial.Count + ", |B| all = " + lBeliefs.Count + ", p = " + Math.Round(dPrecision,2) +
                       ", adr=" + Math.Round(dADR, 2) + ", delta=" + Math.Round(dDeltaV, 2) + ", " + Math.Round(dNewValue, 2) + ", " + cImprovements + "             ");
                    i++;
                }
                if (true)
                {
                    int cProblems = 0;
                    double cCorrect = 0;
                    List<SimpleAlphaVector> lAll = new List<SimpleAlphaVector>(lValueFunction);
                    HashSet<SimpleAlphaVector> lUsed = new HashSet<SimpleAlphaVector>();
                    lAll.AddRange(lClassificationActions);
                    foreach(string sLine in MDPLines)
                    {
                        List<SimpleBelief> lExampleBeliefs = new List<SimpleBelief>();
                        int cSplits = 0, cTest = 0;
                        double dTestCosts = 0.0;
                        bool bCorrect = false;


                        string[] a1 = sLine.Split('|');
                        string[] a2 = a1[1].Split(',');
                        int[] values = new int[a2.Length - 1];
                        for (int j= 0; j < a2.Length - 1; j++)
                        {
                            values[j] = int.Parse(a2[j].Split('=')[1]);
                        }
                        string sClass = a1[4].Split(new char[] { ' ', '/' }, StringSplitOptions.RemoveEmptyEntries)[1];
                        int iClass = ClassNameToIndex[sClass];

                        double dCost = TestExample(lExampleBeliefs, lValueFunction, lUsed, values, iClass, ref cProblems, ref cSplits, ref cTest, ref dTestCosts, ref bCorrect);
                        if (bCorrect)
                            cCorrect++;
                        dADR += (dCost - dTestCosts);
                    }
                    dPrecision = cCorrect / Examples;
                    dADR /= Examples;
                }
                if (lValueFunction.Count > 500)
                {
                    lValueFunction.Clear();
                    lValueFunction.UnionWith(lNewV);
                }
                if (dDeltaV == 0.0)
                    break;
                i++;
                Console.WriteLine();
            }
            Console.WriteLine();


            Console.WriteLine();
            lValueFunction.UnionWith(lClassificationActions);
            return lValueFunction;
        }

        private HashSet<SimpleBelief> CollectRandomBeliefs(int cIterations, int cMinExamples)
        {
            HashSet<SimpleBelief> lBeliefs = new HashSet<SimpleBelief>();

            for (int i = 0; i < cIterations; i++)
            {
                int iExample = rnd.Next(Examples);
                int[] aAttributes = ExamplesAttributes[iExample];
                SimpleBelief b = new SimpleBelief(this, ExamplesAttributes.Count, Attributes);
                List<SimpleBelief> lTrial = new List<SimpleBelief>();
                lTrial.Add(b);

                while (b.Examples.Count > cMinExamples && b.ClassDistribution.Keys.Count() > 1 && !b.SingleValueAttributes)
                {
                    int iAttribute = rnd.Next(Attributes);
                    List<int> lValues = new List<int>(b.AttributeValueDistribution[iAttribute].Keys);
                    if (lValues.Count < 2)
                        continue;
                    lValues.Sort();
                    int iValue = rnd.Next(1, lValues.Count);
                    bool bGreaterThan = rnd.NextDouble() < 0.5;
                    b = b.Next(new SplitOnAttribtueAction(iAttribute, lValues[iValue]), bGreaterThan);
                    lTrial.Add(b);
                }
                foreach (SimpleBelief bCollected in lTrial)
                {
                    int cValuesCount = 0;
                    foreach (var l in bCollected.AttributeValueDistribution)
                        if (l.Count > cValuesCount)
                            cValuesCount = l.Count;
                    if(cValuesCount > 1)
                        lBeliefs.Add(bCollected);
                }
            }
            Console.WriteLine("Collected " + lBeliefs.Count + " beliefs");
            return lBeliefs;
        }

        private List<SimpleBelief> CollectAllBeliefs(int cIterations, int cMinExamples)
        {
            HashSet<SimpleBelief> lBeliefs = new HashSet<SimpleBelief>();

            for (int iExample = 0; iExample < Examples; iExample++)
            {
                for (int i = 0; i < cIterations; i++)
                {
                    int[] aAttributes = ExamplesAttributes[iExample];
                    SimpleBelief b = new SimpleBelief(this, ExamplesAttributes.Count, Attributes);
                    List<SimpleBelief> lTrial = new List<SimpleBelief>();
                    lTrial.Add(b);

                    while (b.Examples.Count > cMinExamples && b.ClassDistribution.Keys.Count() > 1 && !b.SingleValueAttributes)
                    {
                        int iAttribute = rnd.Next(Attributes);
                        List<int> lValues = new List<int>(b.AttributeValueDistribution[iAttribute].Keys);
                        if (lValues.Count < 2)
                            continue;
                        lValues.Sort();
                        int iValue = rnd.Next(1, lValues.Count);
                        bool bGreaterThan = rnd.NextDouble() < 0.5;
                        b = b.Next(new SplitOnAttribtueAction(iAttribute, lValues[iValue]), bGreaterThan);
                        lTrial.Add(b);
                    }

                    foreach (SimpleBelief bCollected in lTrial)
                    {
                        if (bCollected.ClassDistribution.Count > 1)
                        {
                            int cValuesCount = 0;
                            foreach (var l in bCollected.AttributeValueDistribution)
                                if (l.Count > cValuesCount)
                                    cValuesCount = l.Count;
                            if (cValuesCount > 1)
                                lBeliefs.Add(bCollected);
                        }
                    }
                }
            }
            Console.WriteLine("Collected " + lBeliefs.Count + " beliefs");
            return new List<SimpleBelief>(lBeliefs);
        }


        public HashSet<SimpleAlphaVector> Perseus(int cIterations, IEnumerable<SimpleBelief> lBeliefs, IEnumerable<SimpleAlphaVector> lValueFunction, List<SimpleAlphaVector> lClassificationActions)
        {

            DateTime dtStart = DateTime.Now;

            for (int i = 0; i < cIterations; i++)
            {
                HashSet<SimpleAlphaVector> lNewValueFunction = new HashSet<SimpleAlphaVector>();
                List<SimpleBelief> lCurrent = new List<SimpleBelief>(lBeliefs);
                List<SimpleAlphaVector> lAll = new List<SimpleAlphaVector>(lValueFunction);
                lAll.AddRange(lClassificationActions);

                double dMaxDelta = 0.0;
                SimpleBelief bMaxDelta = null;

                Dictionary<int, double> dBeliefValues = new Dictionary<int, double>();
                foreach (SimpleBelief b in lCurrent)
                {
                    SimpleAlphaVector avBest = GetBestVector(b, lAll);
                    double dBestValue = avBest * b;
                    dBeliefValues[b.ID] = dBestValue;
                }

                while (lCurrent.Count > 0)
                {
                    SimpleAlphaVector avNew = null;
                    SimpleBelief bSelected = null;
                    int idx = rnd.Next(lCurrent.Count);
                    bSelected = lCurrent[idx];
                    avNew = Backup(bSelected, lValueFunction, lClassificationActions);
                    double dBestValue = dBeliefValues[bSelected.ID];
                    double dNewValue = double.NegativeInfinity;
                    if(avNew != null)
                        dNewValue = avNew * bSelected;

                    if (dNewValue - dBestValue > dMaxDelta)
                    {
                        dMaxDelta = dNewValue - dBestValue;
                        bMaxDelta = bSelected;
                    }


                    List<SimpleBelief> lNext = new List<SimpleBelief>();
                    if (dNewValue < dBestValue || avNew == null)
                    {
                        SimpleAlphaVector avPrevious = GetBestVector(bSelected, lAll);
                        if (avPrevious.Action is SplitOnAttribtueAction)
                            lNewValueFunction.Add(avPrevious);
                        foreach (SimpleBelief bOther in lCurrent)
                        {
                            if (bOther == bSelected)
                                continue;

                            lNext.Add(bOther);
                        }
                    }
                    else
                    {
                        foreach (SimpleBelief bOther in lCurrent)
                        {
                            //if (bOther.Equals(bSelected))
                            //    Console.Write("*");
                            if (bOther == bSelected)
                                continue;
                            //SimpleAlphaVector avBest = GetBestVector(bOther, lAll);
                            //double dBestValue = avBest * bOther;
                            dBestValue = dBeliefValues[bOther.ID];
                            dNewValue = avNew * bOther;
                            if (dNewValue < dBestValue)
                                lNext.Add(bOther);
                            else
                            {
                                if (dNewValue - dBestValue > dMaxDelta)
                                    dMaxDelta = dNewValue - dBestValue;
                            }
                        }
                        lNewValueFunction.Add(avNew);
                    }
                    lCurrent = lNext;
                    DateTime dtCurrent = DateTime.Now;
                    Console.Write("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
                        + i + ") " + "|V|=" + lValueFunction.Count() + ",|V'|= " + lNewValueFunction.Count + "," + dMaxDelta + ",|B|=" + lCurrent.Count + "," + bMaxDelta + ", T=" + (dtCurrent - dtStart).TotalSeconds + "                    ");
                    //if (lNewValueFunction.Count == 20)
                    //    Console.Write("*");
                }
                lValueFunction = new HashSet<SimpleAlphaVector>(lNewValueFunction);
                //lValueFunction.AddRange(lNewValueFunction);
                if (dMaxDelta < 0.2)
                    break;
                Console.Write("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
                    + i + ") " + "|V|=" + lValueFunction.Count() + ",|V'|=" + lNewValueFunction.Count + ", " + dMaxDelta + "                                             ");

            }
            Console.WriteLine();
            return new HashSet<SimpleAlphaVector>(lValueFunction);
        }

        public HashSet<SimpleAlphaVector> Perseus(int cIterations, int cMinExamples)
        {
            HashSet<SimpleAlphaVector> lValueFunction = new HashSet<SimpleAlphaVector>();
            List<SimpleAlphaVector> lClassificationActions = new List<SimpleAlphaVector>();
            InitClassificationVectors(lClassificationActions);

            //List<SimpleBelief> lBeliefs = CollectRandomBeliefs(cIterations, cMinExamples);
            List<SimpleBelief> lUnfilteredBeliefs = CollectAllBeliefs(cIterations, cMinExamples);
            HashSet<SimpleBelief> lFilteredBeliefs = new HashSet<SimpleBelief>();
            foreach(SimpleBelief b in lUnfilteredBeliefs)
            {
                if (HasImprovingSplit(b, lClassificationActions))
                    lFilteredBeliefs.Add(b);
            }

            Console.WriteLine("Filtered " + lUnfilteredBeliefs.Count + " => " + lFilteredBeliefs.Count);

            lValueFunction = Perseus(cIterations, lFilteredBeliefs, lValueFunction, lClassificationActions);

            lValueFunction = Perseus(cIterations / 10, lUnfilteredBeliefs, lValueFunction, lClassificationActions);

            lValueFunction.UnionWith(lClassificationActions);
            return lValueFunction;
        }

        public HashSet<SimpleAlphaVector> Perseus(int cIterations, int cMinExamples, HashSet<SimpleAlphaVector> lValueFunction, HashSet<SimpleBelief> lBeliefs)
        {
            List<SimpleAlphaVector>  lClassificationActions = new List<SimpleAlphaVector>();
            InitClassificationVectors(lClassificationActions);
            foreach (SimpleAlphaVector av in lClassificationActions)
                lValueFunction.Remove(av);

            lBeliefs = CollectRandomBeliefs(cIterations, cMinExamples);
            HashSet<SimpleBelief> lFilteredBeliefs = new HashSet<SimpleBelief>();
            foreach (SimpleBelief b in lBeliefs)
            {
                if (HasImprovingSplit(b, lClassificationActions))
                    lFilteredBeliefs.Add(b);
            }

            Console.WriteLine("Filtered " + lBeliefs.Count + " => " + lFilteredBeliefs.Count);

            lValueFunction = Perseus(cIterations, lFilteredBeliefs, lValueFunction, lClassificationActions);

            lValueFunction = Perseus(cIterations / 10, lBeliefs, lValueFunction, lClassificationActions);

            lValueFunction.UnionWith(lClassificationActions);
            return lValueFunction;
        }

        private Belief GetInitialBelief()
        {
            Belief b = new Belief(this);
            for (int i = 0; i < States; i++)
                b.AddExample(i);
            return b;
        }

        public void LoadExamples(string sTrainFileName)
        {
            Attributes = -1;
            ExamplesAttributes = new List<int[]>();
            ExamplesClasses = new List<int>();
            int cLines = 0;

            if (ClassNameToIndex.Count == 0)
                Console.WriteLine("You must load data.names before you load the training set.");

            using (StreamReader sr = new StreamReader(sTrainFileName))
            {
                Console.WriteLine("Loading from " + sTrainFileName);
                while (!sr.EndOfStream)
                {
                    string sLine = sr.ReadLine().Trim();
                    if (sLine == "")
                        continue;
                    string[] a = sLine.Split(" \t".ToCharArray());
                    if (Attributes == -1)
                    {
                        Attributes = a.Length - 1;
                        AttributeValuesDistribution = new int[Attributes, 15];//assuming all attributes are in the range [1,100];
                    }
                    int[] values = new int[a.Length - 1];
                    for (int i = 0; i < a.Length - 1; i++)
                    {
                        values[i] = int.Parse(a[i]);
                        AttributeValuesDistribution[i, values[i]]++;
                    }
                    ExamplesAttributes.Add(values);
                    string sClass = a[a.Length - 1];
                    ExamplesClasses.Add(ClassNameToIndex[sClass]);
                    cLines++;
                    if (cLines % 10 == 0)
                        Console.Write("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b" + "Loaded " + cLines + " examples                                       ");
                }
            }
            Console.WriteLine("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b" + "Loaded " + cLines + " examples                                       ");
            States = (int)Math.Pow(2, Attributes) * ExamplesAttributes.Count();
            Examples = ExamplesAttributes.Count;
            /*
            for(int i = 0; i < States; i++)
            {
                bool[] a = null;
                int e = 0;
                ToState(i, out e, out a);
                int j = ToStateIdx(e, a);
                if (i != j)
                    Console.Write("*");
            }
            */
        }

        public void ToState(int iStateIdx, out int iExample, out bool[] ObservedAttributes)
        {
            int iOffset = (int)Math.Pow(2, Attributes);
            ObservedAttributes = new bool[Attributes];
            for(int iAttribute = 0; iAttribute < Attributes; iAttribute++)
            {
                ObservedAttributes[iAttribute] = (iStateIdx % 2) == 1;
                iStateIdx /= 2;
            }
            iExample = iStateIdx;
        }
        public int ToStateIdx(int iExample, bool[] ObservedAttributes)
        {
            int iStateIdx = iExample;
            for (int iAttribute = Attributes - 1; iAttribute >= 0; iAttribute--)
            {
                iStateIdx *= 2;
                if (ObservedAttributes[iAttribute])
                    iStateIdx++;
            }
            return iStateIdx;
        }

        public int Tr(int iState, Action a)
        {
            if (a is ClassificationAction)
                return iState;
            SplitOnAttribtueAction test = (SplitOnAttribtueAction)a;
            int iExample = 0;
            bool[] ObservedAttributes = null;
            ToState(iState, out iExample, out ObservedAttributes);
            if (ObservedAttributes[test.Attribute])
                return iState;
            ObservedAttributes[test.Attribute] = true;
            return ToStateIdx(iExample, ObservedAttributes);
        }
        public double O(int iState, Action a, bool bObservedValue)
        {
            if (a is ClassificationAction)
            {
                if (bObservedValue == false)
                    return 1.0;
                return 0.0;
            }
            SplitOnAttribtueAction split = (SplitOnAttribtueAction)a;
            int iExample = 0;
            bool[] aAttributes = null;
            ToState(iState, out iExample, out aAttributes);
            /*this is for the case that there is a distinctive test action, which we do not model currently.
            if(!aAttributes[split.Attribute])
            {
                if (bObservedValue == false)
                    return 1.0;
                return 0.0;
            }
            */
            if(ExamplesAttributes[iExample][split.Attribute] >= split.Split)
            {
                if (bObservedValue)
                    return 1.0;
            }
            else
            {
                if (!bObservedValue)
                    return 1.0;
            }
            return 0.0;
        }
        public double R(int iState, Action a)
        {
            ToState(iState, out int iExample, out bool[] aAttributes);
            if(a is ClassificationAction c)
            {
                return -1 * ClassificationCosts[c.Class, ExamplesClasses[iExample]];
            }
            if(a is SplitOnAttribtueAction split)
            {
                if (aAttributes[split.Attribute])
                    return 0.0;
                if (ApplyGroupCost(aAttributes))
                    return -1 * TestCosts[split.Attribute, GROUP];
                else
                    return -1 * TestCosts[split.Attribute, SINGLE];
            }
            return 0.0;
        }

        public double R(bool[] aAttributes, Action a)
        {
            if (a is SplitOnAttribtueAction split)
            {
                if (aAttributes[split.Attribute])
                    return 0.0;
                if (ApplyGroupCost(aAttributes))
                    return -1 * TestCosts[split.Attribute, GROUP];
                else
                    return -1 * TestCosts[split.Attribute, SINGLE];
            }
            return 0.0;
        }

        public bool ApplyGroupCost(bool[] aObserved)
        {
            for(int i = 0; i < aObserved.Length; i++)
            {
                if (GroupCosts[i] && aObserved[i])
                    return true;
            }
            return false;
        }

        static int GROUP = 0, SINGLE = 1;

        //breast;no:no:5.02877:discrete.
        //menopause;yes:yes:1.84928:9.2464:discrete.


        public void LoadTestCosts(string sFileName)
        {
            Discount = 0.999;
            //Discount = 1.0;
            TestCosts = null;
            AttributeNames = new List<string>();
            GroupCosts = new List<bool>();
            List<string> lAttributes = new List<string>();
            using (StreamReader sr = new StreamReader(sFileName))
            {
                string sLine = sr.ReadLine(); //classes
                sLine = sLine.Replace(".", "");
                string[] aClasses = sLine.Trim().Split(',');
                Classes = aClasses.Length;
                ClassNameToIndex = new Dictionary<string, int>();
                for (int j = 0; j < Classes; j++)
                    ClassNameToIndex[aClasses[j]] = j;
                while (!sr.EndOfStream)
                {
                    sLine = sr.ReadLine();
                    lAttributes.Add(sLine);
                }
            }
            Attributes = lAttributes.Count;
            TestCosts = new double[Attributes, 2];
            for (int i = 0; i < lAttributes.Count; i++)
            {
                string sLine = lAttributes[i];
                string[] a = sLine.Split(';');
                AttributeNames.Add(a[0]);
                string[] a1 = a[1].Split(':');
                TestCosts[i, GROUP] = double.Parse(a1[2]);
                if (a1[1] == "yes" || a1[0] == "yes")
                {
                    GroupCosts.Add(true);
                    TestCosts[i, SINGLE] = double.Parse(a1[3]);
                }
                else
                {
                    GroupCosts.Add(false);
                    TestCosts[i, SINGLE] = TestCosts[i, GROUP];
                }
            }
        }
        public void LoadClassificationCosts(string sFileName)
        {
            ClassificationCosts = null;
            using (StreamReader sr = new StreamReader(sFileName))
            {
                int iPredicted = 0;
                while (!sr.EndOfStream)
                {
                    string sLine = sr.ReadLine().Trim();
                    string[] a = sLine.Split(new char[] { ' ', '\t' });
                    if (ClassificationCosts == null)
                        ClassificationCosts = new double[a.Length, a.Length];
                    for (int iTrueClass = 0; iTrueClass < a.Length; iTrueClass++)
                        ClassificationCosts[iPredicted, iTrueClass] = int.Parse(a[iTrueClass]);
                    iPredicted++;
                }
            }
        }

        public void InitClassificationVectors(List<AlphaVector> lClassificationVectors)
        {
            for (int iClass = 0; iClass < Classes; iClass++)
            {
                lClassificationVectors.Add(new AlphaVector(this));
                lClassificationVectors[iClass].Action = new ClassificationAction(iClass);
            }
            for (int iExample = 0; iExample < ExamplesClasses.Count; iExample++)
            {
                for (int iClass = 0; iClass < Classes; iClass++)
                    lClassificationVectors[iClass].Values[iExample] = ClassificationCosts[ExamplesClasses[iExample], iClass] * -1;//matrix specifies costs, not rewards
            }
        }
        public void InitClassificationVectors(List<SimpleAlphaVector> lClassificationVectors)
        {
            int cStates = ExamplesAttributes.Count * (int)Math.Pow(2, Attributes);
            for (int iPredictedClass = 0; iPredictedClass < Classes; iPredictedClass++)
            {
                lClassificationVectors.Add(new SimpleAlphaVector(this, cStates));
                lClassificationVectors[iPredictedClass].Action = new ClassificationAction(iPredictedClass);
            }
            for (int iState = 0; iState < cStates; iState++)
            {
                ToState(iState, out int iExample, out bool[] a);
                for (int iPredictedClass = 0; iPredictedClass < Classes; iPredictedClass++)
                    lClassificationVectors[iPredictedClass][iState] = ClassificationCosts[iPredictedClass, ExamplesClasses[iExample]] * -1;//matrix specifies costs, not rewards
            }
        }
        public void Backup(SimpleBelief b, IEnumerable<SimpleAlphaVector> lValueFunction, List<SimpleAlphaVector> lClassificationVectors, int iAttribute, SimpleAlphaVector[] aBestAttribute)
        {
            double dBestAction = double.NegativeInfinity;
            aBestAttribute[iAttribute] = null;
            double dBestClassification = double.NegativeInfinity;
            SimpleAlphaVector avBestClassification = null;

            avBestClassification = GetBestVector(b, lClassificationVectors);
            dBestClassification = avBestClassification * b;

            if (b.AttributeValueDistribution[iAttribute].Keys.Count > 1)
            {
                foreach (int iValue in b.AttributeValueDistribution[iAttribute].Keys)
                {
                    SimpleAlphaVector av = ComputeActionVector(b, iAttribute, iValue, lValueFunction, lClassificationVectors);
                    if (av != null)
                    {
                        double dValue = av * b;
                        if (dValue > dBestAction)
                        {
                            dBestAction = dValue;
                            aBestAttribute[iAttribute] = av;
                        }
                    }
                }
            }

        }


        public bool HasImprovingSplit(SimpleBelief b, List<SimpleAlphaVector> lClassificationVectors, int iAttribute)
        {
            if (b.AttributeValueDistribution[iAttribute].Keys.Count > 1)
            {
                foreach (int iValue in b.AttributeValueDistribution[iAttribute].Keys)
                {
                    bool bImproved = HasImprovingSplit(b, lClassificationVectors, iAttribute, iValue);
                    if (bImproved)
                        return true;
                }
            }
            return false;
        }


        public bool HasImprovingSplit(SimpleBelief b, List<SimpleAlphaVector> lClassificationVectors)
        {

            for (int iAttribute = 0; iAttribute < Attributes; iAttribute++)
            {
                if (b.AttributeValueDistribution[iAttribute].Keys.Count > 1)
                {
                    bool bImproves = HasImprovingSplit(b, lClassificationVectors, iAttribute);
                    if (bImproves)
                        return true;
                }
            }

            return false;
        }

        public bool ConcurrentBackup = true;

        public SimpleAlphaVector Backup(SimpleBelief b, IEnumerable<SimpleAlphaVector> lValueFunction, List<SimpleAlphaVector> lClassificationVectors)
        {
            double dBestAction = double.NegativeInfinity;
            SimpleAlphaVector avBest = null;
            double dBestClassification = double.NegativeInfinity;
            SimpleAlphaVector avBestClassification = null;
            SimpleAlphaVector[] aBestAttribute = new SimpleAlphaVector[Attributes];
            SimpleAlphaVector[] aAuxBestAttribute = new SimpleAlphaVector[Attributes];

            avBestClassification = GetBestVector(b, lClassificationVectors);
            dBestClassification = avBestClassification * b;

            Thread[] aThreads = new Thread[Attributes];
            for (int iAttribute = 0; iAttribute < Attributes; iAttribute++)
            {
                if (b.AttributeValueDistribution[iAttribute].Keys.Count > 1)
                {
                    if(ConcurrentBackup)
                    {
                        int temp = iAttribute;
                        aThreads[iAttribute] = new Thread(() => Backup(b, lValueFunction, lClassificationVectors, temp, aBestAttribute));
                        aThreads[iAttribute].Name = b + ":" + iAttribute;
                        aThreads[iAttribute].Start();
                    }
                    else
                        Backup(b, lValueFunction, lClassificationVectors, iAttribute, aBestAttribute);
                    
                    
                }
            }
            
            for (int iAttribute = 0; iAttribute < Attributes; iAttribute++)
            {
                if (aThreads[iAttribute] != null)
                {
                    aThreads[iAttribute].Join();
                }
                if(aBestAttribute[iAttribute] != null)
                { 
                    double dValue = aBestAttribute[iAttribute] * b;
                    if (dValue > dBestAction)
                    {
                        dBestAction = dValue;
                        avBest = aBestAttribute[iAttribute];
                    }
                }
            }
            if (dBestAction < dBestClassification)
                return null;
            /*
            for (int iAttribute = 0; iAttribute < Attributes; iAttribute++)
            {
                if (aAuxBestAttribute[iAttribute] != null && !aAuxBestAttribute[iAttribute].Equals(aBestAttribute[iAttribute]))
                    Console.Write("*");
            }
            */
                /*
                foreach (int iValue in b.AttributeValueDistribution[iAttribute].Keys)
                {
                    SimpleAlphaVector av = ComputeActionVector(b, iAttribute, iValue, lValueFunction, lClassificationVectors);
                    if (av != null)
                    {
                        double dValue = av * b;
                        if (dValue > dBestAction)
                        {
                            dBestAction = dValue;
                            avBest = av;
                        }
                    }
                }

            }
        } */
                /*
                if (dBestClassification < dBestAction)
                {
                    lValueFunction.Add(avBest);
                }
                */
                return avBest;
        }



        private double TestCost(Belief b, int iAttribute)
        {
            if (b.ObservedAttributes.Contains(true))
                return TestCosts[iAttribute, 0];
            return TestCosts[iAttribute, 1];
        }
        private double TestCost(SimpleBelief b, int iAttribute)
        {
            if (b.ObservedAttributes[iAttribute])
                return 0.0;
            if (b.ObservedAttributes.Contains(true))
                return TestCosts[iAttribute, 0];
            return TestCosts[iAttribute, 1];
        }

        private SimpleAlphaVector GetBestVector(SimpleBelief b, IEnumerable<SimpleAlphaVector> lValueFunction, int cMinExamples)
        {
            double dBestAction = double.NegativeInfinity;
            SimpleAlphaVector avBest = null;


            foreach (SimpleAlphaVector av in lValueFunction)
            {
                //if (av.Action is ClassificationAction)
                //    Console.Write("*");
                if (av.Action is ClassificationAction || b.Examples.Count > cMinExamples)
                {
                    double dValue = av * b;
                    if (dValue > dBestAction)
                    {
                        dBestAction = dValue;
                        avBest = av;
                    }
                }
            }
            return avBest;
        }
        private SimpleAlphaVector GetBestVector(SimpleBelief b, IEnumerable<SimpleAlphaVector> lValueFunction)
        {
            return GetBestVector(b, lValueFunction, 0);
        }

        private AlphaVector GetBestVector(Belief b, IEnumerable<AlphaVector> lValueFunction, bool bConsiderObservedTests)
        {
            double dBestAction = double.NegativeInfinity;
            AlphaVector avBest = null;


            foreach (AlphaVector av in lValueFunction)
            {
                /*
                if (bConsiderObservedTests)
                    if (av.Action is SplitOnAttribtueAction && !b.ObservedAttributes[((SplitOnAttribtueAction)av.Action).Attribute])
                        continue;
                //this is for overfitting
                if (b.Examples.Count < 20 && !(av.Action is ClassificationAction))
                    continue;



                if (av.Action is AttributeTestAction)
                {
                    AttributeTestAction a = (AttributeTestAction)av.Action;
                    if (a.First && b.ObservedCount > 0)
                        continue;
                    if (!a.First && b.ObservedCount == 0)
                        continue;

                }
                */

                double dValue = av * b;
                if (dValue > dBestAction)
                {
                    dBestAction = dValue;
                    avBest = av;
                }

            }
            return avBest;
        }
        // alpha_a,b = r_a + \gamma \sum argmax_o,alpha \in V b alpha_a,o
        private AlphaVector ComputeActionVector(Belief b, int iAttribute, int iValue, IEnumerable<AlphaVector> lValueFunction, List<AlphaVector> lClassificationVectors)
        {
            AlphaVector avBestForTrue = null, avBestForFalse = null;
            double dMaxTrue = double.NegativeInfinity, dMaxFalse = double.NegativeInfinity;
            List<AlphaVector> lAll = new List<AlphaVector>(lValueFunction);
            lAll.AddRange(lClassificationVectors);

            Belief bTrue = null, bFalse = null;
            b.Split(iAttribute, iValue, out bTrue, out bFalse);

            if (bFalse.Examples.Count == 0 || bTrue.Examples.Count == 0)
                return null;

            //need to consider here as well only vectors that will be applicable at the next step. maybe implement av * b for real...
            foreach (AlphaVector av in lAll)
            {
                double dValueTrue = av * bTrue;
                double dValueFalse = av * bFalse;
                
                if (dValueTrue > dMaxTrue)
                {
                    dMaxTrue = dValueTrue;
                    avBestForTrue = av;
                }
                if (dValueFalse > dMaxFalse)
                {
                    dMaxFalse = dValueFalse;
                    avBestForFalse = av;
                }
            }
            //not good - need to consider only relevant examples for each
            //return avBestForTrue + avBestForFalse;
            //return Sum(b, iAttribute, iValue, avBestForTrue, avBestForFalse);
            return Sum(iAttribute, iValue, avBestForTrue, avBestForFalse);
        }

        private bool HasImprovingSplit(SimpleBelief b, List<SimpleAlphaVector> lClassificationVectors, int iAttribute, int iValue)
        {
            SimpleAlphaVector avBestForTrue = null, avBestForFalse = null;
            double dMaxTrue = double.NegativeInfinity, dMaxFalse = double.NegativeInfinity;

            SplitOnAttribtueAction a = new SplitOnAttribtueAction(iAttribute, iValue);
            SimpleBelief bTrue = b.Next(a, true), bFalse = b.Next(a, false);

            if (bFalse.Examples.Count == 0 || bTrue.Examples.Count == 0)
                return false;

            double dBest = double.NegativeInfinity;

                //need to consider here as well only vectors that will be applicable at the next step. maybe implement av * b for real...
            foreach (SimpleAlphaVector av in lClassificationVectors)
            {
                double dValueTrue = av * bTrue;
                double dValueFalse = av * bFalse;
                double dValue = av * b;

                if (dValue > dBest)
                    dBest = dValue;

                if (dValueTrue > dMaxTrue)
                {
                    dMaxTrue = dValueTrue;
                    avBestForTrue = av;
                }
                if (dValueFalse > dMaxFalse)
                {
                    dMaxFalse = dValueFalse;
                    avBestForFalse = av;
                }
            }

            double dAfterSplit = Discount * (dMaxTrue * bTrue.Examples.Count + dMaxFalse * bFalse.Examples.Count) / b.Examples.Count;

            if (!b.ObservedAttributes[iAttribute])
                dAfterSplit += R(b.ObservedAttributes, a);

            return dAfterSplit > dBest;
        }

        private SimpleAlphaVector ComputeActionVector(SimpleBelief b, int iAttribute, double dValue, IEnumerable<SimpleAlphaVector> lValueFunction, List<SimpleAlphaVector> lClassificationVectors)
        {
            SimpleAlphaVector avBestForTrue = null, avBestForFalse = null;
            double dMaxTrue = double.NegativeInfinity, dMaxFalse = double.NegativeInfinity;
            List<SimpleAlphaVector> lAll = new List<SimpleAlphaVector>(lValueFunction);
            lAll.AddRange(lClassificationVectors);


            //if (iAttribute == 0 )
            //    Console.Write("*");

            SplitOnAttribtueAction a = new SplitOnAttribtueAction(iAttribute, dValue);
            SimpleBelief bTrue = b.Next(a, true), bFalse = b.Next(a, false);

            if (bFalse.Examples.Count == 0 || bTrue.Examples.Count == 0)
                return null;

            foreach (SimpleAlphaVector av in lAll)
            {
                if (av != null)
                {
                    double dValueTrue = av * bTrue;
                    double dValueFalse = av * bFalse;

                    if (dValueTrue > dMaxTrue)
                    {
                        dMaxTrue = dValueTrue;
                        avBestForTrue = av;
                    }
                    if (dValueFalse > dMaxFalse)
                    {
                        dMaxFalse = dValueFalse;
                        avBestForFalse = av;
                    }
                }
            }
            SimpleAlphaVector avGTrue = avBestForTrue.G(a, true);
            SimpleAlphaVector avGFalse = avBestForFalse.G(a, false);

            SimpleAlphaVector avSum = avGTrue  + avGFalse;
            /*
            double d1 = avSum * b;
            double d2 = avBestForTrue * bTrue;
            double d3 = avBestForFalse * bFalse;
            double dPrTrue = bTrue.Examples.Count() / (1.0 * b.Examples.Count());
            double dPrFalse = bFalse.Examples.Count() / (1.0 * b.Examples.Count());
            if (Math.Abs(d1 - (d2 * dPrTrue + d3 * dPrFalse)) > 0.01)
                Console.Write("*");

            int iState = CompareVectors(a, b, bTrue, bFalse, avSum, avBestForTrue, avBestForFalse);
            if (iState != -1)
                Console.Write("*");
                */
            avSum.AddRewardAndDiscount(a);
            avSum.Action = a;
            
            return avSum;
        }

        private int CompareVectors(Action a, SimpleBelief b, SimpleBelief bTrue, SimpleBelief bFalse, SimpleAlphaVector aSum, SimpleAlphaVector aTrue, SimpleAlphaVector aFalse)
        {
            for(int iState = 0; iState < States; iState++)
            {
                if(b[iState] > 0)
                {
                    double dValue = aSum[iState];
                    int iNextState = Tr(iState, a);
                    double dTrue = 0.0, dFalse = 0.0;
                    if (bTrue[iNextState] > 0)
                        dTrue = aTrue[iNextState];
                    if (bFalse[iNextState] > 0)
                        dFalse = aFalse[iNextState];
                    if (dFalse != dValue && dTrue != dValue)
                        return iState;
                }
            }
            return -1;
        }

        // alpha_a,b = r_a + \gamma \sum argmax_o,alpha \in V b alpha_a,o
        private AlphaVector ComputeActionVectorII(Belief b, int iAttribute, int iValue, List<AlphaVector> lValueFunction, List<AlphaVector> lClassificationVectors)
        {
            AlphaVector avBestForTrue = null, avBestForFalse = null;
            double dMaxTrue = double.NegativeInfinity, dMaxFalse = double.NegativeInfinity;
            List<AlphaVector> lAll = new List<AlphaVector>(lValueFunction);
            lAll.AddRange(lClassificationVectors);
            foreach (AlphaVector av in lAll)
            {
                double dValueTrue = 0.0, dValueFalse = 0.0;
                foreach (int iExample in b.Examples)
                {
                    if (ExamplesAttributes[iExample][iAttribute] >= iValue)
                        dValueTrue += av.Values[iExample];
                    else
                        dValueFalse += av.Values[iExample];
                }
                if (dValueTrue > dMaxTrue)
                {
                    dMaxTrue = dValueTrue;
                    avBestForTrue = av;
                }
                if (dValueFalse > dMaxFalse)
                {
                    dMaxFalse = dValueFalse;
                    avBestForFalse = av;
                }
            }
            //not good - need to consider only relevant examples for each
            //return avBestForTrue + avBestForFalse;
            return Sum(b, iAttribute, iValue, avBestForTrue, avBestForFalse);
        }

 

        private AlphaVector Sum(Belief b, int iAttribute, int iValue, AlphaVector avBestForTrue, AlphaVector avBestForFalse)
        {
            AlphaVector avSum = new AlphaVector(this);
            foreach (int iExample in b.Examples)
            {
                if (ExamplesAttributes[iExample][iAttribute] >= iValue)
                    avSum.Values[iExample] = avBestForTrue.Values[iExample] * Discount;
                else
                    avSum.Values[iExample] = avBestForFalse.Values[iExample] * Discount;
            }
            avSum.Successors.Add(avBestForTrue);
            avSum.Successors.Add(avBestForFalse);
            return avSum;
        }
        //Sum should be independent of b
        private AlphaVector Sum(int iAttribute, int iValue, AlphaVector avBestForTrue, AlphaVector avBestForFalse)
        {
            AlphaVector avSum = new AlphaVector(this);
            for(int iExample = 0; iExample < States; iExample++)
            {
                if (ExamplesAttributes[iExample][iAttribute] >= iValue)
                    avSum.Values[iExample] = avBestForTrue.Values[iExample] * Discount;
                else
                    avSum.Values[iExample] = avBestForFalse.Values[iExample] * Discount;
            }
            avSum.Successors.Add(avBestForTrue);
            avSum.Successors.Add(avBestForFalse);
            return avSum;
        }

    }
}
