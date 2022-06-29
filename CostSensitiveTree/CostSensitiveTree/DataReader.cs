using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using weka.core;

namespace CostSensitiveTree
{

    class DataReader
    {
        public static bool useMissClassificationFile=true;
        public static double trueClassificationUtility = 300.0;

        public class DataSet
        {
            public List<string[]> m_testSet;
            public List<string[]> m_trainSet;
            public Dictionary<string, Dictionary<string, double>> m_classificationCostMetrix;
            public Dictionary<string, FeatureData> m_features;
            public Instances m_testInstances;
            public Instances m_trainInstances1;
            public Instances m_trainInstances2;
            public HashSet<string> groupTests;
        }

        public static DataSet ReadData(string dataPath, int index,int costIndex)
        {
            StreamReader classificationCostFile = null;
            if (File.Exists(dataPath + @"\CM." + index.ToString()))
                classificationCostFile = new StreamReader(dataPath + @"\CM." + costIndex.ToString());
            StreamReader featuresFile = new StreamReader(dataPath + @"\data.names");
            StreamReader trainFile = new StreamReader(dataPath + @"\data" + index.ToString() + ".rr");
            StreamReader testFile = new StreamReader(dataPath + @"\data" + index.ToString() + ".ss");


            HashSet<string> m_groupTests = new HashSet<string>();
            Dictionary<string, Dictionary<string, double>> classificationCostMetrix = new Dictionary<string, Dictionary<string, double>>();
            Dictionary<string, FeatureData> features = new Dictionary<string, FeatureData>();
            int lineCounter = 0;
            List<string> classValues = new List<string>();

            while (!featuresFile.EndOfStream)
            {
                char[] del = new char[] { ' ', ';', ':' };

                string[] featuresStr = null;
                if (lineCounter == 0)
                {
                    del = new char[] { ',', '.' };
                    featuresStr = featuresFile.ReadLine().Split(del);
                    features.Add("Class", new FeatureData { name = "Class", type = "discrete", isClass = true, cost = 0, GroupedCost = 0, GroupedTest = false, attIndex = -1, SetOfValue = new HashSet<string>() });
                    lineCounter++;
                    foreach (string classVal in featuresStr)
                    {
                        if (classVal != "")
                            classValues.Add(classVal);
                    }
                    continue;
                }
                featuresStr = featuresFile.ReadLine().Split(del);
                string name = featuresStr[0];
                bool isGroup = featuresStr[2] == "yes" || featuresStr[1] == "yes";
                double cost = isGroup ? -double.Parse(featuresStr[4]) : -double.Parse(featuresStr[3]);
                double groupCost = isGroup ? -double.Parse(featuresStr[3]) : cost;
                if (isGroup)
                    m_groupTests.Add(name);
                string type = isGroup ? featuresStr[5].Replace(".", "") : featuresStr[4].Replace(".", "");
                features.Add(name, new FeatureData { name = name, type = type, isClass = false, cost = cost, GroupedCost = groupCost, GroupedTest = isGroup, attIndex = lineCounter - 1, SetOfValue = new HashSet<string>() });

                lineCounter++;
            }

            for (int i = 0; i < classValues.Count; i++)
            {
                classificationCostMetrix.Add(classValues[i], new Dictionary<string, double>());
                for (int j = 0; j < classValues.Count; j++)
                {

                    classificationCostMetrix[classValues[i]].Add(classValues[j], 0.0);
                }
            }

            features["Class"].SetIndex(lineCounter - 1); ;
            features = features.OrderBy(x => x.Value.attIndex).ToDictionary(x => x.Key, x => x.Value);
            lineCounter = 0;

            if (classificationCostFile != null && useMissClassificationFile)
            {
                while (!classificationCostFile.EndOfStream && lineCounter< classificationCostMetrix.Count)
                {
                    char[] del = { ' ', '\t' };
                    string[] costArray = classificationCostFile.ReadLine().Split(del);
                    string valName1 = classificationCostMetrix.ElementAt(lineCounter).Key;
                    for (int i = 0; i < costArray.Length; i++)
                    {
                        if (costArray[i] != "")
                        {
                            string valName2 = classificationCostMetrix[valName1].ElementAt(i).Key;
                            classificationCostMetrix[valName1][valName2] = -double.Parse(costArray[i]);
                        }
                    }
                    lineCounter++;
                }
            }
            else
            {
                for(int i=0;i< classificationCostMetrix.Count;i++)
                {
                    for (int j = 0; j < classificationCostMetrix.Count; j++)
                    {
                        string valName1 = classificationCostMetrix.ElementAt(i).Key;
                        string valName2 = classificationCostMetrix.ElementAt(j).Key;
                        if(i==j)
                        {
                            classificationCostMetrix[valName1][valName2] = trueClassificationUtility;
                        }
                        else
                        {
                            classificationCostMetrix[valName1][valName2] = -trueClassificationUtility;
                        }
                    }
                }
            }

            List<string[]> trainSet = new List<string[]>();
            while (!trainFile.EndOfStream)
            {
                string[] dataLine = trainFile.ReadLine().Split(new char[] { ' ', '\t' });
                trainSet.Add(dataLine);
                for (int i = 0; i < features.Count; i++)
                {
                    var relevantAtt = features.FirstOrDefault(x => x.Value.attIndex == i);
                    if (relevantAtt.Value.type != "discrete" && relevantAtt.Value.type != "continuous" && relevantAtt.Value.type != "boolean")
                    {
                        features[relevantAtt.Key].AddVal(dataLine[i]);
                    }
                }

            }

            List<string[]> testSet = new List<string[]>();
            while (!testFile.EndOfStream)
            {
                string[] dataLine = testFile.ReadLine().Split(new char[] { ' ', '\t' });
                testSet.Add(dataLine);
                for (int i = 0; i < features.Count; i++)
                {
                    var relevantAtt = features.FirstOrDefault(x => x.Value.attIndex == i);
                    if (relevantAtt.Value.type != "discrete" && relevantAtt.Value.type != "continuous" && relevantAtt.Value.type != "boolean")
                    {
                        features[relevantAtt.Key].AddVal(dataLine[i]);
                    }
                }
            }


            DataSet dataSet = new DataSet { m_testSet = testSet, m_trainSet = trainSet, m_classificationCostMetrix = classificationCostMetrix, m_features = features, groupTests= m_groupTests};
            dataSet = AddInstances(dataSet);
            return dataSet;
        }

        public class FeatureData
        {
            public int attIndex;
            public string name;
            public double cost;
            public double GroupedCost;
            public bool GroupedTest;
            public string type;
            public bool isClass;
            public HashSet<string> SetOfValue;
            public void SetIndex(int index)
            {
                attIndex = index;
            }
            public void AddVal(string val)
            {
                SetOfValue.Add(val);
            }
        }

  


        public static DataSet AddInstances(DataSet dataSet)
        {
            FastVector atts;
            atts = new FastVector();
            Dictionary<int, FastVector> mapAttIndexToFastVector = new Dictionary<int, FastVector>();
            foreach (FeatureData featureData in dataSet.m_features.Values)
            {
                if (featureData.name != "Class")
                {
                    if (featureData.SetOfValue.Count == 0)
                    {
                        weka.core.Attribute newAtt = new weka.core.Attribute(featureData.name);
                        newAtt.m_Cost = -featureData.cost;
                        atts.addElement(newAtt);
                    }
                    else
                    {
                        FastVector fvClassVal = new FastVector(featureData.SetOfValue.Count);
                        foreach (string val in featureData.SetOfValue)
                        {
                            fvClassVal.addElement(val);
                        }
                        weka.core.Attribute newAtt = new weka.core.Attribute(featureData.name, fvClassVal);
                        newAtt.m_Cost = -featureData.cost;
                        atts.addElement(newAtt);
                        mapAttIndexToFastVector.Add(featureData.attIndex, fvClassVal);
                    }
                }
                else
                {
                    FastVector classt = new FastVector();
                    foreach (string classVal in dataSet.m_classificationCostMetrix.ElementAt(0).Value.Keys)
                    {
                        classt.addElement(classVal);
                    }
                    mapAttIndexToFastVector.Add(featureData.attIndex, classt);
                    atts.addElement(new weka.core.Attribute("class", classt));
                }
            }


            Instances trainData1 = new Instances("Data", atts, 0);
            Instances trainData2 = new Instances("Data", atts, 0);
            double[] vals = null;

            bool twoTrainSets = true;
            int j = 0;
            int n = dataSet.m_trainSet.Count;
            foreach (string[] line in dataSet.m_trainSet)
            {
                j++;
                int i = 0;
                vals = new double[trainData1.numAttributes()];
                foreach (string field in line)
                {
                    if (mapAttIndexToFastVector.ContainsKey(i))
                    {
                        vals[i] = mapAttIndexToFastVector[i].indexOf(field);
                    }
                    else
                    {
                        vals[i] = Convert.ToDouble(field);
                    }
                    i++;
                }

                if (twoTrainSets)
                {
                    if (((double)j /(double) n) < 1.90)
                        trainData1.add(new Instance(1.0, vals));
                    trainData2.add(new Instance(1.0, vals));
                }
                else
                {
                    trainData1.add(new Instance(1.0, vals));
                    trainData2.add(new Instance(1.0, vals));
                }
            }

            trainData1.setClassIndex(trainData1.numAttributes() - 1);
            trainData2.setClassIndex(trainData2.numAttributes() - 1);

            Instances testData = new Instances("Data", atts, 0);
            vals = null;

            foreach (string[] line in dataSet.m_testSet)
            {
                int i = 0;
                vals = new double[testData.numAttributes()];
                foreach (string field in line)
                {
                    if (mapAttIndexToFastVector.ContainsKey(i))
                    {
                        vals[i] = mapAttIndexToFastVector[i].indexOf(field);
                    }
                    else
                    {
                        vals[i] = Convert.ToDouble(field);
                    }
                    i++;
                }
                testData.add(new Instance(1.0, vals));
            }
            testData.setClassIndex(testData.numAttributes() - 1);


            dataSet.m_testInstances = testData;
            dataSet.m_trainInstances1 = trainData1;
            dataSet.m_trainInstances2 = trainData2;

            return dataSet;
        }


        public static double CalculateMisClassificationValue(DataSet dataSet, Dictionary<string,double> TrueProbs,string classifay)
        {
            double val = 0.0;
            foreach(var trueProb in TrueProbs)
            {
                val += dataSet.m_classificationCostMetrix[classifay][trueProb.Key] * trueProb.Value;
            }
            return val;
        }

        public static string ChooseClass(DataSet dataSet, Dictionary<string,double> classDis)
        {
            string bestClass = null;
            double bestVal = double.MinValue;
            foreach(string optionalClass in classDis.Keys)
            {
                double localVal = 0.0;
                foreach (var trueProb in classDis)
                {
                    localVal += dataSet.m_classificationCostMetrix[optionalClass][trueProb.Key] * trueProb.Value;
                }
                if(localVal>bestVal)
                {
                    bestVal = localVal;
                    bestClass = optionalClass;
                }
            }
            return bestClass;
        }


    }
}
