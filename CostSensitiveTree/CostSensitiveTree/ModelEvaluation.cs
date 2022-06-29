using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using weka.core;

namespace CostSensitiveTree
{
    class ModelEvaluation
    {
        static Instances instances = null;
        
        public static void SetInstances(Instances choosedInstances)
        {
            instances = choosedInstances;
        }
        public static KeyValuePair<double,double> EvaluateModel(MDPModel mdpModel, string sDataPath, HashSet<string> initialyKnow)
        {
            int trueCounter = 0;
            double cost = 0.0;
            Dictionary<string, int> actionsCounter = new Dictionary<string, int>();
            List<string> allData = new List<string>();
            System.Console.WriteLine();
            System.Console.WriteLine("Evaluate Model:");
            double utilityValue = 0.0;
           

            var instancesIterator = instances.enumerateInstances();
            counter = 0;
            double success = 0.0;
            while (instancesIterator.hasMoreElements())
            {
                cost=0;

                Instance instance = (Instance)instancesIterator.nextElement();
               /* if (counter != 117)
                {
                    counter++;
                    continue;
                }
                counter++;*/
                Dictionary<string, Formula> allAttributesValue = Path.GetAttributesValue(instance);
                Dictionary<string, Formula> knownAttributesValue = new Dictionary<string, Formula>();
                Dictionary<string, Formula> knownAttributesValueForReduceStates = new Dictionary<string, Formula>();
                foreach (string attName in initialyKnow)
                {
                    knownAttributesValue.Add(attName, allAttributesValue[attName]);
                }


                State properState = GetProperState(knownAttributesValue, mdpModel);
                foreach (Tree.Node node in properState.path)
                {
                    string attName = node.formula.attributeName;
                    if (!knownAttributesValueForReduceStates.ContainsKey(attName))
                    {
                        if (mdpModel.dataSet.groupTests.Intersect(knownAttributesValueForReduceStates.Keys).Count() > 0)
                        {
                            utilityValue += mdpModel.dataSet.m_features[attName].GroupedCost;
                            cost += mdpModel.dataSet.m_features[attName].GroupedCost;
                        }
                        else
                        {
                            utilityValue += mdpModel.dataSet.m_features[attName].cost;
                            cost += mdpModel.dataSet.m_features[attName].cost;
                        }
                        knownAttributesValueForReduceStates.Add(attName, allAttributesValue[attName]);
                    }
                }

                string policy = "";
                Action action = properState.bestAction;
                policy += action.ToString() + ",  ";
                HashSet<string> attNames = new HashSet<string>();
                while (!action.isClassificationAction)
                {
                    //System.Console.WriteLine(action.ToString());

                    if (mdpModel.dataSet.groupTests.Intersect(knownAttributesValueForReduceStates.Keys).Count() > 0)
                    {
                        utilityValue += mdpModel.dataSet.m_features[action.attributeName].GroupedCost;
                        cost += mdpModel.dataSet.m_features[action.attributeName].GroupedCost;
                    }
                    else
                    {
                        utilityValue += mdpModel.dataSet.m_features[action.attributeName].cost;
                        cost += mdpModel.dataSet.m_features[action.attributeName].cost;
                    }
                    if (knownAttributesValue.ContainsKey(action.attributeName))
                        throw new Exception("buuuggg");

                    knownAttributesValue.Add(action.attributeName, allAttributesValue[action.attributeName]);
                    knownAttributesValueForReduceStates.Add(action.attributeName, allAttributesValue[action.attributeName]);

                    properState = GetProperState(knownAttributesValue, mdpModel);
                    foreach (Tree.Node node in properState.path)
                    {
                        string attName = node.formula.attributeName;
                        if (!knownAttributesValueForReduceStates.ContainsKey(attName))
                        {
                            if (mdpModel.dataSet.groupTests.Intersect(knownAttributesValueForReduceStates.Keys).Count() > 0)
                            {
                                utilityValue += mdpModel.dataSet.m_features[attName].GroupedCost;
                                cost += mdpModel.dataSet.m_features[attName].GroupedCost;
                            }
                            else
                            {
                                utilityValue += mdpModel.dataSet.m_features[attName].cost;
                                cost += mdpModel.dataSet.m_features[attName].cost;
                            }
                            knownAttributesValueForReduceStates.Add(attName, allAttributesValue[attName]);
                        }
                    }
                    action = properState.bestAction;
                    policy += action.ToString() + ",  ";
                }
                policy = policy.Remove(policy.Length - 3, 3);
                
                if (!actionsCounter.ContainsKey(policy))
                {
                    actionsCounter.Add(policy, 0);
                }
                actionsCounter[policy]++;
                int attCount = mdpModel.dataSet.m_testInstances.numAttributes();
                string trueClass = instance.stringValue(attCount-1);
                if (action.classification_string == trueClass)
                {
                    success++;
                    utilityValue += mdpModel.dataSet.m_classificationCostMetrix[action.classification_string][trueClass];
                    cost+= mdpModel.dataSet.m_classificationCostMetrix[action.classification_string][trueClass];
                    policy += "   " + action.classification_int + "/" + trueClass + " - True   // Cost: " + cost;
                    //policy = "true";
                    trueCounter++;
                }
                else
                {
                    utilityValue += mdpModel.dataSet.m_classificationCostMetrix[action.classification_string][trueClass];
                    cost += mdpModel.dataSet.m_classificationCostMetrix[action.classification_string][trueClass];
                    policy += "   " + action.classification_int + "/" + trueClass + " - False   // Cost: " + cost;
                    //policy = "false";
                }
                allData.Add(policy);
                //System.Console.WriteLine(action.ToString());
                //System.Console.WriteLine("Utility Value: " + utilityValue);
                //System.Console.ReadKey();
                System.Console.Write("\r{0}%   ", Math.Round(100 * (double)counter++ / (double)mdpModel.dataSet.m_testInstances.numInstances(), 3));
            }
            Console.WriteLine("\r100%                ");
            WriteToFile(sDataPath + @"Output\Mdp1.txt", allData);
            return new KeyValuePair<double, double>( utilityValue/ (double)instances.numInstances(), success/(double)instances.numInstances());
        }

        public static int counter = 0;
        public static KeyValuePair<double, double> EvaluateModelWithIntermediateUpdates(MDPModel mdpModel, string sDataPath, HashSet<string> initialyKnow)
        {
            counter = 0;
            Dictionary<string, int> actionsCounter = new Dictionary<string, int>();
            double cost = 0.0;
            List<string> allData = new List<string>();
            System.Console.WriteLine();
            System.Console.WriteLine("Evaluate model with updates:");
            double utilityValue = 0.0;

            var instancesIterator = instances.enumerateInstances();
            
            double success = 0.0;
            DateTime start = DateTime.Now;
            while (instancesIterator.hasMoreElements())
            {
                //System.Console.WriteLine("Ins Number:  "+counter);
                Instance instance = (Instance)instancesIterator.nextElement();
               /* if (counter > 20)
                {
                    break;
                    counter++;
                    continue;
                }*/
                counter++;
                cost = 0;

                MDPModel cloneMdpModel = mdpModel.Clone();
                Dictionary<string, Formula> allAttributesValue = Path.GetAttributesValue(instance);
                Dictionary<string, Formula> knownAttributesValue = new Dictionary<string, Formula>();
                Dictionary<string, Formula> knownAttributesValueForReduceStates = new Dictionary<string, Formula>();

                foreach (string attName in initialyKnow)
                {                  
                    knownAttributesValue.Add(attName, allAttributesValue[attName]);
                }

                List<State> propersStates = new List<State>();
                State properState = GetProperState(knownAttributesValue, cloneMdpModel);
                propersStates.Add(properState);
                foreach (Tree.Node node in properState.path)
                {
                    string attName = node.formula.attributeName;
                    if(!knownAttributesValueForReduceStates.ContainsKey(attName))
                    {
                        if (mdpModel.dataSet.groupTests.Intersect(knownAttributesValueForReduceStates.Keys).Count() > 0)
                        {
                            utilityValue += mdpModel.dataSet.m_features[attName].GroupedCost;
                            cost += mdpModel.dataSet.m_features[attName].GroupedCost;
                        }
                        else
                        {
                            utilityValue += mdpModel.dataSet.m_features[attName].cost;
                            cost += mdpModel.dataSet.m_features[attName].cost;
                        }
                        knownAttributesValueForReduceStates.Add(attName, allAttributesValue[attName]);
                    }
                }
                string policy = "";
                Action action = properState.bestAction;
                policy += action.ToString() + ",  ";
                HashSet<string> attNames = new HashSet<string>();
                while (!action.isClassificationAction)
                {

                    if (false)
                    {
                        // remove after debug
                        if (!action.isClassificationAction)
                        {
                            foreach (Path leafPath in properState.leafsPaths.Select(kv => kv.Key))
                            {
                                if (!leafPath.GetNodeList().Select(n => n.formula.attributeName).Contains(action.attributeName))
                                {
                                    bool f = true;
                                    foreach (string attName in leafPath.GetNodeList().Select(n => n.formula.attributeName))
                                    {
                                        if (!knownAttributesValueForReduceStates.Keys.Contains(attName))
                                            f = false;
                                    }
                                    if (f)
                                        Console.WriteLine("BUG");

                                }
                            }
                        }
                        //remove after debug
                        foreach (State s in cloneMdpModel.stateList)
                        {
                            foreach (Action a in s.outgoingActions)
                            {
                                if (!a.isClassificationAction)
                                {
                                    double probSum = a.targetStates.Values.Sum();
                                    if (probSum > 1.0001 || probSum < 0.999)
                                        Console.WriteLine("BUG");
                                }
                            }
                        }
                    }

                    //System.Console.WriteLine(action.ToString());
                    // System.Console.ReadKey();
                    if (attNames.Contains(action.attributeName))
                        throw new Exception("buuuggg");
                    else
                    {
                        attNames.Add(action.attributeName);
                    }
                    if (mdpModel.dataSet.groupTests.Intersect(knownAttributesValueForReduceStates.Keys).Count() > 0)
                    {
                        utilityValue += mdpModel.dataSet.m_features[action.attributeName].GroupedCost;
                        cost += mdpModel.dataSet.m_features[action.attributeName].GroupedCost;
                    }
                    else
                    {
                        utilityValue += mdpModel.dataSet.m_features[action.attributeName].cost;
                        cost += mdpModel.dataSet.m_features[action.attributeName].cost;
                    }
                    if (knownAttributesValue.ContainsKey(action.attributeName))
                        throw new Exception("buuuggg");
                    knownAttributesValue.Add(action.attributeName, allAttributesValue[action.attributeName]);
                    knownAttributesValueForReduceStates.Add(action.attributeName, allAttributesValue[action.attributeName]);
                    cloneMdpModel.UpdateStateAndAction(knownAttributesValueForReduceStates);
                    properState = GetProperState(knownAttributesValue, cloneMdpModel);
                    foreach (Tree.Node node in properState.path)
                    {
                        string attName = node.formula.attributeName;
                        if (!knownAttributesValueForReduceStates.ContainsKey(attName))
                        {
                            if (mdpModel.dataSet.groupTests.Intersect(knownAttributesValueForReduceStates.Keys).Count() > 0)
                            {
                                utilityValue += mdpModel.dataSet.m_features[attName].GroupedCost;
                                cost += mdpModel.dataSet.m_features[attName].GroupedCost;
                            }
                            else
                            {
                                utilityValue += mdpModel.dataSet.m_features[attName].cost;
                                cost += mdpModel.dataSet.m_features[attName].cost;
                            }
                            knownAttributesValueForReduceStates.Add(attName, allAttributesValue[attName]);
                        }
                    }
                    action = properState.bestAction;
                    policy += action.ToString() + ",  ";
                    propersStates.Add(properState);
                }

                policy = policy.Remove(policy.Length - 3, 3);
                
                if (!actionsCounter.ContainsKey(policy))
                {
                    actionsCounter.Add(policy, 0);
                }
                actionsCounter[policy]++;
                int attCount = mdpModel.dataSet.m_testInstances.numAttributes();
                string trueClass = instance.stringValue(attCount - 1);
                if (action.classification_string == trueClass)
                {
                    success++;
                    utilityValue += mdpModel.dataSet.m_classificationCostMetrix[action.classification_string][trueClass];
                    cost += mdpModel.dataSet.m_classificationCostMetrix[action.classification_string][trueClass];
                    policy += "   " + action.classification_int + "/" + trueClass + " - True   // Cost: " + cost;
                }
                else
                {
                    utilityValue += mdpModel.dataSet.m_classificationCostMetrix[action.classification_string][trueClass];
                    cost += mdpModel.dataSet.m_classificationCostMetrix[action.classification_string][trueClass];
                    policy += "   "+ action.classification_int +"/"+trueClass+ " - False   // Cost: " + cost;
                }
                allData.Add(policy);
                //System.Console.WriteLine(action.ToString());
                //System.Console.WriteLine("Utility Value: " + utilityValue);
                //System.Console.ReadKey();
                System.Console.Write("\r{0}%   ", Math.Round((double)counter++ / (double)mdpModel.dataSet.m_testInstances.numInstances(), 3));
                counter++;
            }

            double time = DateTime.Now.Subtract(start).TotalSeconds/ (double)instances.numInstances();
            WriteToFile(sDataPath + @"Output\Mdp2.txt", allData);
            return new KeyValuePair<double, double>(utilityValue/ (double)instances.numInstances(), success / (double)instances.numInstances());
        }



        private static State GetProperState(Dictionary<string, Formula> knownAttributesValue, MDPModel mdpModel)
        {
            List<State> properStates = new List<State>();
            List<State> relevantStates = mdpModel.stateList.Where(s => (s.attributes.Count== knownAttributesValue.Count ||s.bestAction.isClassificationAction) && s.attributes.Union(knownAttributesValue.Keys).Count() == s.attributes.Count).ToList();
             foreach (State state in relevantStates)
            {
                if (state.ConsistWith(knownAttributesValue))
                    properStates.Add(state);
            }
            if(properStates.Count==0)
                throw new Exception("Proper state not founded");


            properStates = properStates.OrderByDescending(s => s.bestValue + s.restTestCoST).ToList();
            return properStates[0];
        }


       

        public static KeyValuePair<double, double> EvaluateBaseModel(string sDataPath, List<Path> paths, DataReader.DataSet dataSet,Tree.TreeType treeType)
        {
            var m_classificationCostMetrix = dataSet.m_classificationCostMetrix;

           /* if(treeType==Tree.TreeType.MetaCost)
            {
                Dictionary<string, Dictionary<string, double>> tClassificationCostMetrix = new Dictionary<string, Dictionary<string, double>>();
                foreach(string row in m_classificationCostMetrix.Keys)
                {
                    foreach (string col in m_classificationCostMetrix[row].Keys)
                    {
                        if (!tClassificationCostMetrix.ContainsKey(col))
                            tClassificationCostMetrix.Add(col, new Dictionary<string, double>());
                        if (!tClassificationCostMetrix[col].ContainsKey(row))
                            tClassificationCostMetrix[col].Add(row, m_classificationCostMetrix[row][col]);
                    }
                }
                m_classificationCostMetrix = tClassificationCostMetrix;
            }*/


            Dictionary<string, int> policyCounter = new Dictionary<string, int>();
            double cost = 0.0;
            List<string> allData = new List<string>();
            //System.Console.WriteLine();
            //System.Console.WriteLine("Evaluate Base Model:");
            double utilityValue = 0.0;

            var instancesIterator = instances.enumerateInstances();
            int counter = 1;
            double success = 0.0;
            while (instancesIterator.hasMoreElements())
            {
                cost = 0.0;
                counter++;
                Instance instance = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> allAttributesValue = Path.GetAttributesValue(instance);
                Dictionary<string, Formula> knownAttributesValue = new Dictionary<string, Formula>();
                List<Path> modelByPaths = new List<Path>(paths);
                string policy = "";
                int attIndex = 0;
                while (modelByPaths.Count > 1)
                {
                    string attName = modelByPaths[0].GetNode(attIndex).formula.attributeName;
                
                    if (!knownAttributesValue.ContainsKey(attName))
                    {
                        policy += "Query on: " + attName + ",  ";
                        if (dataSet.groupTests.Intersect(knownAttributesValue.Keys).Count() > 0)
                        {
                            utilityValue += dataSet.m_features[attName].GroupedCost;
                            cost += dataSet.m_features[attName].GroupedCost;
                        }
                        else
                        {
                            utilityValue += dataSet.m_features[attName].cost;
                            cost += dataSet.m_features[attName].cost;
                        }
                        knownAttributesValue.Add(attName, allAttributesValue[attName]);
                    }

                    modelByPaths = UpdatePathListByAns(allAttributesValue[attName], modelByPaths);
                    attIndex++;
                }
                string modelClassify = modelByPaths[0].lastNode.classValue_String;
                policy += "Classify As: " + modelClassify;

                if (!policyCounter.ContainsKey(policy))
                {
                    policyCounter.Add(policy, 0);
                }
                policyCounter[policy]++;
                int attCount = dataSet.m_testInstances.numAttributes();
                string trueClass = instance.stringValue(attCount - 1);
                if (modelClassify == trueClass)
                {
                    success++;
                    utilityValue += m_classificationCostMetrix[modelClassify][trueClass];
                    cost += m_classificationCostMetrix[modelClassify][trueClass];
                    policy += "   " + modelClassify + "/" + trueClass + " - True   // Cost: " + cost;
                    //policy = "true";
                }
                else
                {
                    utilityValue += m_classificationCostMetrix[modelClassify][trueClass];
                    cost += m_classificationCostMetrix[modelClassify][trueClass];
                    policy += "   " + modelClassify + "/" + trueClass + " - False   // Cost: " + cost;
                    //policy = "false";
                }
                allData.Add(policy);
                // System.Console.WriteLine(action.ToString());
                //System.Console.WriteLine("Utility Value: " + utilityValue);

            }
            Console.WriteLine(" UtilityValue: " + utilityValue / (double)instances.numInstances() + " SuccessRatio: " + success / (double)instances.numInstances());
            WriteToFile(sDataPath + @"Output\BaseModel_"+treeType+"_.txt", allData);
            return new KeyValuePair<double, double>(utilityValue/ (double)instances.numInstances(), success / (double)instances.numInstances());
        }

        public static void WriteToFile(string fileName,List<string> data)
        {
            StreamWriter writer = new StreamWriter(fileName, false);
            int counter = 0;
            foreach (string line in data)
            {
                writer.WriteLine(counter+"   "+line);
                writer.WriteLine();
                counter++;
            }
            writer.Close();
        }

        public static List<Path> UpdatePathListByAns(Formula f,List<Path> orgList)
        {
            List<Path> restPaths = new List<Path>();
            foreach(Path path in orgList)
            {
                var itemsWithIdenticalName = path.GetNodeList().Where(n => n.formula.attributeName==f.attributeName);
                var itemsWithIdenticalNameAndWeaker = itemsWithIdenticalName.Where(n => f.IsWeakStronger(n.formula));
                if (itemsWithIdenticalName.Count() == itemsWithIdenticalNameAndWeaker.Count())
                    restPaths.Add(path);
            }
            return restPaths;
        }

        public static double CalculateTreeExUtility(List<Path> paths, DataReader.DataSet dataSet)
        {
            int debugCounter = 0;
            int successesCounter = 0;

            double globalUtility = 0.0;
            foreach(var path in paths)
            {
                HashSet<string> knownAttributesValue = new HashSet<string>();
                double testsCost = 0.0;
                foreach (Tree.Node node in path)
                {                   
                    if (!node.isLeafNode)
                    {
                        string attName = node.formula.attributeName;
                        if (!knownAttributesValue.Contains(attName))
                        {
                            if (dataSet.groupTests.Intersect(knownAttributesValue).Count() > 0)
                            {
                                testsCost += dataSet.m_features[attName].GroupedCost;
                            }
                            else
                            {
                                testsCost += dataSet.m_features[attName].cost;
                            }
                            knownAttributesValue.Add(attName);
                        }
                    }
                    else
                    {
                        double successesCount = double.Parse(node.ratioStr.Split('/')[0]);
                        double failuresCount = double.Parse(node.ratioStr.Split('/')[1]);
                        successesCount = successesCount - failuresCount;
                        globalUtility += successesCount * ( DataReader.trueClassificationUtility) - (failuresCount *
                    -DataReader.trueClassificationUtility);
                        globalUtility += testsCost * (successesCount + failuresCount);

                        debugCounter += (int)successesCount;
                        debugCounter += (int)failuresCount;
                        successesCounter+=(int)successesCount;
                    }
                }
            }
            if (dataSet.m_trainInstances2.numInstances() != debugCounter)
                throw new Exception("Bug");

            return globalUtility;
        }

        public static KeyValuePair<double, double> EvaluateModelByGraph(MDPModel mdpModel, string sDataPath, HashSet<string> initialyKnow)
        {
            int trueCounter = 0;
            double cost = 0.0;
            Dictionary<string, int> actionsCounter = new Dictionary<string, int>();
            List<string> allData = new List<string>();
            System.Console.WriteLine();
            System.Console.WriteLine("Evaluate Model:");
            double utilityValue = 0.0;


            var instancesIterator = instances.enumerateInstances();
            counter = 0;
            double success = 0.0;

            State initState = mdpModel.stateList.FirstOrDefault(s => s.attributes.Count == 0);
            List<State.PathInfo>  relevantPaths = new List<State.PathInfo>();
            List<State.PathInfo> setOfPaths = new List<State.PathInfo>();
            State.FindPaths(initState, setOfPaths, null);

            foreach (State.PathInfo pi in setOfPaths)
            {
                State.PathInfo prev = pi.m_Parent;
                Path d = new Path();
                d.accuracy = pi.m_Path.accuracy;
                foreach (Tree.Node node in pi.m_Path.GetNodeList())
                    d.AddUniqueNode(node);
                int depth = 0;
                while (prev != null)
                {
                    foreach (Tree.Node node in prev.m_Path.GetNodeList())
                        d.AddUniqueNode(node);
                    prev = prev.m_Parent;
                    depth++;
                }
                relevantPaths.Add(new State.PathInfo(d,pi.m_Action,null,depth));
            }





            while (instancesIterator.hasMoreElements())
            {
                List<State.PathInfo> copyOfRelevantPaths = new List<State.PathInfo>(relevantPaths);
                cost = 0;

                Instance instance = (Instance)instancesIterator.nextElement();

                Dictionary<string, Formula> allAttributesValue = Path.GetAttributesValue(instance);
                Dictionary<string, Formula> knownAttributesValue = new Dictionary<string, Formula>();
                Dictionary<string, Formula> knownAttributesValueForReduceStates = new Dictionary<string, Formula>();
                foreach (string attName in initialyKnow)
                {
                    knownAttributesValue.Add(attName, allAttributesValue[attName]);
                }


                State.PathInfo properState = GetProperPathInfo(knownAttributesValue, copyOfRelevantPaths);
                foreach (Tree.Node node in properState.m_Path)
                {
                    string attName = node.formula.attributeName;
                    if (!knownAttributesValueForReduceStates.ContainsKey(attName))
                    {
                        if (mdpModel.dataSet.groupTests.Intersect(knownAttributesValueForReduceStates.Keys).Count() > 0)
                        {
                            utilityValue += mdpModel.dataSet.m_features[attName].GroupedCost;
                            cost += mdpModel.dataSet.m_features[attName].GroupedCost;
                        }
                        else
                        {
                            utilityValue += mdpModel.dataSet.m_features[attName].cost;
                            cost += mdpModel.dataSet.m_features[attName].cost;
                        }
                        knownAttributesValueForReduceStates.Add(attName, allAttributesValue[attName]);
                    }
                }

                string policy = "";
                Action action = properState.m_Action;
                policy += action.ToString() + ",  ";
                HashSet<string> attNames = new HashSet<string>();
                while (!action.isClassificationAction)
                {
                    //System.Console.WriteLine(action.ToString());

                    if (mdpModel.dataSet.groupTests.Intersect(knownAttributesValueForReduceStates.Keys).Count() > 0)
                    {
                        utilityValue += mdpModel.dataSet.m_features[action.attributeName].GroupedCost;
                        cost += mdpModel.dataSet.m_features[action.attributeName].GroupedCost;
                    }
                    else
                    {
                        utilityValue += mdpModel.dataSet.m_features[action.attributeName].cost;
                        cost += mdpModel.dataSet.m_features[action.attributeName].cost;
                    }
                    if (knownAttributesValue.ContainsKey(action.attributeName))
                        throw new Exception("buuuggg");

                    knownAttributesValue.Add(action.attributeName, allAttributesValue[action.attributeName]);
                    knownAttributesValueForReduceStates.Add(action.attributeName, allAttributesValue[action.attributeName]);

                    properState = GetProperPathInfo(knownAttributesValue, copyOfRelevantPaths);
                    foreach (Tree.Node node in properState.m_Path)
                    {
                        string attName = node.formula.attributeName;
                        if (!knownAttributesValueForReduceStates.ContainsKey(attName))
                        {
                            if (mdpModel.dataSet.groupTests.Intersect(knownAttributesValueForReduceStates.Keys).Count() > 0)
                            {
                                utilityValue += mdpModel.dataSet.m_features[attName].GroupedCost;
                                cost += mdpModel.dataSet.m_features[attName].GroupedCost;
                            }
                            else
                            {
                                utilityValue += mdpModel.dataSet.m_features[attName].cost;
                                cost += mdpModel.dataSet.m_features[attName].cost;
                            }
                            knownAttributesValueForReduceStates.Add(attName, allAttributesValue[attName]);
                        }
                    }
                    action = properState.m_Action;
                    policy += action.ToString() + ",  ";
                }
                policy = policy.Remove(policy.Length - 3, 3);

                if (!actionsCounter.ContainsKey(policy))
                {
                    actionsCounter.Add(policy, 0);
                }
                actionsCounter[policy]++;
                int attCount = mdpModel.dataSet.m_testInstances.numAttributes();
                string trueClass = instance.stringValue(attCount - 1);
                if (action.classification_string == trueClass)
                {
                    success++;
                    utilityValue += mdpModel.dataSet.m_classificationCostMetrix[action.classification_string][trueClass];
                    cost += mdpModel.dataSet.m_classificationCostMetrix[action.classification_string][trueClass];
                    policy += "   " + action.classification_int + "/" + trueClass + " - True   // Cost: " + cost;
                    trueCounter++;
                }
                else
                {
                    utilityValue += mdpModel.dataSet.m_classificationCostMetrix[action.classification_string][trueClass];
                    cost += mdpModel.dataSet.m_classificationCostMetrix[action.classification_string][trueClass];
                    policy += "   " + action.classification_int + "/" + trueClass + " - False   // Cost: " + cost;
                }
                allData.Add(policy);
                //System.Console.WriteLine(action.ToString());
                //System.Console.WriteLine("Utility Value: " + utilityValue);
                //System.Console.ReadKey();
                System.Console.Write("\r{0}%   ", Math.Round(100 * (double)counter++ / (double)mdpModel.dataSet.m_testInstances.numInstances(), 3));
            }
            WriteToFile(sDataPath + @"Output\Mdp1.txt", allData);
            return new KeyValuePair<double, double>(utilityValue / (double)instances.numInstances(), success / (double)instances.numInstances());
        }
        private static State.PathInfo GetProperPathInfo(Dictionary<string, Formula> knownAttributesValue, List<State.PathInfo> pathsInfo)
        {
            List<State.PathInfo> properStates = new List<State.PathInfo>();
           foreach (State.PathInfo pathInfo in pathsInfo)
            {
                if (pathInfo.m_Path.ConsistWith(knownAttributesValue))
                    properStates.Add(pathInfo);
            }
            if (properStates.Count == 0)
                throw new Exception("Proper state not founded");

            properStates = properStates.OrderBy(ps => ps.m_Depth).ToList();
            var ans = properStates[0];
            pathsInfo.Remove(ans);
            return ans;
        }

    }
}
