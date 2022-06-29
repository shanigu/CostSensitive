using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using weka.core;

namespace CostSensitiveTree
{
    class State
    {
        double fowardTestsCost = 0;

        Dictionary<string, ClassToCount> TFNP;

        private static int ID = 0;
        public double bestValue { get; set; }
        public Path path { get; set; }
        public Action bestAction { get; set; }
        public int id { get; set; }
        public List<Action> ingoingActions { get; set;}
        public List<Action> outgoingActions { get; set; }
        public HashSet<string> attributes { get; set; }
        public List<Tree.Node> nodes { get; set; }
        public bool valid { get; set; } = true;

        public double restTestCoST { get; set; } = 0;

        public Dictionary<string, HashSet<Formula>> setOfReduceRules = null;
        public List<KeyValuePair<Path,double>> leafsPaths { get; set; }

        public Dictionary<string, double> classDistribution = null;

        public Dictionary<Instance, Dictionary<string, Formula>> instances = new Dictionary<Instance, Dictionary<string, Formula>>();
    

        public State(Path m_path)
        {
            id = ID;
            ID++;
            ingoingActions = new List<Action>();
            outgoingActions = new List<Action>();
            path = m_path;
            attributes = new HashSet<string>();
            nodes = new List<Tree.Node>();
            attributes = m_path.GetSourceTree().attributesNames;
            for (int i=0; i<path.Count() ; i++)
            {
                nodes.Add(path.GetNode(i));
            }
            setOfReduceRules = new Dictionary<string, HashSet<Formula>>();
            classDistribution = m_path.classDistribution;
            path.state = this;
        }

        public State Clone()
        {
            State newState = new State(path);
            newState.ingoingActions = new List<Action>(ingoingActions);
            newState.outgoingActions = new List<Action>(outgoingActions);
            newState.bestAction = bestAction;
            newState.bestValue = bestValue;
            newState.id = this.id;
            newState.setOfReduceRules = new Dictionary<string, HashSet<Formula>>(this.setOfReduceRules);
            newState.classDistribution = new Dictionary<string, double>(classDistribution);
            newState.leafsPaths = new List<KeyValuePair<Path, double>>(leafsPaths);
            newState.instances = instances;
            return newState;
        }

        public void AddOutgoingActions(Action action)
        {
            outgoingActions.Add(action);
        }

        public void AddIngoingActions(Action action)
        {
            ingoingActions.Add(action);
        }
        public double CanGet(State s2,string testName, DataReader.DataSet dataSet)
        {         
            return 0;            
        }

        string toStringValue = string.Empty;
        public override string ToString()
        {
            toStringValue = "";
            foreach (string attName in attributes)
            {
                toStringValue += attName + "  ,";
            }
            if (toStringValue.Length > 3) 
                toStringValue = toStringValue.Remove(toStringValue.Length - 3, 3);
            return "Id: " + id + "|  "+path.ToString() + "    Attribute: "+toStringValue;
        }

        public static Dictionary<HashSet<string>,Dictionary<HashSet<string>,Dictionary<Path,double>>> debugObject = new Dictionary<HashSet<string>, Dictionary<HashSet<string>, Dictionary<Path, double>>>();
        public static HashSet<int> debugTool = new HashSet<int>();
        public static void AddToDebugObject(HashSet<string> set1,HashSet<string> set2, Path path, double prob)
        {
            if (!debugObject.ContainsKey(set1))
                debugObject.Add(set1, new Dictionary<HashSet<string>, Dictionary<Path, double>>());
            if (!debugObject[set1].ContainsKey(set2))
                debugObject[set1].Add(set2, new Dictionary<Path, double>());
            if (!debugObject[set1][set2].ContainsKey(path))
                debugObject[set1][set2].Add(path, 0.0);
            debugObject[set1][set2][path] += prob;
        }


        public bool UpdateValue(DataReader.DataSet dataSet, Dictionary<string, Formula> knownAtts)
        {
            double maxValue = double.MinValue;
            double restTestCost = 0.0;
            this.bestAction = null;
            this.bestValue = double.MinValue;

            Action maxAction = null;

            Dictionary<Action, List<KeyValuePair<Path, double>>> actionToLeafPaths = new Dictionary<Action, List<KeyValuePair<Path, double>>>();
            Dictionary<Action, double> actionToFowardTestsCosts = new Dictionary<Action, double>();

            if (!debugTool.Contains(attributes.Count))
            {
                debugTool.Add(attributes.Count);
                foreach (var item in debugObject)
                {
                    foreach (var item2 in item.Value)
                    {

                        double sum = item2.Value.Sum(i => i.Value);
                        if (sum < 0.999 || sum > 1.0001)
                        {
                            throw new Exception("bugg");
                        }

                    }
                }
            }
            foreach (Action act in outgoingActions)
            {

                double val = 0.0;
                double localRestTestCost = 0.0;
                if (!act.isClassificationAction)
                {
                    CalculateActionValV2(out val, out localRestTestCost, act, dataSet, knownAtts, actionToLeafPaths, actionToFowardTestsCosts);
                }
                else
                {
                    val += act.reward;

                }
                if (val > maxValue)
                {
                    maxValue = val;
                    maxAction = act;
                    restTestCost = localRestTestCost;
                }
                else
                {
                    if (val == maxValue & act.isClassificationAction)
                    {
                        maxValue = val;
                        maxAction = act;
                        restTestCost = localRestTestCost;
                    }
                }
            }

            if (bestValue < maxValue)
            {
                //if (bestAction != null && bestAction.ToString() != maxAction.ToString())
                //     Console.Write("BUG");
                bestValue = maxValue;
                bestAction = maxAction;
                this.restTestCoST = restTestCost;
                if (bestAction.isClassificationAction)
                {
                    leafsPaths = new List<KeyValuePair<Path, double>>();
                    leafsPaths.Add(new KeyValuePair<Path, double>(path, 1));
                }
                else
                {
                    leafsPaths = actionToLeafPaths[bestAction];
                    fowardTestsCost = actionToFowardTestsCosts[bestAction];
                }

                double sum = 0;
                foreach (var leafPath in leafsPaths)
                {
                    sum += leafPath.Value * leafPath.Key.accuracy;
                }


                if (!bestAction.isClassificationAction && leafsPaths.Count == 1)
                {
                    Action classifyAction = null;
                    double targetValue = 0;
                    State targetState = bestAction.targetStates.ElementAt(0).Key;
                    Path lastPath = null;
                    while (classifyAction == null)
                    {
                        if (targetState.bestAction.isClassificationAction)
                        {
                            classifyAction = targetState.bestAction;
                            targetValue = targetState.bestValue;
                            lastPath = targetState.path;
                        }
                        else
                            targetState = targetState.bestAction.targetStates.ElementAt(0).Key;
                    }


                    bestValue = targetValue;
                    bestAction = classifyAction;
                    this.restTestCoST = 0;
                    leafsPaths = new List<KeyValuePair<Path, double>>();
                    leafsPaths.Add(new KeyValuePair<Path, double>(lastPath, 1));

                }


                // test distribution of the final tree

                if (attributes.Count <= 0)
                {
                    Dictionary<Path, double> pathCounter = new Dictionary<Path, double>();
                    Dictionary<Instance, Dictionary<string, Formula>> instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
                    var instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
                    while (instancesIterator.hasMoreElements())
                    {
                        Instance ins = (Instance)instancesIterator.nextElement();
                        Dictionary<string, Formula> attributeValues = Path.GetAttributesValue(ins);
                        instancesAttributeValues.Add(ins, attributeValues);
                    }


                    List<Path> relevantPaths = new List<Path>();
                    List<PathInfo> setOfPaths = new List<PathInfo>();
                    FindPaths(this, setOfPaths, null);

                    foreach (PathInfo pi in setOfPaths.Where(pi => pi.m_Action.isClassificationAction))
                    {
                        PathInfo prev = pi.m_Parent;
                        Path d = new Path();
                        d.accuracy = pi.m_Path.accuracy;
                        foreach (Tree.Node node in pi.m_Path.GetNodeList())
                            d.AddUniqueNode(node);
                        while (prev != null)
                        {
                            foreach (Tree.Node node in prev.m_Path.GetNodeList())
                                d.AddUniqueNode(node);
                            prev = prev.m_Parent;
                        }
                        relevantPaths.Add(d);
                    }

                    int counter = 0;
                    foreach (var item in instancesAttributeValues)
                    {
                        counter++;
                        bool bug = true;
                        foreach (Path path in relevantPaths)
                        {
                            if (path.ConsistWith(item.Value))
                            {
                                if (!pathCounter.ContainsKey(path))
                                {
                                    pathCounter.Add(path, 0.0);
                                }
                                pathCounter[path]++;
                                bug = false;
                                break;
                            }
                        }
                        if (bug)
                            Console.WriteLine("BUG");
                    }
                    sum = pathCounter.Values.Sum(v => v);




                    double accuracy = pathCounter.Sum(pc => pc.Key.accuracy * (pc.Value / ((double)instancesAttributeValues.Count)));
                    if (sum > instancesAttributeValues.Count)
                        throw new Exception("Instance fit more than one path");
                }


                return true;
            }



            return false;
        }

        class ClassToCount
        {
            public double trueCount;
            public double falseCount;
            public ClassToCount()
            {
                trueCount = 0;
                falseCount = 0;
            }

            public ClassToCount(double tc, double fc)
            {
                trueCount = tc;
                falseCount = fc;
            }
        }

        public void CalculateActionValSmallSample(out double val, out double localRestTestCost, Action act, DataReader.DataSet dataSet, Dictionary<string, Formula> knownAtts, 
            Dictionary<Action, List<KeyValuePair<Path, double>>> actionToLeafPaths, Dictionary<Action, double> actionToFowardTestsCosts)
        {
            val = 0;
            List<KeyValuePair<List<Tree.Node>, double>> CureentLeafPath = new List<KeyValuePair<List<Tree.Node>, double>>();
            List<KeyValuePair<Path, double>> passBackwardLeafPath = new List<KeyValuePair<Path, double>>();

            double localFowardsTestsCosts = 0;
            foreach (var optionalTrantision in act.targetStates)
            {
                val += optionalTrantision.Value * optionalTrantision.Key.fowardTestsCost;
                foreach (var leaf in optionalTrantision.Key.leafsPaths)
                {
                    CureentLeafPath.Add(new KeyValuePair<List<Tree.Node>, double>(leaf.Key.GetNodeList().Where(n => !attributes.Contains(n.formula.attributeName)).ToList(), leaf.Value * optionalTrantision.Value));
                    passBackwardLeafPath.Add(new KeyValuePair<Path, double>(leaf.Key, leaf.Value * optionalTrantision.Value));
                }
            }
            localFowardsTestsCosts = val;
            
            double askProbability = CureentLeafPath.Where(p => p.Key.Count > 0).Sum(p => p.Value);
            askProbability = Math.Round(askProbability, 5);
            double askProbabilityv2 = CureentLeafPath.Where(p => p.Key.Count > 0 && p.Key.Select(k => k.formula.attributeName).Contains(act.attributeName)).Sum(p => p.Value);

            if (knownAtts.Keys.Count == attributes.Count && attributes.Union(knownAtts.Keys).Count() == attributes.Count)
            {
                askProbability = 1;
            }
            else
            {
               if (askProbability == 0)
                {
                    askProbability = 1;
                }
            }

            Dictionary<string, Dictionary<string, double>> accuracyTabel = new Dictionary<string, Dictionary<string, double>>();

            foreach (KeyValuePair<Path, double> leafPath in passBackwardLeafPath)
            {
                string classifyVal = leafPath.Key.lastNode.classValue_String;

                foreach (var cd in leafPath.Key.classDistribution)
                {
                    if (!accuracyTabel.ContainsKey(cd.Key))
                    {
                        accuracyTabel.Add(cd.Key, new Dictionary<string, double>());
                    }
                    if (!accuracyTabel[cd.Key].ContainsKey(classifyVal))
                        accuracyTabel[cd.Key].Add(classifyVal, 0.0);
                    accuracyTabel[cd.Key][classifyVal] += cd.Value * leafPath.Value;
                }
            }


            double sumInsInCourentLeaf = classDistribution.Sum(c => c.Value);

            foreach (var classToRealDisribution in accuracyTabel)
            {
                if (classDistribution.ContainsKey(classToRealDisribution.Key))
                {
                    double sumOfInsFromThisClass = classToRealDisribution.Value.Sum(x => x.Value);
                    double propClassInCourentState = (classDistribution[classToRealDisribution.Key] / classDistribution.Sum(cd => cd.Value));
                    foreach (var predictedClass in classToRealDisribution.Value)
                    {
                        val += (propClassInCourentState) * (predictedClass.Value / sumOfInsFromThisClass) * dataSet.m_classificationCostMetrix[predictedClass.Key][classToRealDisribution.Key];
                    }
                }
            }
            localFowardsTestsCosts += askProbability * act.reward;
            val += askProbability * act.reward;
            localRestTestCost = (1.0 - askProbability) * act.reward;
            act.askProbability = askProbability;

            actionToLeafPaths.Add(act, passBackwardLeafPath);
            actionToFowardTestsCosts.Add(act, localFowardsTestsCosts);

        }

        public void CalculateActionVal(out double val, out double localRestTestCost, Action act, DataReader.DataSet dataSet, Dictionary<string, Formula> knownAtts, 
            Dictionary<Action, List<KeyValuePair<Path, double>>> actionToLeafPaths, Dictionary<Action, double> actionToFowardTestsCosts)
        {

            List<KeyValuePair<List<Tree.Node>, double>> CureentLeafPath = new List<KeyValuePair<List<Tree.Node>, double>>();
            List<KeyValuePair<Path, double>> passBackwardLeafPath = new List<KeyValuePair<Path, double>>();

            val = 0;
            double localFowardsTestsCosts = 0;
            foreach (var optionalTrantision in act.targetStates)
            {
                /* if(optionalTrantisio n.Key.outgoingActions.Count <= 1)
                 {
                     optionalTrantision.Key.leafsPaths = new List<KeyValuePair<List<Tree.Node>, double>>();
                     optionalTrantision.Key.leafsPaths.Add(new KeyValuePair<List<Tree.Node>, double>(optionalTrantision.Key.path.GetNodeList(), 1));
                 }*/
                //val += optionalTrantision.Key.bestValue * optionalTrantision.Value;
                val += optionalTrantision.Value * optionalTrantision.Key.fowardTestsCost;
                foreach (var leaf in optionalTrantision.Key.leafsPaths)
                {
                    CureentLeafPath.Add(new KeyValuePair<List<Tree.Node>, double>(leaf.Key.GetNodeList().Where(n => !attributes.Contains(n.formula.attributeName)).ToList(), leaf.Value * optionalTrantision.Value));
                    passBackwardLeafPath.Add(new KeyValuePair<Path, double>(leaf.Key, leaf.Value * optionalTrantision.Value));
                }
                double x = double.Parse(path.lastNode.ratioStr.Split('/')[0]);
                AddToDebugObject(attributes, optionalTrantision.Key.attributes, optionalTrantision.Key.path, (x / (double)dataSet.m_trainInstances1.numInstances()) * optionalTrantision.Value);
            }

            double askProbability = 0;// CureentLeafPath.Where(p => p.Key.Count > 0).Sum(p => p.Value);
            askProbability = 0;
            foreach (var s in act.targetStates)
            {
                if (s.Key.path.GetNodeList().Select(nl => nl.formula.attributeName).Contains(act.attributeName) || !s.Key.bestAction.isClassificationAction)
                {
                    askProbability += s.Value;
                }
            }

            askProbability = Math.Round(askProbability, 5);
           // double askProbabilityv2 = CureentLeafPath.Where(p => p.Key.Count > 0 && p.Key.Select(k => k.formula.attributeName).Contains(act.attributeName)).Sum(p => p.Value);
            //double askProbabilityv3 = GetAskProbability(this, act, dataSet, passBackwardLeafPath);
            if (knownAtts.Keys.Count == attributes.Count && attributes.Union(knownAtts.Keys).Count() == attributes.Count)
            {
                askProbability = 1;
            }
            else
            {
                // askProbability = GetAskProbability(this, act, dataSet, passBackwardLeafPath);
                if (askProbability == 0)
                {
                    askProbability = 1;
                }
            }


            localFowardsTestsCosts = val;

            val = 0;

            double cCost = askProbability * act.reward;
            localRestTestCost = (1.0 - askProbability) * act.reward;
            double fCost = 0.0;
            double counter = 0;
            Dictionary<string, Dictionary<string, double>> accuracyTabel = new Dictionary<string, Dictionary<string, double>>();

            foreach (var ins in instances)
            {
                List<State> nextStates = act.targetStates.Where(d => d.Value > 0).Select(p => p.Key).ToList();
                var nextStatesWithProb = act.targetStates.Where(d => d.Value > 0).ToList();
                Action bestAction = act;
                double insCost = 0;
                bool insIsRelevant = true;
                bool firstStep = true;
                State cs = this;
                while (!bestAction.isClassificationAction)
                {
                    if (firstStep)
                    {
                        insCost += bestAction.reward * askProbability;
                        firstStep = false;
                    }
                    else
                    {
                        double askProb = 0;
                        foreach(var s in nextStatesWithProb)
                        {
                            if(s.Key.path.GetNodeList().Select(nl=>nl.formula.attributeName).Contains(bestAction.attributeName) ||  !s.Key.bestAction.isClassificationAction)
                            {
                                askProb += s.Value;
                            }
                        }
                        double c = askProb * bestAction.reward;
                        double testCost = bestAction.reward - cs.restTestCoST; ;
                      //  if (Math.Abs(c - testCost) > 0.001)
                       //     Console.WriteLine("BUG");

                        insCost += testCost;
                    }
                    bool notFit = true;
                    State cState = null;
                    foreach (State s in nextStates)
                    {
                        if (s.ConsistWith(ins.Value))
                        {
                            nextStates = !s.bestAction.isClassificationAction ? s.bestAction.targetStates.Where(d => d.Value > 0).Select(p => p.Key).ToList() : null;
                            nextStatesWithProb = !s.bestAction.isClassificationAction ? s.bestAction.targetStates.Where(d => d.Value > 0).ToList() : null;
                            bestAction = s.bestAction;
                            notFit = false;
                            cs = s;
                            break;
                        }
                    }
                    if (notFit)
                    {
                        insIsRelevant = false;
                        break;
                    }

                }

                if (insIsRelevant)
                {
                    counter++;
                    int attCount = ins.Key.numAttributes();
                    string trueClass = ins.Key.stringValue(attCount - 1);

                    if (!accuracyTabel.ContainsKey(trueClass))
                        accuracyTabel.Add(trueClass, new Dictionary<string, double>());
                    if (!accuracyTabel[trueClass].ContainsKey(bestAction.classification_string))
                        accuracyTabel[trueClass].Add(bestAction.classification_string, 0);
                    accuracyTabel[trueClass][bestAction.classification_string]++;


                    //val += dataSet.m_classificationCostMetrix[bestAction.classification_string][trueClass];
                    val += insCost;
                }

            }
            if (counter < 0)
            {
                CalculateActionValSmallSample(out val, out localRestTestCost, act, dataSet, knownAtts, actionToLeafPaths, actionToFowardTestsCosts);
                return;
            }
            val = val / counter;

            foreach (var classToRealDisribution in accuracyTabel)
            {
                if (classDistribution.ContainsKey(classToRealDisribution.Key))
                {
                    double sumOfInsFromThisClass = classToRealDisribution.Value.Sum(x => x.Value);
                    double propClassInCourentState = (classDistribution[classToRealDisribution.Key] / classDistribution.Sum(cd => cd.Value));
                    foreach (var predictedClass in classToRealDisribution.Value)
                    {
                        val += (propClassInCourentState) * (predictedClass.Value / sumOfInsFromThisClass) * dataSet.m_classificationCostMetrix[predictedClass.Key][classToRealDisribution.Key];
                        //val +=  (predictedClass.Value / counter) * dataSet.m_classificationCostMetrix[predictedClass.Key][classToRealDisribution.Key];
                       // double v1= (propClassInCourentState) * (predictedClass.Value / sumOfInsFromThisClass) * dataSet.m_classificationCostMetrix[predictedClass.Key][classToRealDisribution.Key];
                      //  double v2 = (predictedClass.Value / counter) * dataSet.m_classificationCostMetrix[predictedClass.Key][classToRealDisribution.Key];

                    }
                }
            }

            
            // val += act.reward;
            actionToLeafPaths.Add(act, passBackwardLeafPath);
            actionToFowardTestsCosts.Add(act, fCost + cCost);
            // if (askProbability == 0)
            //    Console.WriteLine("bug");}
        }

        public void CalculateActionValV2(out double val, out double localRestTestCost, Action act, DataReader.DataSet dataSet, Dictionary<string, Formula> knownAtts, 
            Dictionary<Action, List<KeyValuePair<Path, double>>> actionToLeafPaths, Dictionary<Action, double> actionToFowardTestsCosts)
        {
            List<KeyValuePair<Path, double>> passBackwardLeafPath = new List<KeyValuePair<Path, double>>();
            foreach (var optionalTrantision in act.targetStates)
            {
                foreach (var leaf in optionalTrantision.Key.leafsPaths)
                {
                    passBackwardLeafPath.Add(new KeyValuePair<Path, double>(leaf.Key, leaf.Value * optionalTrantision.Value));
                }
            }

            double askProbability = 0;
            askProbability = 0;
            foreach (var s in act.targetStates)
            {
                if (s.Key.path.GetNodeList().Select(nl => nl.formula.attributeName).Contains(act.attributeName) || !s.Key.bestAction.isClassificationAction)
                {
                    askProbability += s.Value;
                }
            }

            askProbability = Math.Round(askProbability, 5);
            if (knownAtts.Keys.Count == attributes.Count && attributes.Union(knownAtts.Keys).Count() == attributes.Count)
            {
                askProbability = 1;
            }
            else
            {
                if (askProbability == 0)
                {
                    askProbability = 1;
                }
            }


            val = 0;

            double cCost = askProbability * act.reward;
            localRestTestCost = (1.0 - askProbability) * act.reward;
            double counter = 0;
            double fCost = 0.0;
            Dictionary<string, Dictionary<string, double>> accuracyTabel = new Dictionary<string, Dictionary<string, double>>();

            foreach (var ins in instances)
            {
                List<State> nextStates = act.targetStates.Where(d => d.Value > 0).Select(p => p.Key).ToList();
                var nextStatesWithProb = act.targetStates.Where(d => d.Value > 0).ToList();
                Action bestAction = act;
                double insCost = 0;
                bool insIsRelevant = true;
                bool firstStep = true;
                State cs = this;
                while (!bestAction.isClassificationAction)
                {
                    if (firstStep)
                    {
                        insCost += bestAction.reward * askProbability;
                        firstStep = false;
                    }
                    else
                    {
                        double askProb = 0;
                        foreach (var s in nextStatesWithProb)
                        {
                            if (s.Key.path.GetNodeList().Select(nl => nl.formula.attributeName).Contains(bestAction.attributeName) || !s.Key.bestAction.isClassificationAction)
                            {
                                askProb += s.Value;
                            }
                        }
                        double c = askProb * bestAction.reward;
                        double testCost = bestAction.reward - cs.restTestCoST; ;
                        insCost += testCost;
                    }
                    bool notFit = true;
                    foreach (State s in nextStates)
                    {
                        if (s.ConsistWith(ins.Value))
                        {
                            nextStates = !s.bestAction.isClassificationAction ? s.bestAction.targetStates.Where(d => d.Value > 0).Select(p => p.Key).ToList() : null;
                            nextStatesWithProb = !s.bestAction.isClassificationAction ? s.bestAction.targetStates.Where(d => d.Value > 0).ToList() : null;
                            bestAction = s.bestAction;
                            notFit = false;
                            cs = s;
                            break;
                        }
                    }
                    if (notFit)
                    {
                        insIsRelevant = false;
                        break;
                    }

                }

                if (insIsRelevant)
                {
                    counter++;
                    int attCount = ins.Key.numAttributes();
                    string trueClass = ins.Key.stringValue(attCount - 1);

                    if (!accuracyTabel.ContainsKey(trueClass))
                        accuracyTabel.Add(trueClass, new Dictionary<string, double>());
                    if (!accuracyTabel[trueClass].ContainsKey(bestAction.classification_string))
                        accuracyTabel[trueClass].Add(bestAction.classification_string, 0);
                    accuracyTabel[trueClass][bestAction.classification_string]++;

                    val += insCost;
                }

            }
            if (counter < 0)
            {
                CalculateActionValSmallSample(out val, out localRestTestCost, act, dataSet, knownAtts, actionToLeafPaths, actionToFowardTestsCosts);
                return;
            }
            val = val / counter;

            foreach (var classToRealDisribution in accuracyTabel)
            {
                if (classDistribution.ContainsKey(classToRealDisribution.Key))
                {
                    double sumOfInsFromThisClass = classToRealDisribution.Value.Sum(x => x.Value);
                    double propClassInCourentState = (classDistribution[classToRealDisribution.Key] / classDistribution.Sum(cd => cd.Value));
                    foreach (var predictedClass in classToRealDisribution.Value)
                    {
                        val += (propClassInCourentState) * (predictedClass.Value / sumOfInsFromThisClass) * dataSet.m_classificationCostMetrix[predictedClass.Key][classToRealDisribution.Key];
                    }
                }
            }
            actionToLeafPaths.Add(act, passBackwardLeafPath);
            actionToFowardTestsCosts.Add(act, fCost + cCost);
        }

        public double GetAskProbability(State s,Action tryAction,DataReader.DataSet dataSet, List<KeyValuePair<Path, double>> passBackwardLeafPath)
        {
            Action tmp = s.bestAction;
            s.bestAction = tryAction;
            List<Path> relevantPaths = new List<Path>();
            List<PathInfo> setOfPaths = new List<PathInfo>();
            FindPaths(s, setOfPaths, null);

            foreach (PathInfo pi in setOfPaths.Where(pi => pi.m_Action.isClassificationAction))
            {
                foreach(var leafsPath in passBackwardLeafPath)
                {
                    if (Contains(leafsPath.Key.GetNodeList(), pi.m_Path.GetNodeList()))
                        pi.m_Depth = leafsPath.Value;
                }

                PathInfo prev = pi.m_Parent;
                Path d = new Path();
                d.accuracy = pi.m_Path.accuracy;
                d.accuracy = pi.m_Depth;
                foreach (Tree.Node node in pi.m_Path.GetNodeList())
                    d.AddUniqueNode(node);
                while (prev != null)
                {
                    foreach (Tree.Node node in prev.m_Path.GetNodeList())
                        d.AddUniqueNode(node);
                    prev = prev.m_Parent;
                }
                relevantPaths.Add(d);
            }

           /* Dictionary<Path, double> pathCounter = new Dictionary<Path, double>();
            Dictionary<Instance, Dictionary<string, Formula>> instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
            var instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues = Path.GetAttributesValue(ins);
                //if(ConsistWith(attributeValues))
                    instancesAttributeValues.Add(ins, attributeValues);
            }

            int counter = 0;
            foreach (var item in instancesAttributeValues)
            {
                
                bool bug = true;
                foreach (Path path in relevantPaths)
                {
                    if (path.ConsistWith(item.Value))
                    {
                        if (!pathCounter.ContainsKey(path))
                        {
                            pathCounter.Add(path, 0.0);
                        }
                        pathCounter[path]++;
                        counter++;
                        bug = false;
                        break;
                    }
                }
            }*/

            double askProbabilityv2 = relevantPaths.Where(p => p.GetNodeList().Count > 0 && p.GetNodeList().Select(k => k.formula.attributeName).Contains(tryAction.attributeName)).Sum(p => ((double)p.accuracy));

            askProbabilityv2 = Math.Round(askProbabilityv2, 5);
            return askProbabilityv2;
        }

        public bool Contains(List<Tree.Node> l1, List<Tree.Node> l2)
        {
            if (l1 == null || l2 == null)
                return false;
            if (l1.Count != l2.Count)
                return false;
            foreach(Tree.Node n1 in l1)
            {
                if (!l2.Contains(n1))
                    return false;
            }
            return true;
        }


        public static void FindPaths(State s, List<PathInfo> setOfPaths, PathInfo parent)
        {
            PathInfo pathInfo = new PathInfo(s.path, s.bestAction, parent);
            setOfPaths.Add(pathInfo);
            if (!s.bestAction.isClassificationAction)
            {
                foreach (State ts in s.bestAction.targetStates.Keys)
                {
                    FindPaths(ts, setOfPaths, pathInfo);
                }
            }
        }

        public class PathInfo
        {
            public Path m_Path;
            public Action m_Action;
            public PathInfo m_Parent;
            public double m_Depth = 0;
            public PathInfo(Path path, Action action, PathInfo parent)
            {
                m_Path = path;
                m_Action = action;
                m_Parent = parent;
            }
            public PathInfo(Path path, Action action, PathInfo parent,int depth)
            {
                m_Path = path;
                m_Action = action;
                m_Parent = parent;
                m_Depth = depth;
            }

            public override string ToString()
            {
                return m_Path.ToString() +"     Action: "+m_Action.ToString() ;
            }
        }


        public bool ConsistWith(Dictionary<string, Formula> attributeValues)
        {
            foreach (Tree.Node node in nodes)
            {
                if (!attributeValues.ContainsKey(node.formula.attributeName) || !attributeValues[node.formula.attributeName].IsWeakStronger(node.formula))
                    return false;
            }
            return true;
        }

        public HashSet<Formula> PartialConsistWith(Dictionary<string, Formula> attributeValues)
        {
            HashSet<Formula> falserRules = new HashSet<Formula>();
            foreach (Tree.Node node in nodes)
            {
                if (attributeValues.ContainsKey(node.formula.attributeName))
                { 
                    if (!attributeValues[node.formula.attributeName].IsWeakStronger(node.formula))
                    {
                        falserRules.Add(node.formula);
                    }
                }
            }
            return falserRules;
        }


        public void SetRelevantInstances(DataReader.DataSet dataSet)
        {
            Dictionary<Instance, Dictionary<string, Formula>> instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
            var instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues = GetInstanceValue(ins);
                instancesAttributeValues.Add(ins, attributeValues);
            }

            foreach (var item in instancesAttributeValues)
            {
                bool bug = true;
                if (path.ConsistWith(item.Value))
                {
                    instances.Add(item.Key, item.Value);

                }

            }
        }

        public void SetRelevantInstancesAndClassDistribution(DataReader.DataSet dataSet)
        {

            var instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                int attCount = dataSet.m_testInstances.numAttributes();

                Dictionary<string, Formula> attributeValues = GetInstanceValue(ins);

                if (path.ConsistWith(attributeValues))
                {
                    instances.Add(ins, attributeValues);
                    string trueClass = ins.stringValue(attCount - 1);
                    if (!path.classDistribution.ContainsKey(trueClass))
                        path.classDistribution.Add(trueClass, 0);
                    path.classDistribution[trueClass]++;
                }
            }

 


            double sum = path.classDistribution.Sum(cd => cd.Value);
            var classDistributionorgCount = path.classDistribution;
            path.classDistribution = path.classDistribution.ToDictionary(c => c.Key, c => (double)c.Value / sum);
            sum = path.classDistribution.Sum(cd => cd.Value);
            var choosedClass = DataReader.ChooseClass(dataSet, path.classDistribution);
            double newAccuracy = path.classDistribution[choosedClass] / sum;
            if (newAccuracy != path.accuracy)
                path.accuracy = newAccuracy;
            path.lastNode.classValue_Int = int.Parse(choosedClass);
            path.lastNode.classValue_String = choosedClass;
        }

        public static Dictionary<Instance, Dictionary<string, Formula>> InstanceValues = new Dictionary<Instance, Dictionary<string, Formula>>();
        public static Dictionary<string, Formula> GetInstanceValue(Instance ins)
        {
            if (!InstanceValues.ContainsKey(ins))
            {
                Dictionary<string, Formula> attributesValue = new Dictionary<string, Formula>();
                for (int i = 0; i < ins.numValues(); i++)
                {
                    Formula f = new Formula();
                    var att = ins.attribute(i);
                    f.attributeName = att.name();
                    f.symbol = "=";
                    var val = ins.value(i);
                    f.attributeValue_double = val;
                    f.attributeValue_string = "";
                    attributesValue.Add(f.attributeName, f);
                }
                InstanceValues[ins] = attributesValue;
            }
            return InstanceValues[ins];
        }


    }
}
