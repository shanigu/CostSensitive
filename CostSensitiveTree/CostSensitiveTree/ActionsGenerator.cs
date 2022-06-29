using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using weka.core;

namespace CostSensitiveTree
{
    class ActionsGenerator
    {
        public static List<Action> AddClassificationActions(List<State> states,DataReader.DataSet dataSet)
        {
            System.Console.WriteLine();
            System.Console.WriteLine("Find Classification Actions:");
            int counter = 1;
            List<Action> allClassificationActions = new List<Action>();
            for (int i = 0; i < states.Count; i++)
            {
                State state = states[i];
                Action action = new Action(state, state.path.lastNode.classValue_Int, state.path.lastNode.classValue_String,
                    DataReader.CalculateMisClassificationValue(dataSet, state.path.classDistribution, state.path.lastNode.classValue_String));
                state.outgoingActions.Add(action);
                allClassificationActions.Add(action);
                System.Console.Write("\r{0}%   ", Math.Round(100 * (double)counter++ / (double)states.Count, 3));
            }
            return allClassificationActions;
        }


        public static List<Action> AddQueryAction(List<Tree> trees, Dictionary<Path, State> pathToStateMapper,DataReader.DataSet dataSet)
        {
            System.Console.WriteLine();
            System.Console.WriteLine("Find Query Actions:");
            int counter = 1;
            List<Action> queryActions = new List<Action>();
            foreach (Tree tree in trees)
            {
                List<Tree> relevantTrees = trees.Where(x => tree.attributesNames.Count + 1 == x.attributesNames.Count &&
                                            tree.attributesNames.Intersect(x.attributesNames).Count() == tree.attributesNames.Count).ToList();

                foreach (Path path in tree.reducedPaths)
                {
                    foreach (Tree relevantTree in relevantTrees)
                    {
                        string attName = relevantTree.attributesNames.Except(tree.attributesNames).ElementAt(0);

                        var trantisions = path.GetTransitionV3(dataSet, relevantTree.reducedPaths.ToList());
                    
                        double cost = dataSet.m_features[attName].cost;
                        if (dataSet.groupTests.Intersect(tree.attributesNames).Count()>0)
                        {
                            cost = dataSet.m_features[attName].GroupedCost;
                        }


                        Action queryAction = new Action(pathToStateMapper, path, trantisions, cost, attName);

                        double x = trantisions.Sum(t => t.Value);
                        if (x < 0.99 || 1.01 < x)
                            throw new Exception("need to be 1");

                        pathToStateMapper[path].AddOutgoingActions(queryAction);
                        foreach (Path p2 in trantisions.Keys)
                        {
                            pathToStateMapper[p2].AddIngoingActions(queryAction);
                        }
                        queryActions.Add(queryAction);
                    }
                }
                System.Console.Write("\r{0}%   ", Math.Round(100 * (double)counter++ / (double)trees.Count, 3));
            }
            return queryActions;
        }


        public static void UpdateTransitionOutGoingAction(Action action, DataReader.DataSet dataSet)
        {

            Dictionary<State, double> stateCounter = new Dictionary<State, double>();
            Dictionary<Instance, Dictionary<string, Formula>> instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
            Dictionary<double, int> classCounter = new Dictionary<double, int>();
            var instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues = Path.GetAttributesValue(ins);
                if (isConsistWithReducedSet(attributeValues, action.sourceState.setOfReduceRules) && action.sourceState.ConsistWith(attributeValues))
                {
                    instancesAttributeValues.Add(ins, attributeValues);
                    int attCount = dataSet.m_testInstances.numAttributes();
                    double trueClass = double.Parse(ins.stringValue(attCount - 1));
                    if (!classCounter.ContainsKey(trueClass))
                        classCounter.Add(trueClass, 0);
                    classCounter[trueClass]++;
                }
            }

            foreach (var item in instancesAttributeValues)
            {
                bool bug = true;
                foreach (State state in action.targetStates.Keys)
                {
                    if (state.ConsistWith(item.Value))
                    {
                        if (!stateCounter.ContainsKey(state))
                        {
                            stateCounter.Add(state, 0.0);
                        }
                        stateCounter[state]++;
                        bug = false;
                       // break;
                    }
                }
                if (bug)
                    throw new Exception("Instance not fit to any path");
            }
            double sum = stateCounter.Values.Sum(v => v);
            if (sum != instancesAttributeValues.Count)
                throw new Exception("Instance fit more than one path");
           
            foreach (var item in stateCounter)
            {
                action.targetStates[item.Key] = item.Value / (double)sum;
            }

        }

        public static int debugCounter = 0;
        public static bool UpdateTransitionBySourceState(State sourceState, DataReader.DataSet dataSet,Dictionary<string,HashSet<Formula>> globalReduceRules)
        {
            debugCounter++;
            Dictionary<Instance, Dictionary<string, Formula>> instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
            Dictionary<string, double> classCounter = new Dictionary<string, double>();
            var instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues = Path.GetAttributesValue(ins);

                /* if (isConsistWithReducedSet(attributeValues, globalReduceRules))
                 {
                     continue;
                 }*/


                //    if (sourceState.ConsistWith(attributeValues))
                if (isConsistWithReducedSet(attributeValues, sourceState.setOfReduceRules) && sourceState.ConsistWith(attributeValues))
                {
                    instancesAttributeValues.Add(ins, attributeValues);
                    int attCount = dataSet.m_testInstances.numAttributes();
                    string trueClass = ins.stringValue(attCount - 1);
                    if (!classCounter.ContainsKey(trueClass))
                        classCounter.Add(trueClass, 0);
                    classCounter[trueClass]++;
                }
            }
            if (instancesAttributeValues.Count==0)
            {
                return false;
                Console.WriteLine(ModelEvaluation.counter);
                throw new Exception("no instances- why??");
            }
            foreach (Action outAct in sourceState.outgoingActions)
            {
                if (!outAct.isClassificationAction)
                {
                    Dictionary<State, double> stateCounter = new Dictionary<State, double>();
                    foreach (var item in instancesAttributeValues)
                    {
                        bool bug = true;
                        foreach (State state in outAct.targetStates.Keys)
                        {
                            if (state.ConsistWith(item.Value))
                            {
                                if (!stateCounter.ContainsKey(state))
                                {
                                    stateCounter.Add(state, 0.0);
                                }
                                stateCounter[state]++;
                                bug = false;
                                // break;
                            }
                        }
                      //  if (bug)
                      //      throw new Exception("Instance not fit to any path");
                    }

                    double sum = stateCounter.Values.Sum(v => v);
                   // if (sum != instancesAttributeValues.Count)
                   //     throw new Exception("Instance fit more than one path");

                    foreach (var item in stateCounter)
                    {
                        outAct.targetStates[item.Key] = item.Value / (double)sum;
                    }
                }
            }

            var bestClass = classCounter.OrderByDescending(i => i.Value).ElementAt(0);
            double totalIns = classCounter.Sum(cc => cc.Value);
            double accuracy = (double)bestClass.Value / (double)totalIns;

            Action classifyAction = sourceState.outgoingActions.FirstOrDefault(act => act.isClassificationAction);

            var classDistribution = classCounter.ToDictionary(c => c.Key, c => (double)c.Value / totalIns);
            var choosedClass = DataReader.ChooseClass(dataSet, classDistribution);
            classifyAction.classification_int = int.Parse(choosedClass);
            classifyAction.classification_string = choosedClass;
            classifyAction.reward = DataReader.CalculateMisClassificationValue(dataSet, classDistribution, choosedClass);

            sourceState.classDistribution = classDistribution;
            return true;
            /*classifyAction.classification_int = int.Parse(bestClass.Key);
            classifyAction.classification_string = bestClass.Key.ToString();
            classifyAction.reward = DataReader.CalculateSimpaleValue(accuracy);*/
        }



        // reduce by action
        public static bool UpdateTransitionBySourceStateII(State sourceState, DataReader.DataSet dataSet, Dictionary<string, HashSet<Formula>> globalReduceRules)
        {
            debugCounter++;
            Dictionary<Instance, Dictionary<string, Formula>> instancesAttributeValues = null;
            Dictionary<string, double> classCounter = null;
            java.util.Enumeration instancesIterator = null;
            foreach (Action outAct in sourceState.outgoingActions)
            {
                instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
                classCounter = new Dictionary<string, double>();
                instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues = Path.GetAttributesValue(ins);

                if (sourceState.ConsistWith(attributeValues) && isConsistWithReducedSet(attributeValues, outAct.setOfReduceRules))
                {
                    instancesAttributeValues.Add(ins, attributeValues);
                    int attCount = dataSet.m_testInstances.numAttributes();
                    string trueClass = ins.stringValue(attCount - 1);
                    if (!classCounter.ContainsKey(trueClass))
                        classCounter.Add(trueClass, 0);
                    classCounter[trueClass]++;
                }
            }
            if (instancesAttributeValues.Count == 0)
            {
                return false;
                Console.WriteLine(ModelEvaluation.counter);
                throw new Exception("no instances- why??");
            }
            
                if (!outAct.isClassificationAction)
                {
                    Dictionary<State, double> stateCounter = new Dictionary<State, double>();
                    foreach (var item in instancesAttributeValues)
                    {
                        bool bug = true;
                        foreach (State state in outAct.targetStates.Keys)
                        {
                            if (state.ConsistWith(item.Value))
                            {
                                if (!stateCounter.ContainsKey(state))
                                {
                                    stateCounter.Add(state, 0.0);
                                }
                                stateCounter[state]++;
                                bug = false;
                                // break;
                            }
                        }
                        //  if (bug)
                        //      throw new Exception("Instance not fit to any path");
                    }

                    double sum = stateCounter.Values.Sum(v => v);
                    // if (sum != instancesAttributeValues.Count)
                    //     throw new Exception("Instance fit more than one path");

                    foreach (var item in stateCounter)
                    {
                        outAct.targetStates[item.Key] = item.Value / (double)sum;
                    }
                }
            }

            instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
            classCounter = new Dictionary<string, double>();
            instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues = Path.GetAttributesValue(ins);
                if (sourceState.ConsistWith(attributeValues))
                {
                    instancesAttributeValues.Add(ins, attributeValues);
                    int attCount = dataSet.m_testInstances.numAttributes();
                    string trueClass = ins.stringValue(attCount - 1);
                    if (!classCounter.ContainsKey(trueClass))
                        classCounter.Add(trueClass, 0);
                    classCounter[trueClass]++;
                }
            }
            if (instancesAttributeValues.Count == 0)
            {
                return false;
                Console.WriteLine(ModelEvaluation.counter);
                throw new Exception("no instances- why??");
            }

            var bestClass = classCounter.OrderByDescending(i => i.Value).ElementAt(0);
            double totalIns = classCounter.Sum(cc => cc.Value);
            double accuracy = (double)bestClass.Value / (double)totalIns;

            Action classifyAction = sourceState.outgoingActions.FirstOrDefault(act => act.isClassificationAction);

            var classDistribution = classCounter.ToDictionary(c => c.Key, c => (double)c.Value / totalIns);
            var choosedClass = DataReader.ChooseClass(dataSet, classDistribution);
            classifyAction.classification_int = int.Parse(choosedClass);
            classifyAction.classification_string = choosedClass;
            classifyAction.reward = DataReader.CalculateMisClassificationValue(dataSet, classDistribution, choosedClass);

            sourceState.classDistribution = classDistribution;
            return true;
            /*classifyAction.classification_int = int.Parse(bestClass.Key);
            classifyAction.classification_string = bestClass.Key.ToString();
            classifyAction.reward = DataReader.CalculateSimpaleValue(accuracy);*/
        }

        public static bool UpdateTransitionBySourceStateV1(State sourceState, DataReader.DataSet dataSet, Dictionary<string, HashSet<Formula>> globalReduceRules)
        {
            debugCounter++;
            Dictionary<Instance, Dictionary<string, Formula>> instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
            Dictionary<string, double> classCounter = new Dictionary<string, double>();
            var instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues = Path.GetAttributesValue(ins);
                instancesAttributeValues.Add(ins, attributeValues);
                int attCount = dataSet.m_testInstances.numAttributes();
                string trueClass = ins.stringValue(attCount - 1);
                if (!classCounter.ContainsKey(trueClass))
                {
                    classCounter.Add(trueClass, 0);
                }
                classCounter[trueClass]++;
            }
            if (instancesAttributeValues.Count == 0)
            {
                return false;
                Console.WriteLine(ModelEvaluation.counter);
                throw new Exception("no instances- why??");
            }
            foreach (Action outAct in sourceState.outgoingActions)
            {
                if (!outAct.isClassificationAction)
                {
                    Dictionary<State, double> stateCounter = new Dictionary<State, double>();
                    foreach (var item in instancesAttributeValues)
                    {
                        bool bug = true;
                        foreach (State state in outAct.targetStates.Keys)
                        {
                            if (state.ConsistWith(item.Value))
                            {
                                if (!stateCounter.ContainsKey(state))
                                {
                                    stateCounter.Add(state, 0.0);
                                }
                                stateCounter[state]++;
                                bug = false;
                            }
                        }

                    }

                    double sum = stateCounter.Values.Sum(v => v);


                    foreach (var item in stateCounter)
                    {
                        outAct.targetStates[item.Key] = item.Value / (double)sum;
                    }
                }
            }

            var bestClass = classCounter.OrderByDescending(i => i.Value).ElementAt(0);
            double totalIns = classCounter.Sum(cc => cc.Value);
            double accuracy = (double)bestClass.Value / (double)totalIns;

            Action classifyAction = sourceState.outgoingActions.FirstOrDefault(act => act.isClassificationAction);

            var classDistribution = classCounter.ToDictionary(c => c.Key, c => (double)c.Value / totalIns);
            var choosedClass = DataReader.ChooseClass(dataSet, classDistribution);
            classifyAction.classification_int = int.Parse(choosedClass);
            classifyAction.classification_string = choosedClass;
            classifyAction.reward = DataReader.CalculateMisClassificationValue(dataSet, classDistribution, choosedClass);

            sourceState.classDistribution = classDistribution;
            return true;
        }

        public static bool UpdateTransitionBySourceStateV2Org(State sourceState, DataReader.DataSet dataSet, Dictionary<string, HashSet<Formula>> globalReduceRules)
        {
            debugCounter++;
            Dictionary<Instance, Dictionary<string, Formula>> instancesAttributeValues = null;
            Dictionary<string, double> classCounter = null;
            java.util.Enumeration instancesIterator = null;
            foreach (Action outAct in sourceState.outgoingActions)
            {
                if (!outAct.isClassificationAction)
                {
                    instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
                    classCounter = new Dictionary<string, double>();
                    instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
                    while (instancesIterator.hasMoreElements())
                    {
                        Instance ins = (Instance)instancesIterator.nextElement();
                        Dictionary<string, Formula> attributeValues = Path.GetAttributesValue(ins);

                        if (isConsistWithReducedSet(attributeValues, outAct.setOfReduceRules))
                        {
                            instancesAttributeValues.Add(ins, attributeValues);
                            int attCount = dataSet.m_testInstances.numAttributes();
                            string trueClass = ins.stringValue(attCount - 1);
                            if (!classCounter.ContainsKey(trueClass))
                                classCounter.Add(trueClass, 0);
                            classCounter[trueClass]++;
                        }
                    }
                    if (instancesAttributeValues.Count == 0)
                    {
                        return false;
                        Console.WriteLine(ModelEvaluation.counter);
                        throw new Exception("no instances- why??");
                    }


                    Dictionary<State, double> stateCounter = new Dictionary<State, double>();
                    foreach (var item in instancesAttributeValues)
                    {
                        bool bug = true;
                        foreach (State state in outAct.targetStates.Keys)
                        {
                            if (state.ConsistWith(item.Value))
                            {
                                if (!stateCounter.ContainsKey(state))
                                {
                                    stateCounter.Add(state, 0.0);
                                }
                                stateCounter[state]++;
                                bug = false;
                                // break;
                            }
                        }
                        //  if (bug)
                        //      throw new Exception("Instance not fit to any path");
                    }

                    double sum = stateCounter.Values.Sum(v => v);
                    // if (sum != instancesAttributeValues.Count)
                    //     throw new Exception("Instance fit more than one path");
                    for (int sIndex = 0; sIndex < outAct.targetStates.Keys.Count; sIndex++)
                    {
                        outAct.targetStates[outAct.targetStates.ElementAt(sIndex).Key] = 0.0;
                    }

                    foreach (var item in stateCounter)
                    {
                        outAct.targetStates[item.Key] = item.Value / (double)sum;
                    }

                    foreach(var s in outAct.targetStates.ToList())
                    {
                        if(s.Value<0.0001)
                            outAct.targetStates.Remove(s.Key);
                    }

                }
            }


            instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
            classCounter = new Dictionary<string, double>();
            instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues = Path.GetAttributesValue(ins);
                if (sourceState.ConsistWith(attributeValues) )//&& isConsistWithReducedSet(attributeValues, globalReduceRules))
                {
                    instancesAttributeValues.Add(ins, attributeValues);
                    int attCount = dataSet.m_testInstances.numAttributes();
                    string trueClass = ins.stringValue(attCount - 1);
                    if (!classCounter.ContainsKey(trueClass))
                        classCounter.Add(trueClass, 0);
                    classCounter[trueClass]++;
                }
            }
            if (instancesAttributeValues.Count == 0)
            {
                return false;
                Console.WriteLine(ModelEvaluation.counter);
                throw new Exception("no instances- why??");
            }

            var bestClass = classCounter.OrderByDescending(i => i.Value).ElementAt(0);
            double totalIns = classCounter.Sum(cc => cc.Value);
            double accuracy = (double)bestClass.Value / (double)totalIns;

            Action classifyAction = sourceState.outgoingActions.FirstOrDefault(act => act.isClassificationAction);

            var classDistribution = classCounter.ToDictionary(c => c.Key, c => (double)c.Value / totalIns);
            var choosedClass = DataReader.ChooseClass(dataSet, classDistribution);
            classifyAction.classification_int = int.Parse(choosedClass);
            classifyAction.classification_string = choosedClass;
            classifyAction.reward = DataReader.CalculateMisClassificationValue(dataSet, classDistribution, choosedClass);

            sourceState.classDistribution = classDistribution;
            return true;
            /*classifyAction.classification_int = int.Parse(bestClass.Key);
            classifyAction.classification_string = bestClass.Key.ToString();
            classifyAction.reward = DataReader.CalculateSimpaleValue(accuracy);*/
        }
        // reduce by action 
        // +
        // dont delete ins that not fit the source path (allready reduced Opposite path in the action generation step)
        public static bool UpdateTransitionBySourceStateV2(State sourceState, DataReader.DataSet dataSet, Dictionary<string, HashSet<Formula>> globalReduceRules)
        {
            debugCounter++;
            Dictionary<Instance, Dictionary<string, Formula>> instancesAttributeValues = null;
            Dictionary<string, double> classCounter = null;
            java.util.Enumeration instancesIterator = null;
            foreach (Action outAct in sourceState.outgoingActions)
            {
                if (!outAct.isClassificationAction)
                {
                    /* instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
                     classCounter = new Dictionary<string, double>();
                     instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
                     while (instancesIterator.hasMoreElements())
                     {
                         Instance ins = (Instance)instancesIterator.nextElement();
                         Dictionary<string, Formula> attributeValues = Path.GetAttributesValue(ins);

                        // if (isConsistWithReducedSet(attributeValues, outAct.setOfReduceRules))
                         {
                             instancesAttributeValues.Add(ins, attributeValues);
                             int attCount = dataSet.m_testInstances.numAttributes();
                             string trueClass = ins.stringValue(attCount - 1);
                             if (!classCounter.ContainsKey(trueClass))
                                 classCounter.Add(trueClass, 0);
                             classCounter[trueClass]++;
                         }
                     }
                     if (instancesAttributeValues.Count == 0)
                     {
                         return false;
                         Console.WriteLine(ModelEvaluation.counter);
                         throw new Exception("no instances- why??");
                     }


                     Dictionary<State, double> stateCounter = new Dictionary<State, double>();
                     foreach (var item in instancesAttributeValues)
                     {
                         bool bug = true;
                         foreach (State state in outAct.targetStates.Keys)
                         {
                             if (state.ConsistWith(item.Value))
                             {
                                 if (!stateCounter.ContainsKey(state))
                                 {
                                     stateCounter.Add(state, 0.0);
                                 }
                                 stateCounter[state]++;
                                 bug = false;
                                 // break;
                             }
                         }
                         //  if (bug)
                         //      throw new Exception("Instance not fit to any path");
                     }

                     double sum = stateCounter.Values.Sum(v => v);
                     // if (sum != instancesAttributeValues.Count)
                     //     throw new Exception("Instance fit more than one path");
                     for(int sIndex=0;sIndex<outAct.targetStates.Keys.Count; sIndex++)
                     {
                         outAct.targetStates[outAct.targetStates.ElementAt(sIndex).Key] = 0.0;
                     }*/

                    double sum = outAct.targetStates.Values.Sum();

                    foreach (var s in outAct.targetStates.Keys.ToList())
                    {
                        outAct.targetStates[s] = outAct.targetStates[s] / (double)sum;
                    }
                }
            }

            /*instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
            classCounter = new Dictionary<string, double>();
            instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues = Path.GetAttributesValue(ins);
                if (sourceState.ConsistWith(attributeValues))
                {
                    instancesAttributeValues.Add(ins, attributeValues);
                    int attCount = dataSet.m_testInstances.numAttributes();
                    string trueClass = ins.stringValue(attCount - 1);
                    if (!classCounter.ContainsKey(trueClass))
                        classCounter.Add(trueClass, 0);
                    classCounter[trueClass]++;
                }
            }
            if (instancesAttributeValues.Count == 0)
            {
                return false;
                Console.WriteLine(ModelEvaluation.counter);
                throw new Exception("no instances- why??");
            }

            var bestClass = classCounter.OrderByDescending(i => i.Value).ElementAt(0);
            double totalIns = classCounter.Sum(cc => cc.Value);
            double accuracy = (double)bestClass.Value / (double)totalIns;

            Action classifyAction = sourceState.outgoingActions.FirstOrDefault(act => act.isClassificationAction);

            var classDistribution = classCounter.ToDictionary(c => c.Key, c => (double)c.Value / totalIns);
            var choosedClass = DataReader.ChooseClass(dataSet, classDistribution);
            classifyAction.classification_int = int.Parse(choosedClass);
            classifyAction.classification_string = choosedClass;
            classifyAction.reward = DataReader.CalculateMisClassificationValue(dataSet, classDistribution, choosedClass);

            sourceState.classDistribution = classDistribution;
            */
            return true;
            /*classifyAction.classification_int = int.Parse(bestClass.Key);
            classifyAction.classification_string = bestClass.Key.ToString();
            classifyAction.reward = DataReader.CalculateSimpaleValue(accuracy);*/
        }

        public static bool UpdateTransitionBySourceStateIII(State sourceState, DataReader.DataSet dataSet, Dictionary<string, HashSet<Formula>> globalReduceRules)
        {
            debugCounter++;
            Dictionary<Instance, Dictionary<string, Formula>> instancesAttributeValues = null;
            Dictionary<string, double> classCounter = null;
            java.util.Enumeration instancesIterator = null;
            foreach (Action outAct in sourceState.outgoingActions)
            {
                instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
                classCounter = new Dictionary<string, double>();
                instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
                while (instancesIterator.hasMoreElements())
                {
                    Instance ins = (Instance)instancesIterator.nextElement();
                    Dictionary<string, Formula> attributeValues = Path.GetAttributesValue(ins);

                    if (sourceState.ConsistWith(attributeValues) && isConsistWithReducedSet(attributeValues, outAct.setOfReduceRules))
                    {
                        instancesAttributeValues.Add(ins, attributeValues);
                        int attCount = dataSet.m_testInstances.numAttributes();
                        string trueClass = ins.stringValue(attCount - 1);
                        if (!classCounter.ContainsKey(trueClass))
                            classCounter.Add(trueClass, 0);
                        classCounter[trueClass]++;
                    }
                }
                if (instancesAttributeValues.Count == 0)
                {
                    return false;
                    Console.WriteLine(ModelEvaluation.counter);
                    throw new Exception("no instances- why??");
                }

                if (!outAct.isClassificationAction)
                {
                    Dictionary<State, double> stateCounter = new Dictionary<State, double>();
                    foreach (var item in instancesAttributeValues)
                    {
                        bool bug = true;
                        foreach (State state in outAct.targetStates.Keys)
                        {
                            if (state.ConsistWith(item.Value))
                            {
                                if (!stateCounter.ContainsKey(state))
                                {
                                    stateCounter.Add(state, 0.0);
                                }
                                stateCounter[state]++;
                                bug = false;
                                // break;
                            }
                        }
                        //  if (bug)
                        //      throw new Exception("Instance not fit to any path");
                    }

                    double sum = stateCounter.Values.Sum(v => v);
                    // if (sum != instancesAttributeValues.Count)
                    //     throw new Exception("Instance fit more than one path");

                    foreach (var item in stateCounter)
                    {
                        outAct.targetStates[item.Key] = item.Value / (double)sum;
                    }
                }
            }

            instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
            classCounter = new Dictionary<string, double>();
            instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues = Path.GetAttributesValue(ins);
                if (sourceState.ConsistWith(attributeValues) && isConsistWithReducedSet(attributeValues,globalReduceRules))
                {
                    instancesAttributeValues.Add(ins, attributeValues);
                    int attCount = dataSet.m_testInstances.numAttributes();
                    string trueClass = ins.stringValue(attCount - 1);
                    if (!classCounter.ContainsKey(trueClass))
                        classCounter.Add(trueClass, 0);
                    classCounter[trueClass]++;
                }
            }
            if (instancesAttributeValues.Count == 0)
            {
                return false;
                Console.WriteLine(ModelEvaluation.counter);
                throw new Exception("no instances- why??");
            }

            var bestClass = classCounter.OrderByDescending(i => i.Value).ElementAt(0);
            double totalIns = classCounter.Sum(cc => cc.Value);
            double accuracy = (double)bestClass.Value / (double)totalIns;

            Action classifyAction = sourceState.outgoingActions.FirstOrDefault(act => act.isClassificationAction);

            var classDistribution = classCounter.ToDictionary(c => c.Key, c => (double)c.Value / totalIns);
            var choosedClass = DataReader.ChooseClass(dataSet, classDistribution);
            classifyAction.classification_int = int.Parse(choosedClass);
            classifyAction.classification_string = choosedClass;
            classifyAction.reward = DataReader.CalculateMisClassificationValue(dataSet, classDistribution, choosedClass);

            sourceState.classDistribution = classDistribution;
            return true;
            /*classifyAction.classification_int = int.Parse(bestClass.Key);
            classifyAction.classification_string = bestClass.Key.ToString();
            classifyAction.reward = DataReader.CalculateSimpaleValue(accuracy);*/
        }



        public static bool isConsistWithReducedSet(Dictionary<string, Formula> attributeValues, Dictionary<string, HashSet<Formula>> setOfReduceRules)
        {
            foreach(var insVal in attributeValues)
            {
                if(setOfReduceRules.ContainsKey(insVal.Key))
                {
                    foreach(var reducedVal in setOfReduceRules[insVal.Key])
                    {
                        if (insVal.Value.IsWeakStronger(reducedVal))
                            return false;
                    }
                }
            }
            return true;
        }

    }
}
