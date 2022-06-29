using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using weka.core;

namespace CostSensitiveTree
{
    class MDPModel
    {
        public List<State> stateList { get; set; }
        public List<Action> classifyActions { get; set; }
        public List<Action> queryActions { get; set; }
        public Dictionary<State, Action> optimalPolicy { get; set; }
        public DataReader.DataSet dataSet { get; set; }
        public Dictionary<string, HashSet<Formula>> globalReduceRules { get; set; }



        public MDPModel(List<State> m_stateList,List<Action> m_classifyActions, List<Action> m_queryActions, DataReader.DataSet m_dataSet)
        {
            stateList = m_stateList;
            classifyActions = m_classifyActions;
            queryActions = m_queryActions;
            dataSet = m_dataSet;
            globalReduceRules = new Dictionary<string, HashSet<Formula>>();
        }


        public MDPModel Clone()
        {
            Dictionary<State, State> oldNewStatesMapper = new Dictionary<State, State>();
            foreach(State oldState in stateList)
            {
                oldNewStatesMapper.Add(oldState, oldState.Clone());
            }
            Dictionary<Action, Action> actionMapper = new Dictionary<Action, Action>();
            List<Action> m_classifyActions = new List<Action>();
            foreach(Action cAction in classifyActions)
            {
                m_classifyActions.Add(cAction.Clone(oldNewStatesMapper));
                actionMapper.Add(cAction, m_classifyActions.Last());
            }
            List<Action> m_queryActions = new List<Action>();
            foreach (Action qAction in queryActions)
            {
                m_queryActions.Add(qAction.Clone(oldNewStatesMapper));
                actionMapper.Add(qAction, m_queryActions.Last());
            }

            foreach(State newState in oldNewStatesMapper.Values)
            {
                newState.ingoingActions = newState.ingoingActions.Select(s => actionMapper[s]).ToList();
                newState.outgoingActions = newState.outgoingActions.Select(s => actionMapper[s]).ToList();
            }

            MDPModel newMdpModel = new MDPModel(oldNewStatesMapper.Values.ToList(), m_classifyActions, m_queryActions, dataSet);

            return newMdpModel;
        }

        public static int debugCounter=0;

        public void UpdateStateAndAction(Dictionary<string, Formula> knownAttributesValue) // TODO: fix
        {
            HashSet<Action> updateActions = new HashSet<Action>();
            HashSet<State> toUpdateList= new HashSet<State>();
            //            results += utilityValue1.Value.ToString() + "   " + utilityValue1.Key.ToString() + "   ";
            //Console.WriteLine("ID   " + debugCounter++);
            foreach (State state in stateList.ToList())
            {
                if(state.attributes.Intersect(knownAttributesValue.Keys).Count() != knownAttributesValue.Keys.Count)
                {
                    state.valid = false;
                    stateList.Remove(state);
                    continue;
                }

                var falseRulesList = state.PartialConsistWith(knownAttributesValue);
                if (falseRulesList.Count > 0)
                {
                    foreach (var falseRule in falseRulesList)
                    {
                        string newAttName = falseRule.attributeName;
                        state.valid = false;
                        foreach (Action act in state.ingoingActions.ToList())
                        {
                            if (stateList.Contains(act.sourceState))
                            {

                                updateActions.Add(act);
                                toUpdateList.Add(act.sourceState);
                                act.targetStates.Remove(state);

                                if (!act.sourceState.setOfReduceRules.ContainsKey(newAttName))
                                    act.sourceState.setOfReduceRules.Add(newAttName, new HashSet<Formula>());
                                act.sourceState.setOfReduceRules[newAttName].Add(falseRule);

                                if (!act.setOfReduceRules.ContainsKey(newAttName))
                                    act.setOfReduceRules.Add(newAttName, new HashSet<Formula>());
                                act.setOfReduceRules[newAttName].Add(falseRule);
                               // if (act.setOfReduceRules.ElementAt(0).Value.Count > 1)
                               //     Console.WriteLine();


                                if (!globalReduceRules.ContainsKey(newAttName))
                                    globalReduceRules.Add(newAttName, new HashSet<Formula>());
                                globalReduceRules[newAttName].Add(falseRule);

                                if (act.targetStates.Count == 0)
                                {
                                    act.sourceState.outgoingActions.Remove(act);
                                }
                                else
                                {
                                    double sum = act.targetStates.Values.Sum();
                                    foreach (State s in act.targetStates.Keys.ToList())
                                    {
                                        act.targetStates[s] = act.targetStates[s] / (double)sum;
                                    }
                                }
                            }
                        }
                        stateList.Remove(state);
                    }
                }
            }

            foreach (State sourceState in toUpdateList)
            {
                if (sourceState.valid)
                {
                    ActionsGenerator.UpdateTransitionBySourceStateV2Org(sourceState, dataSet, globalReduceRules);
                }               
            }
            ValueIteration(knownAttributesValue, false);
            stateList = stateList.OrderBy(s => s.attributes.Count).ToList();
        }




        public void ValueIteration(Dictionary<string, Formula> knownAtts, bool bPrint = true)
        {
            if (bPrint)
            {
                System.Console.WriteLine();
                System.Console.WriteLine("Value Iteration:");
            }
            optimalPolicy = new Dictionary<State, Action>();
            HashSet<State> currentStateLayer =new HashSet<State>( stateList.Where(s => s.attributes.Count == dataSet.m_trainInstances2.numAttributes() - 1));
            HashSet<State> nextStateLayer = null;
            List<State> allState1 = new List<State>();
            int attNum = dataSet.m_trainInstances2.numAttributes() - 2;
            HashSet<State> allState2 = new HashSet<State>();
            bool stop = false;
            while (!stop)
            {
                stop = true;
                nextStateLayer = new HashSet<State>();
                foreach(State s in currentStateLayer)
                {
                    //if (s.attributes.Count== 0)
                    //    Console.Write("");
                    s.UpdateValue(dataSet, knownAtts);
                    optimalPolicy.Add(s, s.bestAction);
                   /* foreach(Action action in s.ingoingActions)
                    {
                        if(action.sourceState.valid)
                            nextStateLayer.Add(action.sourceState);
                    }*/

                    allState1.Add(s);
                    allState2.Add(s);
                }
                if(attNum>=0)
                {
                    foreach (State prevLayer in stateList.Where(s => s.attributes.Count == attNum))
                    {
                        nextStateLayer.Add(prevLayer);
                    }
                    attNum--;
                }
                if (nextStateLayer.Count > 0)
                {
                    stop = false;
                    currentStateLayer = nextStateLayer;
                }
                if (bPrint)
                    System.Console.Write("\r{0}%   ", Math.Round(100 * (double)allState1.Count / (double)stateList.Count, 3));
            }
            if(allState1.Count > stateList.Count)
            {
                throw new Exception("Loop");
            }
            if (allState2.Count != stateList.Count(s=>s.valid))
            {
                var ex1 = stateList.Except(allState2);
               // throw new Exception("Unaccessible States");
            }
        }

    }
}
