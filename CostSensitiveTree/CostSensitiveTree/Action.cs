using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CostSensitiveTree
{
    class Action
    {
        private static int ID = 0;
        public int id { get; set; }
        public State sourceState { get; set; }
        public Dictionary<State,double> targetStates { get; set; }
        public double reward { get; set; }

        public string attributeName = string.Empty;
        public bool isClassificationAction { get; set; }
        public int classification_int { get; set; }
        public string classification_string { get; set; }
        public double askProbability { get; set; } = 0.0;
        public Dictionary<string, HashSet<Formula>> setOfReduceRules { get; set; }  = null;

        public Action()
        {
            id = ID;
            ID++;
            setOfReduceRules = new Dictionary<string, HashSet<Formula>>();
        }

        public Action(Dictionary<Path,State> mapper, Path sourcePath, Dictionary<Path,double> trantisions, double m_reward, string m_attributeName)
        {
            id = ID;
            ID++;
            sourceState = mapper[sourcePath];
            targetStates = new Dictionary<State, double>();
            foreach (var item in trantisions)
            {
                targetStates.Add(mapper[item.Key], item.Value);
            }
            reward = m_reward;
            attributeName = m_attributeName;
            setOfReduceRules = new Dictionary<string, HashSet<Formula>>();
        }

        public Action(State m_sourceState, Dictionary<State, double> trantisions, double m_reward, string m_attributeName)
        {
            id = ID;
            ID++;
            sourceState = m_sourceState;
            targetStates = new Dictionary<State, double>();
            foreach (var item in trantisions)
            {
                targetStates.Add(item.Key, item.Value);
            }
            reward = m_reward;
            attributeName = m_attributeName;
            setOfReduceRules = new Dictionary<string, HashSet<Formula>>();
        }

        public Action(Dictionary<Path, State> mapper, Path sourcePath, int classInt,string classString, double m_reward)
        {
            id = ID;
            ID++;
            sourceState = mapper[sourcePath];
            isClassificationAction = true;
            classification_int = classInt;
            classification_string = classString;
            reward = m_reward;
            setOfReduceRules = new Dictionary<string, HashSet<Formula>>();
        }

        public Action(State m_sourceState, int classInt, string classString, double m_reward)
        {
            id = ID;
            ID++;
            sourceState = m_sourceState;
            isClassificationAction = true;
            classification_int = classInt;
            classification_string = classString;
            reward = m_reward;
            setOfReduceRules = new Dictionary<string, HashSet<Formula>>();
        }


        public Action Clone(Dictionary<State,State> oldToNewStatesMapper)
        {
            Action newAction = new Action { attributeName = this.attributeName, classification_int = this.classification_int, classification_string = this.classification_string, isClassificationAction = this.isClassificationAction, reward = this.reward, sourceState = oldToNewStatesMapper[this.sourceState] };
            if (!isClassificationAction)
            {
                newAction.targetStates = new Dictionary<State, double>();
                foreach (var item in targetStates)
                {
                    newAction.targetStates.Add(oldToNewStatesMapper[item.Key], item.Value);
                }
            }
            setOfReduceRules = new Dictionary<string, HashSet<Formula>>(setOfReduceRules);
            return newAction;
        }
        public override string ToString()
        {
            if (isClassificationAction)
                return "Classify as: " + classification_int;
            else
            {
                return "Query on: " + attributeName;
            }
        }
    }
}
