using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using weka.core;

namespace CostSensitiveTree
{
    class Path
    {
        List<Tree.Node> nodes = null;
        Tree sourceTree = null;
        public State state = null;
        public double accuracy = -1;
        public bool isFullPath = true;
        public Tree.Node lastNode = null;
        public HashSet<string> attributes { get; set; }
        public HashSet<string> consideredAttributes { get; set; }

        public Dictionary<string, double> classDistribution = null;

        public static Path GetEmptyPath(DataReader.DataSet dataSet)
        {
            Path path = new Path();
            Tree.Node node = new Tree.Node();
            node.isLeafNode = true;
            Dictionary<string, double> classDistribution = new Dictionary<string, double>();
            var instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues = GetAttributesValue(ins);
                int attCount = dataSet.m_testInstances.numAttributes();
                string trueClass = ins.stringValue(attCount - 1);
                if (!classDistribution.ContainsKey(trueClass))
                    classDistribution.Add(trueClass, 0);
                classDistribution[trueClass]++;
            }

            double sum = classDistribution.Sum(cd => cd.Value);
            classDistribution = classDistribution.ToDictionary(c => c.Key, c => (double)c.Value / sum);
            var choosedClass = DataReader.ChooseClass(dataSet, classDistribution);
            node.classValue_Int = int.Parse(choosedClass);
            node.classValue_String = choosedClass;
            path.lastNode = node;
            //path.sourceTree = new Tree()
            return path;
        }


        public Path()
        {
            nodes = new List<Tree.Node>();
            attributes = new HashSet<string>();
            classDistribution = new Dictionary<string, double>();
        }
        public void AddNode(Tree.Node node)
        {
            nodes.Add(node);
            if (node.isLeafNode)
            {
                accuracy = node.ratioValue;
                lastNode = node;
            }
            else
            {
                attributes.Add(node.formula.attributeName);
            }
        }

        public void AddUniqueNode(Tree.Node node)
        {
            if(!nodes.Contains(node))
                nodes.Add(node);
        }

        public Tree.Node GetNode(int index)
        {
            return nodes[index];
        }

        public int Count()
        {
            return nodes.Count;
        }

        public Path GetReducePath(HashSet<string> indicesSet, HashSet<string> p_consideredAttributes)
        {
            Path newPath = new Path();
            consideredAttributes = p_consideredAttributes;
            newPath.sourceTree = sourceTree;
            newPath.accuracy = accuracy;
            newPath.lastNode = lastNode;
            newPath.nodes = nodes.Where(n => indicesSet.Contains(n.formula.attributeName)).ToList();
            //newPath.nodes = newPath.nodes.OrderBy(n => n.formula.attributeName).ToList();
            newPath.attributes = new HashSet<string>(newPath.nodes.Select(n => n.formula.attributeName));
            //if (newPath.nodes.Count == 0)
            //    return null;
            foreach (Tree.Node node in nodes)
            {
                if (!node.isLeafNode & !indicesSet.Contains(node.formula.attributeName))
                {
                    newPath.isFullPath = false;
                    newPath.accuracy = -1;
                    newPath.lastNode = null;
                }
            }


            if (newPath.isFullPath)
            {
                newPath.classDistribution = classDistribution;
            }
            else
            {
                throw new Exception("bug: we consider only full path");
            }

            return newPath;
        }

        public override bool Equals(object obj)
        {
            if (obj is Path)
            {
                Path path2 = (Path)obj;
                if (path2.Count() == Count() && path2.sourceTree==sourceTree)
                {
                    for (int i = 0; i < Count(); i++)
                    {
                        if (!path2.GetNode(i).Equals(GetNode(i)))
                        {
                            return false;
                        }
                    }
                    return true;
                }
            }
            return false;
        }

  

        public bool Contains(List<Tree.Node> l1, List<Tree.Node> l2)
        {
            if (l1 == null || l2 == null)
                return false;
            if (l1.Count != l2.Count)
                return false;
            foreach (Tree.Node n1 in l1)
            {
                if (!l2.Contains(n1))
                    return false;
            }
            return true;
        }

        int hashCode = -1;
        public override int GetHashCode()
        {
            if (hashCode == -1)
            {
                hashCode = 0;
                foreach (Tree.Node node in nodes)
                {
                    hashCode += node.formula.attributeName.GetHashCode();
                }
            }
            return hashCode;
        }

        string toStringValue = "";
        public override string ToString()
        {
            if (toStringValue == "")
            {
                if (nodes.Count > 0)
                {
                    foreach (Tree.Node node in nodes)
                    {
                        toStringValue += node.ToString() + ",  ";
                    }
                    toStringValue = toStringValue.Remove(toStringValue.Length - 3, 3);
                }
                else
                    toStringValue = "Empty";
            }
            return toStringValue;
        }

        public void SetSourceTree(Tree tree)
        {
            sourceTree = tree;
        }

        public Tree GetSourceTree()
        {
            return sourceTree;
        }

        public string CanPass(Path path2)
        {
            HashSet<string> differenceSet = new HashSet<string>();

            int counter = path2.attributes.Where(att => !attributes.Contains(att)).Count();
            if (counter != 1)
                return null;

            differenceSet = new HashSet<string>( path2.nodes.Where(n => !nodes.Contains(n)).Select(n => n.formula.attributeName).ToList());
            if (differenceSet.Count > 1)
                return null;

            return differenceSet.ElementAt(0);
        }

        public List<Tree.Node> filterByAttName(string attName)
        {
            return this.nodes.Where(N => N.formula.attributeName == attName).ToList();
        }
        public System.Collections.IEnumerator GetEnumerator()
        {
            foreach(Tree.Node node in nodes)
            {
                yield return node;
            }
        }

        public List<Tree.Node> GetNodeList()
        {
            return nodes;
        }


        public static Dictionary<Instance, Dictionary<string, Formula>> InstanceAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();

        public static Dictionary<string,Formula> GetAttributesValue(Instance ins)
        {
            if (!InstanceAttributeValues.ContainsKey(ins))
            {
                Dictionary<string, Formula> attributesValue = new Dictionary<string, Formula>();
                for (int i = 0; i < ins.numValues() - 1; i++)
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
                InstanceAttributeValues[ins] = attributesValue;
            }
            return InstanceAttributeValues[ins];
        }


        public bool ConsistWith(Dictionary<string, Formula> attributeValues)
        {
            foreach(Tree.Node node in nodes)
            {
                if (!attributeValues.ContainsKey(node.formula.attributeName) || !attributeValues[node.formula.attributeName].IsWeakStronger(node.formula))
                    return false;
            }
            return true;
        }

        private bool IOppositetWith(Path path2)
        {
            foreach (Tree.Node node in nodes)
            {
                List<Tree.Node> relevantNodes = path2.nodes.Where(n => n.formula.attributeName == node.formula.attributeName).ToList();
                foreach (Tree.Node relevantNode in relevantNodes)
                {
                    if (node.formula.IOppositeFormula(relevantNode.formula))
                        return true;
                }
            }
            return false;
        }


        public Dictionary<Path,double> GetTransition(DataReader.DataSet dataSet,List<Path> paths)
        {
            Dictionary<Path, double> pathCounter = new Dictionary<Path, double>();
            Dictionary<Instance, Dictionary<string, Formula>> instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
            var instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues =  GetAttributesValue(ins);
                if(this.ConsistWith(attributeValues))
                {
                    instancesAttributeValues.Add(ins, attributeValues);
                }
            }
            foreach(var item in instancesAttributeValues)
            {
                bool bug = true;
                foreach(Path path in paths)
                {
                    if(path.ConsistWith(item.Value))
                    {
                        if(!pathCounter.ContainsKey(path))
                        {
                            pathCounter.Add(path, 0.0);
                        }
                        pathCounter[path]++;
                        bug = false;
                        //break;
                    }
                }
                if (bug)
                    throw new Exception("Instance not fit to any path");
            }
            double sum = pathCounter.Values.Sum(v => v);
            if (sum != instancesAttributeValues.Count)
                throw new Exception("Instance fit more than one path");
            Dictionary<Path, double> pathRatio = new Dictionary<Path, double>();
            foreach (var item in pathCounter)
            {
                pathRatio.Add(item.Key,item.Value / (double)sum);
            }
            return pathRatio;
        }


        public Dictionary<Path, double> GetTransitionV2(DataReader.DataSet dataSet, List<Path> paths)
        {
            Dictionary<Path, double> pathCounter = new Dictionary<Path, double>();
            Dictionary<Instance, Dictionary<string, Formula>> instancesAttributeValues = new Dictionary<Instance, Dictionary<string, Formula>>();
            var instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues = GetAttributesValue(ins);
                instancesAttributeValues.Add(ins, attributeValues);
            }

            //  dont delete ins that not fit the source path 
            // reduced Opposite path insted 
            List<Path> relevantPaths = paths.Where(p2 => !IOppositetWith(p2)).ToList();


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
                        bug = false;
                    }
                }
            }
            double sum = pathCounter.Values.Sum(v => v);
            if (sum > instancesAttributeValues.Count)
                throw new Exception("Instance fit more than one path");
            Dictionary<Path, double> pathRatio = new Dictionary<Path, double>();
            foreach (var item in pathCounter)
            {
                pathRatio.Add(item.Key, item.Value / (double)sum);
            }
            return pathRatio;
        }


        public Dictionary<Path, double> GetTransitionV3(DataReader.DataSet dataSet, List<Path> paths)
        {
            Dictionary<Path, double> pathCounter = new Dictionary<Path, double>();

            //  dont delete ins that not fit the source path 
            // reduced Opposite path insted 
            List<Path> relevantPaths = paths.Where(p2 => !IOppositetWith(p2)).ToList();



            foreach (Path path in relevantPaths)
            {
                pathCounter.Add(path, (double)path.state.instances.Count);
            }
            
            double sum = pathCounter.Values.Sum(v => v);
            Dictionary<Path, double> pathRatio = new Dictionary<Path, double>();
            foreach (var item in pathCounter)
            {
                pathRatio.Add(item.Key, item.Value / (double)sum);
            }
            return pathRatio;
        }


        public Path SetClassDistribution(DataReader.DataSet dataSet)
        {
            var instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues = GetAttributesValue(ins);
                if (ConsistWith(attributeValues))
                {
                    int attCount = dataSet.m_testInstances.numAttributes();
                    string trueClass = ins.stringValue(attCount - 1);
                    if (!classDistribution.ContainsKey(trueClass))
                        classDistribution.Add(trueClass, 0);
                    classDistribution[trueClass]++;
                }
            }

            double sum = classDistribution.Sum(cd=>cd.Value);
            var classDistributionorgCount = classDistribution;
            classDistribution = classDistribution.ToDictionary(c => c.Key, c => (double)c.Value / sum);
            sum = classDistribution.Sum(cd => cd.Value);
            var choosedClass = DataReader.ChooseClass(dataSet, classDistribution);
            double newAccuracy = classDistribution[choosedClass] / sum;
            if (newAccuracy != accuracy)
                accuracy = newAccuracy;
            lastNode.classValue_Int = int.Parse(choosedClass);
            lastNode.classValue_String = choosedClass;

            return this;
        }

        public Path SetClassDistributionByAccuracy(DataReader.DataSet dataSet)
        {
            var instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues = GetAttributesValue(ins);
                if (ConsistWith(attributeValues))
                {
                    int attCount = dataSet.m_testInstances.numAttributes();
                    string trueClass = ins.stringValue(attCount - 1);
                    if (!classDistribution.ContainsKey(trueClass))
                        classDistribution.Add(trueClass, 0);
                    classDistribution[trueClass]++;
                }
            }

            double max = -1;
            string choosedClass = "";
            foreach (var classCount in classDistribution)
            {
                if(classCount.Value>max)
                {
                    choosedClass = classCount.Key;
                    max = classCount.Value;
                }
            }

            double newAccuracy = max / classDistribution.Values.Sum();
            if (newAccuracy != accuracy)
                accuracy = newAccuracy;
            lastNode.classValue_Int = int.Parse(choosedClass);
            lastNode.classValue_String = choosedClass;

            return this;
        }

        public Path SetClassDistributionByDef(DataReader.DataSet dataSet)
        {

            string[] strArr = lastNode.ToString().Split(' ');
            string choosedClass = strArr[1];

            double tp = 0;
            double total = 0;
            var instancesIterator = dataSet.m_trainInstances2.enumerateInstances();
            while (instancesIterator.hasMoreElements())
            {
                Instance ins = (Instance)instancesIterator.nextElement();
                Dictionary<string, Formula> attributeValues = GetAttributesValue(ins);
                if (ConsistWith(attributeValues))
                {
                    int attCount = dataSet.m_trainInstances2.numAttributes();
                    string trueClass = ins.stringValue(attCount - 1);
                    if (choosedClass == trueClass)
                    {
                        tp++;
                    }
                    total++;
                }

            }

            accuracy = tp / total;



            lastNode.classValue_Int = int.Parse(choosedClass);
            lastNode.classValue_String = choosedClass;

            return this;
        }

    }
}
