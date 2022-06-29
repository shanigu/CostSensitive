using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using weka.classifiers.meta;
using weka.classifiers.trees;
using weka.core;
using weka.filters.unsupervised.attribute;

namespace CostSensitiveTree
{
    class Tree
    {
        public enum TreeType {C45, EG2, CSID3, IDX, EG2_v1, CSID3_v1, IDX_v1, MetaCost, LowerCost };
        public static int ID=0;
        public int id;
        public bool[] m_attributesVector = null;
        public weka.classifiers.Classifier tree = null;
        public const int EmptyValue = int.MinValue;
        public List<Path> paths = null;
        public DataReader.DataSet m_dataSet = null;
        public int[] indices = null;
        public HashSet<string> attributesNames = null;
        public HashSet<Path> reducedPaths = null;
        public Tree(bool[] attributesVector, DataReader.DataSet dataSet)
        {
            id = ID++;

            m_attributesVector = attributesVector;
            m_dataSet = dataSet;
            int countOfAtt =1;
            for (int i = 0; i < attributesVector.Length; i++)
            {
                if (attributesVector[i])
                {
                    countOfAtt++;
                }
            }
            indices = new int[countOfAtt];
            attributesNames = new HashSet<string>();
            int index = 0;
            for (int i=0;i< attributesVector.Length;i++)
            {
                if(attributesVector[i])
                {
                    indices[index] = i;
                    attributesNames.Add(dataSet.m_features.Values.FirstOrDefault(f => f.attIndex == i).name);
                    index++;
                }
            }
            indices[index] = attributesVector.Length;
        }

        public void BuildTree(Instances trainDataForTreeCreation, TreeType treeType,DataReader.DataSet dataSet)
        {
            tree = GetJ48Tree(out paths, trainDataForTreeCreation, treeType, dataSet);
        }

        private weka.classifiers.Classifier GetJ48Tree(out List<Path> paths, Instances trainDataForTreeCreation, TreeType treeType, DataReader.DataSet dataSet)
        {
            Instances reducedInst = new Instances(trainDataForTreeCreation);
            Remove attributeFilter = new Remove();
            attributeFilter.setInvertSelection(true);
            attributeFilter.setAttributeIndicesArray(indices);
            attributeFilter.setInputFormat(reducedInst);
            reducedInst = weka.filters.Filter.useFilter(reducedInst, attributeFilter);

            // copy cost
            var enumerateAttributes = reducedInst.enumerateAttributes();
            while (enumerateAttributes.hasMoreElements())
            {
                weka.core.Attribute newAtt = (weka.core.Attribute)enumerateAttributes.nextElement();
                weka.core.Attribute orgAtt = trainDataForTreeCreation.attribute(newAtt.name());
                newAtt.m_Cost = orgAtt.m_Cost;
            }
            string strTreeType = treeType == TreeType.C45 ? "C45" : (treeType == TreeType.CSID3 ? "CSID3" : (treeType == TreeType.EG2 ? "EG2" : (treeType == TreeType.IDX ? "IDX" : (treeType == TreeType.CSID3_v1 ? "CSID3_v1" : (treeType == TreeType.EG2_v1 ? "EG2_v1" : (treeType == TreeType.IDX_v1 ? "IDX_v1" : (treeType == TreeType.MetaCost ? "MetaCost" : (treeType == TreeType.LowerCost ? "LowerCost" : ""))))))));


            if (strTreeType == "MetaCost")
            {
                weka.classifiers.CostMatrix costMatrix = new weka.classifiers.CostMatrix(dataSet.m_classificationCostMetrix.Count);
                for (int i = 0; i < dataSet.m_classificationCostMetrix.Count; i++)
                {
                    for (int j = 0; j < dataSet.m_classificationCostMetrix.Count; j++)
                    {
                        string predict = dataSet.m_classificationCostMetrix.ElementAt(i).Key;
                        string trueVal = dataSet.m_classificationCostMetrix.ElementAt(i).Value.ElementAt(j).Key;
                            costMatrix.setElement(j, i, -1 * dataSet.m_classificationCostMetrix[predict][trueVal]);
                    }
                }
                weka.classifiers.meta.MetaCost MetaCostJ48 = new weka.classifiers.meta.MetaCost();
                MetaCostJ48.setNumIterations(10);

                J48 t48 = new J48();
               // t48.setUnpruned(true);
                MetaCostJ48.setClassifier(t48);
                MetaCostJ48.setCostMatrix(costMatrix);

                MetaCostJ48.buildClassifier(reducedInst);
                string MetaCostJ48treeStr = MetaCostJ48.toString();
                paths = GetTreePaths(MetaCostJ48treeStr);
                return MetaCostJ48;
            }

            CostSensitiveClassifier costSensitiveClassifier = new CostSensitiveClassifier();

            J48 j48 = new J48(); 

            j48.SetSplitType(strTreeType);

            j48.setUnpruned(false);
            // Train & build
            j48.buildClassifier(reducedInst);
            string treeStr = j48.toString();
            paths = GetTreePaths(treeStr);
            return j48;
        }


        public List<Path> GetTreePaths(string tree)
        {

            string[] strLines = tree.Split('\n');
            List<string> lines = new List<string>();
            bool startCopy = false;
            bool f1 = false;
            foreach (string str in strLines)
            {
                if (startCopy)
                {
                    if (str != "")
                        lines.Add(str);
                    else
                        break;
                }
                if (f1)
                {
                    f1 = false;
                    startCopy = true;
                    if (str != "")
                        lines.Add(str);
                }
                if (str == "------------------")
                {
                    f1 = true;
                }

            }
            List<List<String>> lists = new List<List<string>>();
            // Break lines into parts.
            foreach (string orgLine in lines)
            {
                List<string> temp = new List<string>();
                string line = orgLine;
                while (line.IndexOf("|") != -1)
                {
                    temp.Add("|");
                    int index = line.IndexOf("|");
                    line = index >= 0 ? line.Remove(index, index + 4) : line;
                }
                temp.Add(line.Trim());
                lists.Add(temp);
            }



            // This is a ordered list of parents for any given node while traversing the tree.
            List<string> parentClauses = new List<string>();
            // this describes the depth
            //int depth = -1;

            // all the paths in the tree.
            List<List<string>> paths = new List<List<string>>();


            foreach (List<string> list in lists)
            {
                int currDepth = 0;
                for (int i = 0; i < list.Count(); i++)
                {
                    string token = list[i];
                    // find how deep is this node in the tree.
                    if (token.Equals("|"))
                    {
                        currDepth++;
                    }
                    else
                    {    // now we get to the string token for the node.
                         // if leaf, so we get one path..
                        if (token.Contains(":"))
                        {
                            List<string> path = new List<string>();
                            for (int index = 0; index < currDepth; index++)
                            {
                                path.Add(parentClauses[index]);
                            }

                            char[] del = new char[] { ':' };
                            string[] tmpStr = token.Split(del);
                            foreach (string str in tmpStr)
                            {
                                if (str != "")
                                    path.Add(str);
                            }
                            paths.Add(path);
                        }
                        else
                        {
                            // add this to the current parents list
                            if (parentClauses.Count > currDepth)
                                parentClauses[currDepth] = token;
                            else
                                parentClauses.Add(token);
                        }
                    }
                }
            }

            List<Path> pathsOfNodes = new List<Path>();
            foreach (List<string> path in paths)
            {
                int classIndex = -1;
                Path pathOfNodes = new Path();
                pathOfNodes.SetSourceTree(this);
                foreach (string strNode in path)
                {
                    Node node = new Node();
                    if (strNode.Contains('('))
                    {
                        char[] del = new char[] { '(', ')', ' ', '/' };
                        string[] tmpStr = strNode.Split(del);
                        double firstNum = double.Parse(tmpStr[3]);
                        double secoundNum = 0;
                        if (tmpStr.Length > 5)
                            secoundNum = double.Parse(tmpStr[4]);
                        double ratio = (firstNum - secoundNum) / (firstNum);
                        node.ratioValue = ratio;
                        node.ratioStr = firstNum + "/" + secoundNum;
                        int val;
                        if (int.TryParse(tmpStr[1], out val))
                        {
                            node.classValue_Int = val;
                            node.classValue_String = "";
                        }
                        else
                        {
                            node.classValue_Int = EmptyValue;
                            node.classValue_String = tmpStr[1];
                        }
                        node.formula.attributeName = "Leaf";
                        node.formula.symbol = "";
                        node.isLeafNode = true;
                        node.formula.attributeValue_double = -1;
                    }
                    else
                    {
                        int result;
                        if (int.TryParse(strNode, out result))
                        {
                            classIndex = result;
                        }
                        else
                        {
                            char[] del = new char[] { ' ' };
                            string[] tmpStr = strNode.Split(del);
                            node.ratioStr = "";
                            node.formula.attributeName = tmpStr[0];
                            node.formula.symbol = tmpStr[1];
                            node.isLeafNode = false;
                            node.classValue_Int = -1;
                            node.ratioValue = -1;
                            double val;
                            if (double.TryParse(tmpStr[2], out val))
                            {
                                node.formula.attributeValue_double = val;
                                node.formula.attributeValue_string = "";
                            }
                            else
                            {
                                node.formula.attributeValue_double = EmptyValue;
                                node.formula.attributeValue_string = tmpStr[2];
                            }
                        }
                    }
                    pathOfNodes.AddNode(node);
                }
                pathsOfNodes.Add(pathOfNodes);
            }
            return pathsOfNodes;
        }

        public override string ToString()
        {
            string toString = "";
                if (this.attributesNames != null)
                {
                    toString += "(  ";
                    foreach (string attName in this.attributesNames)
                    {
                        toString += attName + "  ";
                    }
                    toString += ")";
                }
            return toString;
        }

        public class Node
        {
            public Formula formula;
            public bool isLeafNode;
            public int classValue_Int;
            public string classValue_String;
            public double ratioValue;
            public string ratioStr;

            public Node()
            {
                formula = new Formula();
            }
            public override string ToString()
            {
                if (!isLeafNode)
                {
                    if (formula.attributeValue_string == "")
                        return (formula.attributeName + " " + formula.symbol + " " + formula.attributeValue_double);
                    else
                        return (formula.attributeName + " " + formula.symbol + " " + formula.attributeValue_string);
                }
                else
                {
                    if (classValue_String == "")
                        return ("Class: " + classValue_Int + "  (" + ratioStr + ")" + "  [" + ratioValue + "]");
                    else
                        return ("Class: " + classValue_String + "  (" + ratioStr + ")" + "  [" + ratioValue + "]");
                }
            }

            public override bool Equals(object obj)
            {
                if(obj is Node)
                {
                    Node node2 = (Node)obj;
                    if(formula.attributeName.Equals(node2.formula.attributeName))
                    {
                        if(formula.symbol.Equals(node2.formula.symbol))
                        {
                            if (formula.attributeValue_double.Equals(node2.formula.attributeValue_double) & formula.attributeValue_string.Equals(node2.formula.attributeValue_string))
                                return true;
                        }
                    }
                }
                return false;
            }

            public override int GetHashCode()
            {
                return formula.attributeName.GetHashCode();
            }
        }

    }
}
