using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using weka;
using weka.core;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using java.io;
using weka.filters.unsupervised.attribute;
using weka.classifiers.trees;
using weka.classifiers.meta;
using System.Text.RegularExpressions;
using System.Collections;

namespace CostSensitiveTree
{
    class Program
    {
        static void Main(string[] args)
        {
            RunExperiment(@"..\..\..\..\Datasets\", "car", 1);
        }

        static void RunExperiment(string sDataPath, string domainName, int costIndex, int cAttrbiutesInAllTrees = -1)
        {

            string times = "";
            double time = 0.0;
            DataReader.DataSet dataSet = DataReader.ReadData(sDataPath + domainName, costIndex, costIndex);
            HashSet<string> initialyKnow = null;
            //cAttrbiutesInAllTrees == -1 for learning trees for all attributes
            List<bool[]> permutations = GetAllPermutations(dataSet.m_features.Count - 1, cAttrbiutesInAllTrees, dataSet, out initialyKnow);
            List<Tree> trees = new List<Tree>();
            Tree t = null;


            DateTime start = DateTime.Now;
            System.Console.WriteLine("Build Trees:");
            int counter = 1;
            foreach (bool[] attributesVector in permutations)
            {
                t = new Tree(attributesVector, dataSet);
                t.BuildTree(dataSet.m_trainInstances1,Tree.TreeType.C45,dataSet);
                trees.Add(t);
                System.Console.Write("\r{0}%   ", Math.Round(100 * (double)counter++ / (double)permutations.Count, 3));
            }


            time = DateTime.Now.Subtract(start).TotalSeconds;
            times += time.ToString() + " ";
            start = DateTime.Now;

            trees = trees.OrderBy(tree => tree.indices.Length).ToList();
            HashSet<Path> uniquePath = new HashSet<Path>();
            System.Console.WriteLine();
            System.Console.WriteLine("Find Path:");
            counter = 1;
            HashSet<Path> newPaths;
            int cLeaves = 0;

            foreach (Tree tree in trees)
            {
                newPaths = new HashSet<Path>();
                GetAllRelevantPaths(tree, tree.attributesNames, dataSet, newPaths);
                tree.reducedPaths = newPaths;
                uniquePath.UnionWith(newPaths);
                System.Console.Write("\r{0}%   ", 100 * Math.Round((double)counter++ / (double)trees.Count, 3));
            }
            cLeaves = uniquePath.Count;
            time = DateTime.Now.Subtract(start).TotalSeconds;
            times += time.ToString() + " ";
            start = DateTime.Now;
            System.Console.WriteLine("unique paths = " + cLeaves);

            System.Console.WriteLine("Setting instances:");
            Dictionary<Path, State> pathToStateMapper = new Dictionary<Path, State>();
            counter = 0;
            foreach (Path path in uniquePath)
            {
                State state = new State(path);
                pathToStateMapper.Add(path, state);
                state.SetRelevantInstancesAndClassDistribution(dataSet);
                //path.SetClassDistribution(dataSet);
                //state.SetRelevantInstances(dataSet);
                System.Console.Write("\r{0}%   ", 100 * Math.Round((double)counter / (double)uniquePath.Count, 3));
                counter++;
            }
            time = DateTime.Now.Subtract(start).TotalSeconds;
            times += time.ToString() + " ";
            start = DateTime.Now;

            List<State> allStates = pathToStateMapper.Values.ToList();
            var classifyActions = ActionsGenerator.AddClassificationActions(allStates, dataSet);
            var queryActions = ActionsGenerator.AddQueryAction(trees, pathToStateMapper, dataSet);

            time = DateTime.Now.Subtract(start).TotalSeconds;
            times += time.ToString() + " ";
            start = DateTime.Now;

            MDPModel mdpModel = new MDPModel(allStates, classifyActions, queryActions, dataSet);
            mdpModel.ValueIteration(new Dictionary<string, Formula>());

            List<State> states = mdpModel.stateList.OrderBy(s => s.attributes.Count).ToList();
            //Set Instances type
            ModelEvaluation.SetInstances(dataSet.m_testInstances);

            // var utilityValue0 = ModelEvaluation.EvaluateModelByGraph(mdpModel, initialyKnow);
            time = DateTime.Now.Subtract(start).TotalSeconds;
            times += time.ToString() + " ";
            start = DateTime.Now;

            var utilityValue1 = ModelEvaluation.EvaluateModel(mdpModel, sDataPath, initialyKnow);
            time = DateTime.Now.Subtract(start).TotalSeconds;
            times += time.ToString() + " ";

            start = DateTime.Now;

            var utilityValue2 = ModelEvaluation.EvaluateModelWithIntermediateUpdates(mdpModel, sDataPath, initialyKnow);

            time = DateTime.Now.Subtract(start).TotalSeconds;
            times += time.ToString() + " ";

            //bugbug;
            if (false)//value updates during classification
            {
                //if (utilityValue1.Key < utilityValue2.Key)
                //utilityValue1 = utilityValue2;
            }   
               

            bool[] trueVector = new bool[dataSet.m_features.Count - 1];
            for (int i = 0; i < dataSet.m_features.Count - 1; i++)
            {
                trueVector[i] = true;
            }
            start = DateTime.Now;
            Tree fullC45Tree = new Tree(trueVector, dataSet);
            fullC45Tree.BuildTree(dataSet.m_trainInstances2,Tree.TreeType.C45, dataSet);
            time = DateTime.Now.Subtract(start).TotalSeconds;
            times += time.ToString() + " ";
            newPaths = new HashSet<Path>();
            GetAllRelevantPaths(fullC45Tree, fullC45Tree.attributesNames, dataSet, newPaths);
            var pathsWithValue = (newPaths).Select<Path, Path>(p => p.SetClassDistribution(dataSet));

            var C45Tree_UtilityValue = ModelEvaluation.EvaluateBaseModel(sDataPath, pathsWithValue.ToList(), dataSet, Tree.TreeType.C45);


            foreach(var pTos in pathToStateMapper)
            {
                if(pTos.Value.bestAction.isClassificationAction)
                {
                        if(pTos.Key.lastNode.classValue_String!=pTos.Value.bestAction.classification_string)
                        {
                            System.Console.WriteLine("BUG");
                        } 
                }
            }



            start = DateTime.Now;
            Tree MetaCostC45Tree = new Tree(trueVector, dataSet);
            MetaCostC45Tree.BuildTree(dataSet.m_trainInstances2, Tree.TreeType.MetaCost, dataSet);
            time = DateTime.Now.Subtract(start).TotalSeconds;
            times += time.ToString() + " ";
            newPaths = new HashSet<Path>();
            GetAllRelevantPaths(MetaCostC45Tree, MetaCostC45Tree.attributesNames, dataSet, newPaths);
            var MetaCostC45TreepathsWithValue = (newPaths).Select<Path, Path>(p => p.SetClassDistributionByDef(dataSet));
            var MetaCostC45_UtilityValue = ModelEvaluation.EvaluateBaseModel(sDataPath, MetaCostC45TreepathsWithValue.ToList(), dataSet, Tree.TreeType.MetaCost);

            start = DateTime.Now;
            Tree fullEG2Tree = new Tree(trueVector, dataSet);
            fullEG2Tree.BuildTree(dataSet.m_trainInstances2, Tree.TreeType.EG2, dataSet);
            time = DateTime.Now.Subtract(start).TotalSeconds;
            times += time.ToString() + " ";
            newPaths = new HashSet<Path>();
            GetAllRelevantPaths(fullEG2Tree, fullEG2Tree.attributesNames, dataSet, newPaths);
            var pathsWithValue2 = (newPaths).Select<Path, Path>(p => p.SetClassDistribution(dataSet));
            var EG2Tree_UtilityValue = ModelEvaluation.EvaluateBaseModel(sDataPath, pathsWithValue2.ToList(), dataSet, Tree.TreeType.EG2);

            start = DateTime.Now;
            Tree fullCSID3Tree = new Tree(trueVector, dataSet);
            fullCSID3Tree.BuildTree(dataSet.m_trainInstances2, Tree.TreeType.CSID3, dataSet);
            time = DateTime.Now.Subtract(start).TotalSeconds;
            times += time.ToString() + " ";
            newPaths = new HashSet<Path>();
            GetAllRelevantPaths(fullCSID3Tree, fullCSID3Tree.attributesNames, dataSet, newPaths);
            var pathsWithValue3 = (newPaths).Select<Path, Path>(p => p.SetClassDistribution(dataSet));
            var CSID3Tree_UtilityValue = ModelEvaluation.EvaluateBaseModel(sDataPath, pathsWithValue3.ToList(), dataSet, Tree.TreeType.CSID3);

            start = DateTime.Now;
            Tree fullIDXTree = new Tree(trueVector, dataSet);
            fullIDXTree.BuildTree(dataSet.m_trainInstances2, Tree.TreeType.IDX, dataSet);
            time = DateTime.Now.Subtract(start).TotalSeconds;
            times += time.ToString() + " ";
            newPaths = new HashSet<Path>();
            GetAllRelevantPaths(fullIDXTree, fullIDXTree.attributesNames, dataSet, newPaths);
            var pathsWithValue4 = (newPaths).Select<Path, Path>(p => p.SetClassDistribution(dataSet));
            var IDXTree_UtilityValue = ModelEvaluation.EvaluateBaseModel(sDataPath, pathsWithValue4.ToList(), dataSet, Tree.TreeType.IDX);


            Tree OrginalC45Tree = new Tree(trueVector, dataSet);
            OrginalC45Tree.BuildTree(dataSet.m_trainInstances2, Tree.TreeType.C45, dataSet);

            newPaths = new HashSet<Path>();
            GetAllRelevantPaths(OrginalC45Tree, OrginalC45Tree.attributesNames, dataSet, newPaths);
            var pathsWithValue5 = (newPaths).Select<Path, Path>(p => p.SetClassDistributionByDef(dataSet));
            var OrginalC45_UtilityValue = ModelEvaluation.EvaluateBaseModel(sDataPath, pathsWithValue5.ToList(), dataSet, Tree.TreeType.C45);

            start = DateTime.Now;
            Tree LowerCostTree = new Tree(trueVector, dataSet);
            LowerCostTree.BuildTree(dataSet.m_trainInstances2, Tree.TreeType.LowerCost, dataSet);
            time = DateTime.Now.Subtract(start).TotalSeconds;
            times += time.ToString() + " ";
            newPaths = new HashSet<Path>();
            GetAllRelevantPaths(LowerCostTree, LowerCostTree.attributesNames, dataSet, newPaths);
            var pathsWithValue6 = (newPaths).Select<Path, Path>(p => p.SetClassDistribution(dataSet));
            var LowerCost_UtilityValue = ModelEvaluation.EvaluateBaseModel(sDataPath, pathsWithValue6.ToList(), dataSet, Tree.TreeType.LowerCost);
     



            string results = "";
            results += cLeaves + "  ";
            results += utilityValue1.Value.ToString() + "   " + utilityValue1.Key.ToString() + "   ";
            results += utilityValue2.Value.ToString() + "   " + utilityValue2.Key.ToString() + "   ";
            results += IDXTree_UtilityValue.Value.ToString() + "   " + IDXTree_UtilityValue.Key.ToString() + "   ";
            results += CSID3Tree_UtilityValue.Value.ToString() + "   " + CSID3Tree_UtilityValue.Key.ToString() + "   ";
            results += EG2Tree_UtilityValue.Value.ToString() + "   " + EG2Tree_UtilityValue.Key.ToString() + "   ";
            results += C45Tree_UtilityValue.Value.ToString() + "   " + C45Tree_UtilityValue.Key.ToString() + "   ";
            results += OrginalC45_UtilityValue.Value.ToString() + "   " + OrginalC45_UtilityValue.Key.ToString() + "   ";
            results += MetaCostC45_UtilityValue.Value.ToString() + "   " + MetaCostC45_UtilityValue.Key.ToString() + "   ";







             string results2 = "";
             results2 += utilityValue1.Value.ToString() + "   " + utilityValue1.Key.ToString() + "   ";
             results2 += IDXTree_UtilityValue.Key.ToString() + "   " ;
             results2 += CSID3Tree_UtilityValue.Key.ToString() + "   ";
             results2 += EG2Tree_UtilityValue.Key.ToString() + "   ";
             results2 += C45Tree_UtilityValue.Key.ToString() + "   ";
             results2 += OrginalC45_UtilityValue.Key.ToString() + "   ";
             results2 += MetaCostC45_UtilityValue.Key.ToString() + "   ";

             StreamWriter resFile = new StreamWriter(sDataPath + "res.txt", true);
           // resFile.WriteLine(results2);
            resFile.WriteLine(MetaCostC45_UtilityValue.Value.ToString() + "   " + MetaCostC45_UtilityValue.Key.ToString());
            resFile.Close();
             //System.Console.WriteLine(results);


            StreamWriter allResults = new StreamWriter(sDataPath + "allResults.txt", true);
            allResults.WriteLine(costIndex + "\t" + times + "\t" + results);
            allResults.Close();


            /*Tree fullEG2v1Tree = new Tree(trueVector, dataSet);
            fullEG2v1Tree.BuildTree(dataSet.m_trainInstances2, Tree.TreeType.EG2_v1);
            newPaths = new HashSet<Path>();
            GetAllRelevantPaths(fullEG2v1Tree, fullEG2v1Tree.attributesNames, dataSet, newPaths);
            var pathsWithValue5 = (newPaths).Select<Path, Path>(p => p.SetClassDistribution(dataSet));
            var EG2v1Tree_UtilityValue = ModelEvaluation.EvaluateBaseModel(pathsWithValue5.ToList(), dataSet, Tree.TreeType.EG2_v1);

            Tree fullCSID3v1Tree = new Tree(trueVector, dataSet);
            fullCSID3v1Tree.BuildTree(dataSet.m_trainInstances2, Tree.TreeType.CSID3_v1);
            newPaths = new HashSet<Path>();
            GetAllRelevantPaths(fullCSID3v1Tree, fullCSID3v1Tree.attributesNames, dataSet, newPaths);
            var pathsWithValue6 = (newPaths).Select<Path, Path>(p => p.SetClassDistribution(dataSet));
            var CSID3v1Tre_UtilityValue = ModelEvaluation.EvaluateBaseModel(pathsWithValue6.ToList(), dataSet, Tree.TreeType.CSID3_v1);

            Tree fullIDXv1Tree = new Tree(trueVector, dataSet);
            fullIDXv1Tree.BuildTree(dataSet.m_trainInstances2, Tree.TreeType.IDX_v1);
            newPaths = new HashSet<Path>();
            GetAllRelevantPaths(fullIDXv1Tree, fullIDXv1Tree.attributesNames, dataSet, newPaths);
            var pathsWithValue7 = (newPaths).Select<Path, Path>(p => p.SetClassDistribution(dataSet));
            var IDXv1Tree_UtilityValue = ModelEvaluation.EvaluateBaseModel(pathsWithValue7.ToList(), dataSet, Tree.TreeType.IDX_v1);*/
        }


        public static void WriteMatrix(double reward, int row)
        {
            StreamWriter file = new StreamWriter(@"D:\Cost Sensitive Tree - Relevant Folder\Data\ICETfiles\ICETfiles\Costs\cm" + ((int)reward), false);
            for(int i =0; i < row; i++)
            {
                for (int j = 0; j < row; j++)
                {
                    if(i==j)
                    {
                        file.Write(-reward + " ");
                    }
                    else
                    {
                        file.Write(reward + " ");
                    }
                }
                file.WriteLine();
            }
            file.Close();
        }


        public static HashSet<Path> GetAllRelevantPaths(Tree tree, HashSet<string> attSet, DataReader.DataSet dataSet, HashSet<Path> uniquePath)
        {
            HashSet<Path> pathsInTree = new HashSet<Path>();
            foreach (Path path in tree.paths)
            {
                Path reducedPath = path.GetReducePath(attSet, attSet);
                if (reducedPath != null)
                {
                    pathsInTree.Add(reducedPath);
                    if (!uniquePath.Contains(reducedPath))
                    {
                        uniquePath.Add(reducedPath);
                    }
                }
            }

            return uniquePath;
        }












        public static Dictionary<Path, Dictionary<string, List<List<Path>>>> GetAllPairs(HashSet<Path> allPathsInOneTree)
        {
            System.Console.WriteLine();
            System.Console.WriteLine("Find Transition:");
            if (allPathsInOneTree == null || allPathsInOneTree.Count == 0)
                return null;
            List<Path> allPathsInTree = allPathsInOneTree.OrderBy(p => p.Count()).ToList();
            Dictionary<Path, Dictionary<string, List<List<Path>>>> allGroup = new Dictionary<Path, Dictionary<string, List<List<Path>>>>();
            for (int i = 0; i < allPathsInTree.Count; i++)
            {
                Path path1 = allPathsInTree.ElementAt(i);
                Dictionary<string, List<Path>> groupingByAttName = new Dictionary<string, List<Path>>();
                Dictionary<string, List<List<Path>>> groupingSetByAttName = new Dictionary<string, List<List<Path>>>();
                System.Console.Write("\r{0}%   ", 100 * Math.Round((double)(i + 1) / (double)allPathsInTree.Count, 3));
                var relevant = allPathsInTree;//.Where(v => v.Count() <= path1.Count()+1).ToList();
                for (int j = 0; j < relevant.Count; j++)
                {
                    Path path2 = relevant.ElementAt(j);
                    string diff = path1.CanPass(path2);
                    if (diff != null)
                    {
                        //TreeAndTest treeAndTest = new TreeAndTest { tree = path2.GetSourceTree(), attributeName = diff };
                        if (!groupingByAttName.ContainsKey(diff))
                        {
                            groupingByAttName.Add(diff, new List<Path>());
                        }
                        groupingByAttName[diff].Add(path2);
                    }
                }
                foreach (var item in groupingByAttName)
                {
                    groupingSetByAttName.Add(item.Key, GetFullSetsBySearch(item.Value, item.Key));
                }
                allGroup.Add(path1, groupingSetByAttName);
            }
            return allGroup;
        }


        public static int counter = 0;
        public static List<List<Path>> GetFullSetsBySearch(List<Path> paths, string attName)
        {
            counter++;
            if (counter == 10)
                System.Console.WriteLine("");
            List<List<Path>> allSets = new List<List<Path>>();
            List<DummyState> openList = new List<DummyState>();
            HashSet<DummyState> goalStates = new HashSet<DummyState>(new DummyState_EqualityComparer());
            bool stop = false;

            foreach (Path path in paths)
            {
                openList.Add(new DummyState(path, attName));
                //if (path.GetNodeList().Count(n=>n.formula.attributeName== attName)>1)
                //    System.Console.WriteLine("");
            }
            while (openList.Count > 0)
            {
                DummyState state = openList[0];
                openList.RemoveAt(0);
                foreach (Path path in paths)
                {

                    DummyState newState = state.ApplyAction(path, attName);
                    if (newState != null)
                    {
                        if (newState.IsCloseState())
                        {
                            goalStates.Add(newState);
                        }
                        else
                        {
                            openList.Add(newState);
                        }
                    }
                }

            }
            List<List<Path>> allSubSet = new List<List<Path>>();
            foreach (DummyState state in goalStates)
            {
                allSubSet.Add(state.paths);
            }

            return allSubSet;
        }

        public class DummyState
        {
            public List<List<Tree.Node>> openPaths;

            public List<Path> paths;
            public bool isFull;

            public DummyState(Path path, string attName)
            {
                openPaths = new List<List<Tree.Node>>();
                List<Tree.Node> openNode = new List<Tree.Node>();
                foreach (Tree.Node node in path)
                {
                    if (node.formula.attributeName == attName)
                        openNode.Add(node);
                }
                openPaths.Add(openNode);
                paths = new List<Path>();
                paths.Add(path);
                isFull = false;
            }

            public DummyState(DummyState s)
            {
                openPaths = new List<List<Tree.Node>>();
                foreach (var path in s.openPaths)
                {
                    List<Tree.Node> copyOfPath = new List<Tree.Node>(path);
                    openPaths.Add(copyOfPath);
                }
                paths = new List<Path>(s.paths);
                isFull = s.isFull;
            }

            public void AddOpenPath(List<Tree.Node> path, string attName)
            {
                List<Tree.Node> openNode = new List<Tree.Node>();
                foreach (Tree.Node node in path)
                {
                    if (node.formula.attributeName == attName)
                        openNode.Add(node);
                }
                openPaths.Add(openNode);
            }

            public override string ToString()
            {
                if (openPaths.Count == 0)
                    return "GoalState";
                else

                {
                    string str = "";
                    foreach (Tree.Node node in openPaths.Last())
                    {
                        str += node.ToString() + ",  ";
                    }
                    return str;
                }
            }

            public DummyState Clone()
            {
                return new DummyState(this);
            }

            public bool IsCloseState()
            {
                if (openPaths.Count == 0)
                    return true;

                return false;
            }
            public DummyState ApplyAction(Path path, string attName)
            {
                List<Tree.Node> reducedPath = path.GetNodeList().Where(n => n.formula.attributeName == attName).ToList();
                if (CanReduced(openPaths.Last(), reducedPath))
                {
                    DummyState newState = new DummyState(this);
                    newState.paths.Add(path);
                    //newState.openPaths[newState.openPaths.Count - 1].RemoveAt(newState.openPaths[newState.openPaths.Count - 1].Count - 1); 
                    if (reducedPath.Count() > openPaths.Last().Count)
                    {
                        newState.AddOpenPath(reducedPath, attName);
                    }
                    else
                    {
                        bool stop = false;
                        newState.openPaths[newState.openPaths.Count - 1].RemoveAt(newState.openPaths[newState.openPaths.Count - 1].Count - 1);
                        if (newState.openPaths[newState.openPaths.Count - 1].Count == 0)
                            newState.openPaths.RemoveAt(newState.openPaths.Count - 1);
                        while (!stop)
                        {
                            stop = true;
                            if (newState.openPaths.Count > 1)
                            {
                                if (newState.openPaths[newState.openPaths.Count - 1].Count == newState.openPaths[newState.openPaths.Count - 2].Count &&
                                    newState.openPaths[newState.openPaths.Count - 2].Last().formula.IsInverseSign(newState.openPaths[newState.openPaths.Count - 1].Last().formula))
                                {
                                    newState.openPaths.RemoveAt(newState.openPaths.Count - 1);
                                    newState.openPaths[newState.openPaths.Count - 1].RemoveAt(newState.openPaths[newState.openPaths.Count - 1].Count - 1);
                                    if (newState.openPaths[newState.openPaths.Count - 1].Count == 0)
                                        newState.openPaths.RemoveAt(newState.openPaths.Count - 1);
                                    stop = false;
                                }
                            }
                        }
                    }
                    return newState;
                }
                return null;
            }

            public bool CanReduced(List<Tree.Node> oldPath, List<Tree.Node> newPath)
            {
                if (oldPath.Count > newPath.Count)
                    return false;
                int i = 0;
                for (; i < oldPath.Count - 1; i++)
                {
                    if (!oldPath[i].Equals(newPath[i]))
                        return false;
                }
                if (!oldPath[i].formula.IsInverseSign(newPath[i].formula))
                    return false;

                return true;
            }

        }



        class List_Path_EqualityComparer : IEqualityComparer<List<Path>>
        {
            public bool Equals(List<Path> l1, List<Path> l2)
            {
                if (l1.Count == l2.Count)
                {
                    foreach (Path p1 in l1)
                    {
                        if (!l2.Contains(p1))
                            return false;
                    }
                    return true;
                }
                return false;
            }

            public int GetHashCode(List<Path> l1)
            {
                return l1.GetHashCode();
            }

        }

        class DummyState_EqualityComparer : IEqualityComparer<DummyState>
        {
            public bool Equals(DummyState l1, DummyState l2)
            {
                if (l1.paths.Count == l2.paths.Count)
                {
                    foreach (Path p1 in l1.paths)
                    {
                        if (!l2.paths.Contains(p1))
                            return false;
                    }
                    return true;
                }
                return false;
            }

            public int GetHashCode(DummyState l1)
            {
                return 4;
                int x = 0;
                foreach (Path p in l1.paths)
                    x += p.GetHashCode();
                return x;
            }

        }


        public struct TreeAndTest
        {
            public Tree tree;
            public string attributeName;



            public override string ToString()
            {
                return "Test: " + attributeName + " |  Tree: " + tree.ToString();
            }
        }

        public static List<bool[]> GetAllPermutations(int orginalSize, int size, DataReader.DataSet dataSet, out HashSet<string> initialyKnow)
        {
            if (size == -1)
                size = orginalSize;
            List<int> orderIndex = dataSet.m_features.Where(x => x.Key != "Class").OrderBy(x => x.Value.cost).Select(x => x.Value.attIndex).ToList();
            //size = 5;
            List<int> relevantIndex = orderIndex.GetRange(0, size);
            initialyKnow = new HashSet<string>(dataSet.m_features.Where(f => !relevantIndex.Contains(f.Value.attIndex) && f.Value.name != "Class").Select(kv => kv.Value.name));
            List<bool[]> allPermutations = new List<bool[]>();
            bool[] array = new bool[size];
            allPermutations.Add((bool[])array.Clone());
            bool stop = false;
            while (!stop)
            {
                stop = true;
                for (int i = (size - 1); i > -1; i--)
                {
                    if (array[i] == false)
                    {
                        array[i] = true;
                        stop = false;
                        allPermutations.Add((bool[])array.Clone());
                        break;
                    }
                    else
                    {
                        array[i] = false;
                    }
                }
            }

            List<bool[]> allFullPermutations = new List<bool[]>();
            foreach (bool[] permutation in allPermutations)
            {
                bool[] vector = new bool[orginalSize];
                for (int j = 0; j < orginalSize; j++)
                {
                    vector[j] = true;
                }
                for (int i = 0; i < permutation.Length; i++)
                {
                    vector[relevantIndex[i]] = permutation[i];
                }
                allFullPermutations.Add(vector);
            }

            return allFullPermutations;
        }


        static void GetClassification(string path, int outputName)
        {
            string courentPath = Directory.GetCurrentDirectory();
            weka.classifiers.misc.SerializedClassifier classifier = new weka.classifiers.misc.SerializedClassifier();
            classifier.setModelFile(new java.io.File(@"classifier/weka-neural-network"));

            List<string> records = new List<string>();
            List<string> tmp = new List<string>();
            StreamReader wr = new StreamReader(@"PotatoVectors.txt");
            while (!wr.EndOfStream)
            {
                tmp.Add(wr.ReadLine());
            }
            Random randomGenerator = new Random();
            while (tmp.Count > 0)
            {
                int rand = randomGenerator.Next(0, tmp.Count);
                records.Add(tmp[0]);
                tmp.RemoveAt(0);

            }

            FastVector atts;
            atts = new FastVector();
            string line = "Avg	Var	Perimeter	AreaSize	AvgBlue	AvgGreen	AvgRed	NumOfStains	xVAR	yVAR	TEXTURE_AVG	TEXTURE_VAR	Dist1	Dist2	Dist3	Dist4	Dist5	Dist6	Dist7	Dist8	Dist9	Dist10	Dist11	Dist12	Dist13	Dist14	Dist15	Dist16	Dist17	Dist18	Dist19	Dist20	Dist21	Dist22	Dist23	Dist24	Dist25	Dist26";
            //string line = attributeFile.ReadLine();
            string[] attributes = line.Split('\t');
            foreach (string att in attributes)
            {
                atts.addElement(new weka.core.Attribute(att));
            }
            FastVector classt = new FastVector();
            classt.addElement("Kolt");
            classt.addElement("Rez");
            classt.addElement("KK");
            classt.addElement("Gerev");
            classt.addElement("Healthy");

            atts.addElement(new weka.core.Attribute("class", classt));

            Instances data = new Instances("PotatoClassification", atts, 0);
            double[] vals;
            foreach (string str in records)
            {
                char[] delimeter = { ' ', '\t', '\"' };
                string[] tmpRec = str.Split(delimeter);
                List<string> record = new List<string>();
                bool first = false;
                foreach (string filed in tmpRec)
                {

                    if (!filed.Equals(""))
                    {
                        record.Add(filed);
                    }


                }
                vals = new double[data.numAttributes()];
                int i = 0;
                foreach (string filed in record)
                {
                    vals[i] = Convert.ToDouble(filed);
                }
                data.add(new Instance(1.0, vals));

            }

            try
            {
                StreamWriter fs = new StreamWriter(path + "//result" + outputName + ".txt", false);
                for (int i = 0; i < data.numInstances(); i++)
                {
                    weka.core.Instance currentInst = data.instance(i);
                    double predictedClass = classifier.classifyInstance(currentInst);
                    fs.WriteLine(classt.elementAt((int)predictedClass).ToString());
                    fs.Flush();
                }
                fs.Close();
            }
            catch (java.lang.Exception ex)
            {
                System.Console.Write(ex.toString());
            }

        }

    }
}
