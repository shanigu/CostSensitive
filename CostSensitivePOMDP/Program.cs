using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CostSensitivePOMDP
{
    class Program
    {
        static void TestDomain(string sDomain, int iCM, int iSplit)
        {
            
            string sPath = @"G:\Data\CostSensitive\" + sDomain + @"\";
            StreamWriter sw = new StreamWriter(sPath + "Results.txt", true);
            sw.WriteLine();
            sw.WriteLine(sDomain + "," + iCM + "," + iSplit + ", " + DateTime.Now);
            Console.WriteLine(sDomain + "," + iCM + "," + iSplit + ", " + DateTime.Now);
            sw.Close();
            //try
            {
                //bugbug; //Need to measure after each iteration of BFS the current ADR and save, so that we can produce graphs
                DateTime dtStart = DateTime.Now;
                POMDP p = new POMDP();
                p.ConcurrentBackup = false;
                //p.LoadExamples(@"G:\Data\CostSensitive\breast\data1.small.rr");
                p.LoadTestCosts(@"G:\Data\CostSensitive\" + sDomain + @"\data.names");
                p.LoadExamples(@"G:\Data\CostSensitive\" + sDomain + @"\data" + iSplit + ".rr");
                p.LoadClassificationCosts(@"G:\Data\CostSensitive\" + sDomain + @"\CM." + iCM);
                //List<SimpleAlphaVector> lValueFunction = p.DecisionTreePointBasedValueIteration(50, 20);
                HashSet<SimpleBelief> lBeliefs = new HashSet<SimpleBelief>();
                HashSet<SimpleAlphaVector> lValueFunction = new HashSet<SimpleAlphaVector>();
                //p.ReadMDPPolicy(@"C:\Users\shanigu\Downloads\Data\output\" + "TrainClassification." + sDomain + ".CM" + iCM + ".txt");
                lValueFunction = p.BFSPointBasedValueIteration(20, 10, lValueFunction, lBeliefs, sPath, "data" + iSplit + ".ss");
                //HashSet<SimpleAlphaVector> lValueFunction2 = p.MDPPointBasedValueIteration(@"C:\Users\shanigu\Downloads\Data\output\" + "TrainClassification." + sDomain + ".CM" + iCM + ".txt", lValueFunction);
                //HashSet<SimpleAlphaVector> lValueFunction2 = p.Perseus(50, 20, lValueFunction, lBeliefs);
                DateTime dtEnd = DateTime.Now;
                p.TestPolicy(sPath, "data" + iSplit + ".ss", lValueFunction);
                Console.WriteLine("Time: " + (dtEnd - dtStart).TotalSeconds);
                sw = new StreamWriter(sPath + "Results.txt", true);
                sw.WriteLine();
                sw.WriteLine(sDomain + "," + iCM + "," + iSplit + ", T=" + "Time: " + (dtEnd - dtStart).TotalSeconds);
                sw.Close();
            }
            /*
            catch (Exception e)
            {
                sw = new StreamWriter(sPath + "Results.txt", true);
                sw.WriteLine();
                sw.WriteLine(sDomain + "," + iCM + ", e=" + e.Message);
                sw.Close();
            }
            */
        }
        public static void Main(string[] args)
        {
            //TestDomain("car", 3, 1);
            //..TestDomain("glass", 3);
            for (int i = 3; i <= 3; i++ )
                //for(int j = 1; j <= 10; j++)
                    TestDomain("car", i, 1);

            /*
            for (int i = 1; i <= 3; i++)
                TestDomain("iris", i);
            for (int i = 1; i <= 3; i++)
                TestDomain("car", i);
            for (int i = 1; i <= 3; i++)
                TestDomain("tictactoe", i);
                */
            //TestDomain("iris", 2);

        }
    }
}
