using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatasetGeneration
{
    public class Example
    {
        public Dictionary<int, int> Attributes;
        public int Class;
        //public DecisionTree.Node Source;
        public int ID;

        public static int Examples = 0;

        public Example()
        {
            ID = Examples++;
            Attributes = new Dictionary<int, int>();
        }

        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < Attributes.Count; i++)
                s += "a" + i + "=" + Attributes[i] + ", ";
            s += "c" + Class;
            return s;
        }

        public void Write(StreamWriter sw)
        {
            for(int i = 0; i < Attributes.Count; i++)
            {
                sw.Write(Attributes[i] + "\t");
            }
            sw.WriteLine((Class + 1));
        }
    }
}
