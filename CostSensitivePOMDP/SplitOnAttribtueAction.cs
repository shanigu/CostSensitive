using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CostSensitivePOMDP
{
    class SplitOnAttribtueAction : Action
    {
        public int Attribute { get; private set; }
        public double Split { get; private set; }

        public SplitOnAttribtueAction(int a, double v)
        {
            Attribute = a;
            Split = v;
        }
        public override string ToString()
        {
            return "Split " + Attribute + " >= " + Split ;
        }
        public override bool Equals(object obj)
        {
            if(obj is SplitOnAttribtueAction)
            {
                SplitOnAttribtueAction a = (SplitOnAttribtueAction)obj;
                if (a.Attribute == Attribute && a.Split == Split)
                    return true;
                return false;
            }
            return false;
        }
    }
}
