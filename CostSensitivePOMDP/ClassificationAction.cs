using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CostSensitivePOMDP
{
    class ClassificationAction : Action
    {
        public int Class { get; private set; }
        public ClassificationAction(int c)
        {
            Class = c;
        }
        public override string ToString()
        {
            return "Classify " + Class;
        }
    }
}
