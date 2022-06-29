using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatasetGeneration
{
    //not working
    public class HalfSpaces : GenerationModel
    {
        public Dictionary<int, double> AttributeBoundaries;
        public Dictionary<int, Dictionary<int, bool>> ClassBoundaries;
        public int AttributesPerClass;

        public HalfSpaces(int cAttributesPerClass, int cClasses, int cAttributes, int cValues, int iRandomSeed = 0) : base(cClasses, cAttributes, cValues, -1, iRandomSeed)
        {
            AttributesPerClass = cAttributesPerClass;
            AttributeBoundaries = new Dictionary<int, double>();
            ClassBoundaries = new Dictionary<int, Dictionary<int, bool>>();

            bool bAbove = true;
            for (int i = 0; i < Classes; i++)
            {
                AttributeBoundaries[i] = Rnd.Next(Values - 2) + 1;
                ClassBoundaries[i] = new Dictionary<int, bool>();
                bAbove = Rnd.NextDouble() < 0.5;
                ClassBoundaries[i][i] = bAbove;
                for (int j = 0; j < cAttributesPerClass - 1; j++)
                {
                    int iAttribute = Rnd.Next(Attributes);
                    while (iAttribute == i)
                        iAttribute = Rnd.Next(Attributes);
                    bAbove = Rnd.NextDouble() < 0.5;
                    ClassBoundaries[i][iAttribute] = bAbove;

                }
            }
        }

        public override bool Classify(Example e)
        {
            throw new NotImplementedException();
        }

        public override Example Sample()
        {
            throw new NotImplementedException();
        }
    }
}
