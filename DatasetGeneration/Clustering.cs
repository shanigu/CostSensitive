using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatasetGeneration
{
    class Clustering : GenerationModel
    {
        public List<int[]> CentroidValues;
        public double[] CentroidDistribution;
        public int[] MapCentroidToClass;
        public int[] MapCentroidToWrappingClass;
        public int Centroids;
        public bool Wrapping;


        public Clustering(int cClasses, int cCentroids, int cAttributes, int cValues, bool bWrapping, int iDifficultyLevel = 0, int iRandomSeed = 0) : base(cClasses, cAttributes, cValues, -1, iRandomSeed)
        {
            Centroids = cCentroids;
            CentroidValues = new List<int[]>();
            CentroidDistribution = new double[Centroids];
            Wrapping = bWrapping;

            double dMass = 1.0;

            for (int iCentroid = 0; iCentroid < cCentroids; iCentroid++)
            {
                CentroidDistribution[iCentroid] = Rnd.NextDouble() * dMass * 0.5;
                dMass = dMass - CentroidDistribution[iCentroid];
            }
            CentroidDistribution[Classes - 1] += dMass;
            if (iDifficultyLevel == 0)
                SelectRandomCentroids();
            else
                SelectCentroidsDifficultyLevel(iDifficultyLevel);

            MapCentroidToClass = new int[cCentroids];
            MapCentroidToWrappingClass = new int[cCentroids];
            for (int iCentroid = 0; iCentroid < Centroids; iCentroid++)
            {
                if (iCentroid < Classes)
                    MapCentroidToClass[iCentroid] = iCentroid;
                else
                    MapCentroidToClass[iCentroid] = Rnd.Next(Classes);
                if(bWrapping)
                    MapCentroidToWrappingClass[iCentroid] = Rnd.Next(Classes);
            }
        }

        public void SelectCentroidsDifficultyLevel(int iDifficultyLevel)
        {
            for (int iCentroid = 0; iCentroid < Centroids; iCentroid++)
            {
                List<int> lAttributes = new List<int>();
                for (int iAttribute = 0; iAttribute < Attributes; iAttribute++)
                {
                    lAttributes.Add(iAttribute);
                }
                while(lAttributes.Count > iDifficultyLevel)
                {
                    lAttributes.RemoveAt(Rnd.Next(lAttributes.Count));
                }
                int[] aCentroid = new int[Attributes];
                for (int iAttribute = 0; iAttribute < Attributes; iAttribute++)
                {
                    if (lAttributes.Contains(iAttribute))
                        aCentroid[iAttribute] = Rnd.Next(Values);
                    else
                        aCentroid[iAttribute] = -1;
                }
                CentroidValues.Add(aCentroid);
            }
        }

        public void SelectRandomCentroids()
        {
            for (int iCentroid = 0; iCentroid < Centroids; iCentroid++)
            {
                int[] aCentroid = new int[Attributes];
                for (int iAttribute = 0; iAttribute < Attributes; iAttribute++)
                {
                    aCentroid[iAttribute] = Rnd.Next(Values);
                }
                CentroidValues.Add(aCentroid);
            }
        }

        public override bool Classify(Example e)
        {
            double dMinDistance = double.PositiveInfinity;
            int iBestCentroid = -1;
            for (int iCentroid = 0; iCentroid < Centroids; iCentroid++)
            {
                double d = distance(e, iCentroid);
                if(d < dMinDistance)
                {
                    dMinDistance = d;
                    iBestCentroid = iCentroid;
                }
            }
            return MapCentroidToClass[iBestCentroid] == e.Class;
        }

        public double distance(Example e, int iClass)
        {
            double d = 0.0;
            for (int iAttribute = 0; iAttribute < Attributes; iAttribute++)
            {
                if(CentroidValues[iClass][iAttribute] != -1)
                    d += Math.Pow(e.Attributes[iAttribute] - CentroidValues[iClass][iAttribute], 2);
            }
            d = Math.Sqrt(d);
            return d;
        }


        public Example SampleInternal(int iCentroid)
        {
            Example e = new Example();
            e.Class = MapCentroidToClass[iCentroid];

            for (int iAttribute = 0; iAttribute < Attributes; iAttribute++)
            {
                if (CentroidValues[iCentroid][iAttribute] != -1)
                {
                    int dStdDev = (4 - AttributeUsefulness[iAttribute]);
                    int iValue = Normal(Rnd, CentroidValues[iCentroid][iAttribute], dStdDev);
                    if (iValue >= Values)
                        iValue = Values;
                    if (iValue < 0)
                        iValue = 0;
                    e.Attributes[iAttribute] = iValue;
                }
                else
                    e.Attributes[iAttribute] = Rnd.Next(Values);
            }
            return e;
        }

        public Example SampleExternal(int iCentroid)
        {
            Example e = new Example();
            e.Class = MapCentroidToWrappingClass[iCentroid];

            for (int iAttribute = 0; iAttribute < Attributes; iAttribute++)
            {
                //int dStdDev = (4 - AttributeUsefulness[iAttribute]);
                int dStdDev = 1;
                int iMean = CentroidValues[iCentroid][iAttribute] + AttributeUsefulness[iAttribute] + 1;
                if(Rnd.NextDouble() < 0.5 || iMean >= Values)
                    iMean = CentroidValues[iCentroid][iAttribute] - AttributeUsefulness[iAttribute] - 1;
                int iValue = Normal(Rnd, iMean, dStdDev);
                if (iValue >= Values)
                    iValue = Values;
                if (iValue < 0)
                    iValue = 0;
                e.Attributes[iAttribute] = iValue;
            }
            return e;
        }

        public override Example Sample()
        {
            double dProb = Rnd.NextDouble();
            int iCentroid = -1;
            while (dProb > 0)
            {
                iCentroid++;
                dProb -= CentroidDistribution[iCentroid];
            }
            Example e = SampleInternal(iCentroid);
            if (Wrapping && Rnd.Next() < 0.5)
                e = SampleExternal(iCentroid);

            return e;
        }
    }
}
