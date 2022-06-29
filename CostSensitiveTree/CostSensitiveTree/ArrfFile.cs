using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using weka.core;
using weka.core.converters;

namespace CostSensitiveTree
{

    public class CSV2Arff
    {
        /**
         * takes 2 arguments:
         * - CSV input file
         * - ARFF output file
         */
        public static void CreteArffFile(string csvFileName, string arffFileName )
        {
            // load CSV
            weka.core.converters.CSVLoader loader = new CSVLoader();
        loader.setSource(new java.io.File(csvFileName));
        Instances data = loader.getDataSet();

        // save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
    saver.setFile(new java.io.File(arffFileName));
    saver.setDestination(new java.io.File(arffFileName));
    saver.writeBatch();
  }
}
}
