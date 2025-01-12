// See https://aka.ms/new-console-template for more informatio
using Accord.MachineLearning.VectorMachines;
using Accord.Statistics.Kernels;
using Accord.Statistics.Analysis;

using _AccordData;
using Accord.Math.Geometry;
using System;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.DataSets;

Console.WriteLine("Getting Iris Data");
var test = new irisData();
double[][] inputs = test.iData;
int[] outputs = test.classLabels;


var teacher = new MulticlassSupportVectorLearning<Linear>()
{
    Learner = (param) => new SequentialMinimalOptimization<Linear>()
    {
        // If you would like to use other kernels, simply replace
        // the generic parameter to the desired kernel class, such
        // as for example, Polynomial or Gaussian:

        Kernel = new Linear() // use the Linear kernel
    }

};


// Estimate the multi-class support vector machine using one-vs-one method
MulticlassSupportVectorMachine<Linear> ovo = teacher.Learn(test.iData, test.classLabels);

// Compute classification error
GeneralConfusionMatrix cm = GeneralConfusionMatrix.Estimate(ovo, test.iData, test.classLabels);

double error = cm.Error;         // should be 0.066666666666666652
double accuracy = cm.Accuracy;   // should be 0.93333333333333335
double kappa = cm.Kappa;         // should be 0.9
double chiSquare = cm.ChiSquare; // should be 248.52216748768473
Console.WriteLine("Multi Class SVM uisng SMO training algorithm");
Console.WriteLine("Error = {0:p2}, Accuracy = {1:p4}, kappa =  {2:F2}, chiSquare {3:F4}\n", error, accuracy, kappa, chiSquare);

teacher = new MulticlassSupportVectorLearning<Linear>()
{
    // using LIBLINEAR's L2-loss SVC dual for each SVM
    Learner = (p) => new LinearDualCoordinateDescent()
    {
        Loss = Loss.L2
    }
};
MulticlassSupportVectorMachine<Linear> ovo2 = teacher.Learn(inputs,outputs);
cm = GeneralConfusionMatrix.Estimate(ovo2, inputs, outputs);

error = cm.Error;         // should be 0.066666666666666652
accuracy = cm.Accuracy;   // should be 0.93333333333333335
kappa = cm.Kappa;         // should be 0.9
 chiSquare = cm.ChiSquare; // should be 248.52216748768473
Console.WriteLine("Multi Class SVM uisng Linear Coordinat Descent algorithm ");
Console.WriteLine("Error = {0:p2}, Accuracy = {1:p4}, kappa =  {2:F2}, chiSquare {3:F4}", error, accuracy, kappa, chiSquare);