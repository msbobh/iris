// See https://aka.ms/new-console-template for more informatio
using Accord.MachineLearning.VectorMachines;
using Accord.Statistics.Kernels;
using Accord.Statistics.Analysis;

using _AccordData;
using Accord.Math.Geometry;
using System;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.DataSets;
using Accord.MachineLearning;
using Accord.Math.Optimization.Losses;
using Accord.Math;

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

// In this example, we will learn a multi-class SVM using the one-vs - one(OvO)
// approach. The OvO approacbh can decompose decision problems involving multiple 
// classes into a series of binary ones, which can then be solved using SVMs.

// Ensure we have reproducible results
Accord.Math.Random.Generator.Seed = 0;

// We will try to learn a classifier
// for the Fisher Iris Flower dataset
// inputs and outputs defined above

// We will use mini-batches of size 32 to learn a SVM using SGD
var batches = MiniBatches.Create(batchSize: 32, maxIterations: 1000,
   shuffle: ShuffleMethod.EveryEpoch, input: inputs, output: outputs);

// Now, we can create a multi-class teaching algorithm for the SVMs
 var teachermsv = new MulticlassSupportVectorLearning<Linear, double[]>
{
    // We will use SGD to learn each of the binary problems in the multi-class problem
    Learner = (p) => new AveragedStochasticGradientDescent<Linear, double[], LogisticLoss>()
    {
        LearningRate = 1e-3,
        MaxIterations = 1 // so the gradient is only updated once after each mini-batch
    }
};

// The following line is only needed to ensure reproducible results. Please remove it to enable full parallelization
teacher.ParallelOptions.MaxDegreeOfParallelism = 1; // (Remove, comment, or change this line to enable full parallelism)

// Now, we can start training the model on mini-batches:
foreach (var batch in batches)
{
    teachermsv.Learn(batch.Inputs, batch.Outputs);
}

// Get the final model:
var svm = teachermsv.Model;

// Now, we should be able to use the model to predict 
// the classes of all flowers in Fisher's Iris dataset:
int[] prediction = svm.Decide(inputs);

// And from those predictions, we can compute the model accuracy:
 cm = new GeneralConfusionMatrix(expected: outputs, predicted: prediction);
 accuracy = cm.Accuracy; // should be approximately 0.973
Console.WriteLine("SVM model using o-vs-o using Average Stochastic Gradient Descent to learn the model");
Console.WriteLine("Accuracy = {0:p2}%, Precision = {1:p2}, Recall = {2:p2}", accuracy, cm.Precision[1], cm.Recall[2]);

var mn = new MNISTData();
Console.WriteLine("MNIST Data created");

var breast = new BreastData();
Console.WriteLine("Breast Cancer Data created");
var breastInputs = breast.iData.Select(arr => arr.Select(i => (double)i.GetValueOrDefault()).ToArray()).ToArray();
int[]? breastOutputs = breast.classLabels;
double[]? breastLabels = breastOutputs.Select(i => (double)i).ToArray();

Console.WriteLine("create the sequential minimal optimization teacher with a Gaussian Kernel");
// Now, we can create the sequential minimal optimization teacher
var learn = new SequentialMinimalOptimization<Gaussian>()
{
    UseComplexityHeuristic = true,
    UseKernelEstimation = true
};

// And then we can obtain a trained SVM by calling its Learn method
SupportVectorMachine<Gaussian> svm4 = learn.Learn(breastInputs, breastOutputs);

// Finally, we can obtain the decisions predicted by the machine:
GeneralConfusionMatrix _br2 = GeneralConfusionMatrix.Estimate(svm4, breastInputs, breastOutputs);
Console.ForegroundColor = ConsoleColor.Red;
Console.WriteLine("Accuracy = {0:p2}%", _br2.Accuracy);
Console.ResetColor();





