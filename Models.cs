using System;
using Accord.MachineLearning.VectorMachines;
using Accord.Statistics.Kernels;
using Accord.Statistics.Analysis;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math;
using Accord.Statistics.Models.Fields.Features;
using System.Reflection.Emit;
using System.Security.Cryptography.X509Certificates;
using Accord.DataSets;
using Accord.MachineLearning;
using Accord.Math.Optimization.Losses;
using Accord.Math.Optimization;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Distributions.Fitting;
using System.Diagnostics;

namespace Models;
public class MultiClassSVMGaussian
{
    public GeneralConfusionMatrix? GCM { get; set; }
    public double Accuracy { get; }
    public double? Precision { get; }
    public TimeSpan Elapsed { get; }
    public string RunTime { get; } = string.Empty;

    public MultiClassSVMGaussian(in double[][] _Features, in int[] _Labels)
    {
        System.Diagnostics.Stopwatch aTimer = new System.Diagnostics.Stopwatch();
        aTimer.Start();
        var teacher = new MulticlassSupportVectorLearning<Gaussian>()
        {
            Learner = (param) => new SequentialMinimalOptimization<Gaussian>()
            {
                Kernel = new Gaussian()
            }
        };
        MulticlassSupportVectorMachine<Gaussian> ovo = teacher.Learn(_Features, _Labels);
        GeneralConfusionMatrix cm = GeneralConfusionMatrix.Estimate(ovo, _Features, _Labels);
        aTimer.Stop();
        Accuracy = cm.Accuracy;
        Precision = cm.Precision[0];
        Elapsed = aTimer.Elapsed;
        RunTime = aTimer.Elapsed.ToString(@"hh\:mm\:ss\.fff");


    }
}

public class MultiClassSVMLinear
{
    public GeneralConfusionMatrix? _CM { get; set; }
    public double Accuracy { get; set; }
    public double? Precision { get; set; }
    public string RunTime { get; } = string.Empty;  

    public MultiClassSVMLinear(in double[][] _Features, in int[] _Labels)
    {
        System.Diagnostics.Stopwatch aTimer = new System.Diagnostics.Stopwatch();
        var teacher = new MulticlassSupportVectorLearning<Linear>()
        {
            Learner = (param) => new SequentialMinimalOptimization<Linear>()
            {
                Complexity = 100
            }
        };
        MulticlassSupportVectorMachine<Linear> ovo = teacher.Learn(_Features, _Labels);
        GeneralConfusionMatrix cm = GeneralConfusionMatrix.Estimate(ovo, _Features, _Labels);
        double accuracy = cm.Accuracy;
        double[] _precision = cm.Precision;
        Precision = _precision[0];
        RunTime = "(h:m:s:ms)" + aTimer.Elapsed.ToString(@"hh\:mm\:ss\.fff");
    }
}

public class MulticlassSVM_PolynomialKernel
{
    // In this example, we will show how its possible to learn a 
    // non-linear SVM using a linear algorithm by using a explicit
    // expansion of the kernel function:
    public GeneralConfusionMatrix? _CM { get; }
    public double[] Precision { get; }
    public double Accuracy { get; }
    public System.TimeSpan Elapsed { get; }

    public string RunTime { get; } = string.Empty;
    public MulticlassSVM_PolynomialKernel(in double[][] _Features, in int[] _Labels)
    {
        double[][] Inputs = _Features;
        int[] Outputs = _Labels;
        // Ensure we have reproducible results
        Accord.Math.Random.Generator.Seed = 0;
        System.Diagnostics.Stopwatch aTimer = new System.Diagnostics.Stopwatch();

        NormalOptions options = new NormalOptions()
        {
            Regularization = 1e-5
        };

        // We will use mini-batches of size 32 to learn a SVM using SGD
        var batches = MiniBatches.Create(batchSize: 256, maxIterations: 1000,
           shuffle: ShuffleMethod.EveryEpoch, input: Inputs, output: Outputs);

        // We will use an explicit Polynomial kernel expansion
        var polynomial = new Polynomial(2);

        // Now, we can create a multi-class teaching algorithm for the SVMs
        var teacher = new MulticlassSupportVectorLearning<Linear, double[]>
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
        aTimer.Start();

        //var svm = teacher.Learn(polynomial.Transform(Inputs), Outputs);
        // Now, we can start training the model on mini-batches:
        foreach (var batch in batches)
        {
            teacher.Learn(polynomial.Transform(batch.Inputs), batch.Outputs);
        }

        // Get the final model:
        var svm = teacher.Model;

        // The following line is only needed to ensure reproducible results. Please remove it to enable full parallelization
        svm.ParallelOptions.MaxDegreeOfParallelism = 1; // (Remove, comment, or change this line to enable full parallelism)
        aTimer.Stop();
        // Now, we should be able to use the model to predict 
        // the classes of all flowers in Fisher's Iris dataset:
        int[] prediction = svm.Decide(polynomial.Transform(Inputs));

        // And from those predictions, we can compute the model accuracy:
         _CM = new GeneralConfusionMatrix(expected: Outputs, predicted: prediction);
        Accuracy = _CM.Accuracy; // should be approximately 0.92
        Precision = _CM.Precision;
        //Console.WriteLine("Elapsed time: {0}", aTimer.Elapsed.ToString(@"hh\:mm\:ss.fff"));
        RunTime = "(h:m:s:ms)" + aTimer.Elapsed.ToString(@"hh\:mm\:ss\.fff");
        TimeSpan Elapsed = aTimer.Elapsed;
    }

}

public class MultiNomialLogisticRegressionLBFGS
{
    public double Accuracy { get; }
    public double Precision { get; }
    public TimeSpan Elapsed { get; }
    public string RunTime { get; } = string.Empty;
    public MultiNomialLogisticRegressionLBFGS(in double[][] _Features, in int[] _Labels)
    {
        System.Diagnostics.Stopwatch aTimer = new System.Diagnostics.Stopwatch();
        // Create a Conjugate Gradient algorithm to estimate the regression
        var _mlbfgs = new MultinomialLogisticLearning<BroydenFletcherGoldfarbShanno>();

        // Now, we can estimate our model using BFGS
        aTimer.Start();
        MultinomialLogisticRegression _mlr = _mlbfgs.Learn(_Features, _Labels);
        aTimer.Stop();
        // We can compute the model answers
        int[] _Answers = _mlr.Decide(_Features);
       var _CM = new GeneralConfusionMatrix(expected: _Labels, predicted: _Answers);
       GeneralConfusionMatrix _cm = GeneralConfusionMatrix.Estimate(_mlr, _Features, _Labels);

        // And also the probability of each of the answers
        double[][] probabilities = _mlr.Probabilities(_Features);

        // Now we can check how good our model is at predicting
        double error = new ZeroOneLoss(_Labels).Loss(_Answers);
        Accuracy = _CM.Accuracy;
        Precision = _CM.Precision[0];
        Elapsed = aTimer.Elapsed;
        RunTime = "(h:m:s:ms)" + aTimer.Elapsed.ToString(@"hh\:mm\:ss\.fff");   
    }
}

public class MultiNomialPolyKernel_no_batches
{
    public double Accuracy { get; }
    public double Precision { get; }
    public TimeSpan Elapsed { get; }
    public string RunTime { get; } = string.Empty;
    public MultiNomialPolyKernel_no_batches(in double[][] _Features, in int[] _Labels)
    {
        System.Diagnostics.Stopwatch aTimer = new System.Diagnostics.Stopwatch();
        var teacher = new MulticlassSupportVectorLearning<Linear, double[]>
        {
            // We will use SGD to learn each of the binary problems in the multi-class problem
            Learner = (p) => new AveragedStochasticGradientDescent<Linear, double[], LogisticLoss>()
            {
                LearningRate = 1e-3,
                MaxIterations = 1 // so the gradient is only updated once after each mini-batch
            }
        };

        aTimer.Start();
        var svmPoly = teacher.Learn(_Features, _Labels);

        aTimer.Stop();
        // We can compute the model answers
        int[] _Answers = svmPoly.Decide(_Features);
        var _CM = new GeneralConfusionMatrix(expected: _Labels, predicted: _Answers);
        // And also the probability of each of the answers
        double[][] probabilities = svmPoly.Probabilities(_Features);
        // Now we can check how good our model is at predicting
        double error = new ZeroOneLoss(_Labels).Loss(_Answers);
        Accuracy = _CM.Accuracy;
        Precision = _CM.Precision[0];
        Elapsed = aTimer.Elapsed;
        RunTime = "(h:m:s:ms)" + aTimer.Elapsed.ToString(@"hh\:mm\:ss\.fff");

    }
}




