using Accord.MachineLearning.Bayes;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Analysis;
using Accord.Statistics.Distributions.Fitting;
using Accord.Statistics.Distributions.Univariate;
using System;

public class NaiveBayes
{
    public GeneralConfusionMatrix? GCM { get; set; }
    public double Accuracy { get; }
    public double? Precision { get; }
    public TimeSpan Elapsed { get; }
    public string RunTime { get; } = string.Empty;
    public NaiveBayes(in double[][] _Features, in int[] _Labels)
	{

        // Create a new Gaussian distribution naive Bayes learner
        var teacher = new NaiveBayesLearning<NormalDistribution>();

        // Set options for the component distributions
        teacher.Options.InnerOption = new NormalOptions
        {
            Regularization = 1e-5 // to avoid zero variances
        };

        // Learn the naive Bayes model
        NaiveBayes<NormalDistribution> bayes = teacher.Learn(_Features, _Lables);

        // Use the model to predict class labels
        int[] predicted = bayes.Decide(_Features);

        // Estimate the model error. The error should be zero:
        double error = new ZeroOneLoss(_Labels).Loss(predicted);

        // Create a confusion matrix to analyze the results



    }
}
