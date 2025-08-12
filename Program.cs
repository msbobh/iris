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
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression;
using Accord.Math.Optimization;
using Models;
using Accord.IO;
using Accord.Statistics.Models.Fields.Features;
using System.Reflection.Emit;
using System.Diagnostics;

// Load up 3 Different data sets
//
var _IrisData = new irisData();
var _Winedata = new WineData();
var _BreastData = new BreastData();

// MNIST Data
// Try the csv version of MNIST data
string filename = Path.Combine(Directory.GetCurrentDirectory(), "mnist_test.csv");
double[][] MNISTCSV_data;
using (CsvReader reader = new CsvReader(filename, hasHeaders: true))
{
    // Read the data into a 2D array
    MNISTCSV_data = reader.ToJagged<double>();

}

int[] MNISTLabels = MNISTCSV_data.GetColumn(0).ToInt32();
//int index = MNISTCSV_data.DeepToMatrix().GetLength(1);
int[] range = Enumerable.Range(1, (MNISTCSV_data.DeepToMatrix().GetLength(1)) - 1).ToArray();
MNISTCSV_data = MNISTCSV_data.GetColumns(range);
//
// Linear Support Vector with a polynomial kernel
//

//Iris Data
var _SVM_Poly = new MulticlassSVM_PolynomialKernel(_IrisData.iData, _IrisData.classLabels);
Console.WriteLine("Using Iris Data Multiclass SVM Polynomial Kernel");
Console.ForegroundColor = ConsoleColor.Green;
Console.Write(" Accuracy = {0:p2}", _SVM_Poly.Accuracy);
Console.ResetColor();
Console.WriteLine(" Precision = {0:p2}, Elapsed:{1}\n", _SVM_Poly.Precision[0], _SVM_Poly.RunTime);

// Wisconsin Diagnostic Breast Cancer dataset
var WDBreastData = new WisconsinDiagnosticBreastCancer();
double[][] _Cancer = WDBreastData.Features; // get the flower characteristics
int[] _CancerOutputs = WDBreastData.ClassLabels;   // get the expected flower classes
var _WDBD = new MulticlassSVM_PolynomialKernel(_Cancer, _CancerOutputs);
Console.WriteLine("Using Breast Cancer Data MulticlassSVM Polynomial Kernel");
Console.ForegroundColor = ConsoleColor.Green;
Console.Write (" Accuracy = {0:p2}",  _WDBD.Accuracy);
Console.ResetColor();
Console.WriteLine (" Precision = {0:p2}, Elapsed = {1}\n", _WDBD.Precision[0],_WDBD.RunTime);

// Wine Data
// Fails, there are no samples for class label 0
// var _Wine_MultiSVM_Poly_Kernel = new MulticlassSVM_PolynomialKernel(_Winedata.iData, _Winedata.classLabels);
// Console.WriteLine("Wine Data");
// Console.WriteLine(" Accuracy = {0:p4}, Precision = {1:p4}, Elapsed {2}\n", _Wine_MultiSVM_Poly_Kernel.Accuracy, _Wine_MultiSVM_Poly_Kernel.Precision[0], _Wine_MultiSVM_Poly_Kernel.Elapsed);

// MNIST Data
var _MNIST_Poly = new MulticlassSVM_PolynomialKernel(MNISTCSV_data, MNISTLabels);
Console.WriteLine("Using MNIST Data Polynomial Kernel");
Console.ForegroundColor = ConsoleColor.Green;
Console.Write(" Accuracy = {0:p2}", _MNIST_Poly.Accuracy);
Console.ResetColor();
Console.WriteLine (" Precision {0}, Elapsed {1}", _MNIST_Poly.Precision[0], _MNIST_Poly.RunTime);


// Iris Data
var _iris_poly_nobatch = new MultiNomialPolyKernel_no_batches(_IrisData.iData, _IrisData.classLabels);
Console.WriteLine("Iris Data - MultiNomialPolyKernel_no_batches");
Console.ForegroundColor = ConsoleColor.Green;
Console.Write("Accuracy = {0:p4})", _iris_poly_nobatch.Accuracy);
Console.ResetColor (); 
Console.WriteLine ( " Precision = {0:p4}, Elapsed {1}\n",_iris_poly_nobatch.Precision, _iris_poly_nobatch.RunTime);



// This is commented out because it is really slow
/*var _MNIST_SVM_Poly = new MulticlassSVM_PolynomialKernel(MNISTCSV_data, MNISTLabels);
// var _MNIST_SVM_Poly = new MulticlassSVM_PolynomialKernel(_MNISTData._training.Item1.Select(s => s.ToDense()).ToArray(), _MNISTData._training.Item2.ToInt32());
Console.WriteLine("Using MNIST Data");
Console.WriteLine("  MNIST Data - MultiClassSVM Polynomial kernel:Accuracy = {0:p2}, Precision = {1:p2}, Elapsed {2}\n", _MNIST_SVM_Poly.Accuracy, _MNIST_SVM_Poly.Precision[0], _MNIST_SVM_Poly.RunTime);
*/
// ***********************************************************************************************


var _IRIS_Gaussian = new MultiClassSVMGaussian (_IrisData.iData, _IrisData.classLabels);
Console.WriteLine("Using Iris Data - Multi Class SVM  Gaussian Kernel:");
Console.ForegroundColor = ConsoleColor.Green;
Console.Write(" Accuracy = {0:p4}", _IRIS_Gaussian.Accuracy);
Console.ResetColor();
Console.WriteLine(" Precision = {0:p2}, Elapsed: {1} \n", _IRIS_Gaussian.Precision, _IRIS_Gaussian.RunTime);

var _WBCD_SVMGaussian = new MultiClassSVMGaussian(_Cancer, _CancerOutputs);
Console.WriteLine("Using Wisconsin Breast Data with Multiclass SVM Gaussian Kernel");
Console.ForegroundColor = ConsoleColor.Green;
Console.Write(" Accuracy = {0:p2}", _WBCD_SVMGaussian.Accuracy);
Console.ResetColor();
Console.WriteLine(" Precision = {0:p2}, Elapsed: {1} \n", _WBCD_SVMGaussian.Precision, _WBCD_SVMGaussian.RunTime);

var _Iris = new MulticlassSVM_PolynomialKernel(_IrisData.iData, _IrisData.classLabels);
Console.WriteLine("Using Iris Data - Multi Class SVM linear kernel w/ Polynomial Transform, Average Stochastic Gradient Descent");
Console.ForegroundColor = ConsoleColor.Green;
Console.Write(" Accuracy = {0:p2}", _Iris.Accuracy);
Console.ResetColor();
Console.WriteLine(" Precision = {0:p2}, Elapsed: {1} \n", _Iris.Precision[1], _Iris.RunTime);


//
// Logistic Regression BFGS
//

var Iris_MNLR_BFGS = new MultiNomialLogisticRegressionLBFGS(_IrisData.iData, _IrisData.classLabels);

Console.ForegroundColor = ConsoleColor.Magenta;
Console.WriteLine("Iris Data - Multinomial Logistic (BFGS)");
Console.ForegroundColor = ConsoleColor.Green;
Console.Write("Accuracy = {0:p2}%", Iris_MNLR_BFGS.Accuracy);
Console.ResetColor();
Console.WriteLine(" Precision = {0:p2} {1}", Iris_MNLR_BFGS.Precision, Iris_MNLR_BFGS.RunTime);


/*var MNIST_MNLR_BFGS = new MultiNomialLogisticRegressionLBFGS(MNISTCSV_data, MNISTLabels);
Console.ForegroundColor = ConsoleColor.Magenta;
Console.WriteLine("MNIST Data - Multinomial Logistic (BFGS) Accuracy = {0:p2}%; Precision = {1:p2}, {2}", MNIST_MNLR_BFGS.Accuracy, MNIST_MNLR_BFGS.Precision, MNIST_MNLR_BFGS.RunTime);
Console.ResetColor();
*/
var Cancer_MNLR_BFGS = new MultiNomialLogisticRegressionLBFGS(_Cancer, _CancerOutputs);
Console.WriteLine("Breast Cancer Data - Multinomial Logistic (BFGS)");
Console.ForegroundColor = ConsoleColor.Red;
Console.Write(" Accuracy = {0:p2}", Cancer_MNLR_BFGS.Accuracy);
Console.ResetColor();
Console.WriteLine(" Precision = {0:p2}, {1}", Cancer_MNLR_BFGS.Precision, Cancer_MNLR_BFGS.RunTime);


var Wine_MNLR_BFGS = new MultiNomialLogisticRegressionLBFGS(_Winedata.iData, _Winedata.classLabels);
Console.WriteLine("Wine Data - Multinomial Logistic (BFGS)");
Console.ForegroundColor = ConsoleColor.Red;
Console.Write(" Accuracy = {0:p2}", Wine_MNLR_BFGS.Accuracy);
Console.ResetColor();
Console.WriteLine (" Precision = {0:p2}, {1}", Wine_MNLR_BFGS.Precision, Wine_MNLR_BFGS.RunTime);

// Logistic Regression with Stochastic Gradient Descent



