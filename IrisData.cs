using Accord.DataSets;
using Accord.Math;

/* The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Ronald Fisher in his 1936 paper 
 * The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis. It is sometimes called
 * Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic
 * variation of Iris flowers of three related species. Two of the three species were collected in the Gaspé Peninsula 
 * "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".
 * The data set consists of 50 samples from each of three species of Iris(Iris setosa, Iris virginica and Iris versicolor). 
 * Four features were measured from each sample: the length and the width of the sepals and petals, in centimetres. 
 * Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species 
 * from each other.*/

namespace _AccordData
{
    public class irisData          
    {  
        public double[][] iData { get; }  // get the flower data
        public int[] classLabels { get; }
        
        public irisData()
        {
            var _iris = new Iris(); // create the internal instance of the data in the consctructor for the class
            classLabels = _iris.ClassLabels;
            iData = _iris.Instances;

        }
    }

    public class MNISTData
    {
        //public double [] TrainingData { get; } // Handwritten digit data set
        public Tuple<Sparse<double>[], double[]> _training;
        public Tuple<Sparse<double>[], double[]> _testData;
        public string Destination { get; }

        public MNISTData()
        {
            var _MNIST = new Accord.DataSets.MNIST();
            _training = _MNIST.Training;
            Destination = _MNIST.Path;
            _testData = _MNIST.Testing;

        }
    }

    public class BreastData // Breast Cancer Data
    {
        public int?[][] iData { get; }
        public int[] classLabels { get; }
        public BreastData()
        {
            var _breast = new WisconsinOriginalBreastCancer();
            iData = _breast.Features;
            classLabels = _breast.ClassLabels;
        }
    }
    public class WineData
    {
        public double[][] iData { get; }
        public int[] classLabels { get; }
        public WineData()
        {
            var _wine = new Wine();
            iData = _wine.Instances;
            classLabels = _wine.ClassLabels;
        }
    }
}
