using Microsoft.ML;
using System;
using System.IO;

namespace BinaryClassification
{
    class Program
    {
        // filenames for training and test data
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "processed.cleveland.data.csv");

        static void Main(string[] args)
        {
            var context = new MLContext();

            // load data
            Console.WriteLine("Loading data...");
            var data = context.Data.LoadFromTextFile<HeartData>(dataPath, hasHeader: false, separatorChar: ',');

            // split training data
            var partitions = context.Data.TrainTestSplit(data, 0.2);

            // build pipeline
            var pipeline = context.Transforms.CustomMapping<HeartData, ToLabel>((i, o) =>
            {
                o.Label = i.RawLabel > 0 ? true : false;
            }, contractName: "Label")
            .Append(context.Transforms.Concatenate(
                "Features",
                "Age",
                "Sex",
                "Cp",
                "TrestBps",
                "Chol",
                "Fbs",
                "RestEcg",
                "Thalac",
                "Exang",
                "OldPeak",
                "Slope",
                "Ca",
                "Thal"))
                
            .Append(context.BinaryClassification.Trainers.FastTree("Label", "Features"));

            // Train the model
            Console.WriteLine("Training Model...");
            var model = pipeline.Fit(partitions.TrainSet);

            // evaluate the model
            Console.WriteLine("Evaluating the modle...");
            var testData = model.Transform(partitions.TestSet);
            var metrics = context.BinaryClassification.Evaluate(testData, "Label", "Score");

            // report the results
            Console.WriteLine($"  Accuracy:          {metrics.Accuracy}");
            Console.WriteLine($"  Auc:               {metrics.AreaUnderRocCurve}");
            Console.WriteLine($"  Auprc:             {metrics.AreaUnderPrecisionRecallCurve}");
            Console.WriteLine($"  F1Score:           {metrics.F1Score}");
            Console.WriteLine($"  LogLoss:           {metrics.LogLoss}");
            Console.WriteLine($"  LogLossReduction:  {metrics.LogLossReduction}");
            Console.WriteLine($"  PositivePrecision: {metrics.PositivePrecision}");
            Console.WriteLine($"  PositiveRecall:    {metrics.PositiveRecall}");
            Console.WriteLine($"  NegativePrecision: {metrics.NegativePrecision}");
            Console.WriteLine($"  NegativeRecall:    {metrics.NegativeRecall}");
            Console.WriteLine();

            // make a prediction
            Console.WriteLine("Making a prediction...");
            var predictionEngine = context.Model.CreatePredictionEngine<HeartData, HeartPrediction>(model);

            // create a sample patient
            var heartData = new HeartData()
            {
                Age = 36.0f,
                Sex = 1.0f,
                Cp = 4.0f,
                TrestBps = 145.0f,
                Chol = 210.0f,
                Fbs = 0.0f,
                RestEcg = 2.0f,
                Thalac = 148.0f,
                Exang = 1.0f,
                OldPeak = 1.9f,
                Slope = 2.0f,
                Ca = 1.0f,
                Thal = 7.0f,
            };

            // make the prediction
            var prediction = predictionEngine.Predict(heartData);

            // report the results
            Console.WriteLine($"  Age: {heartData.Age} ");
            Console.WriteLine($"  Sex: {heartData.Sex} ");
            Console.WriteLine($"  Cp: {heartData.Cp} ");
            Console.WriteLine($"  TrestBps: {heartData.TrestBps} ");
            Console.WriteLine($"  Chol: {heartData.Chol} ");
            Console.WriteLine($"  Fbs: {heartData.Fbs} ");
            Console.WriteLine($"  RestEcg: {heartData.RestEcg} ");
            Console.WriteLine($"  Thalac: {heartData.Thalac} ");
            Console.WriteLine($"  Exang: {heartData.Exang} ");
            Console.WriteLine($"  OldPeak: {heartData.OldPeak} ");
            Console.WriteLine($"  Slope: {heartData.Slope} ");
            Console.WriteLine($"  Ca: {heartData.Ca} ");
            Console.WriteLine($"  Thal: {heartData.Thal} ");
            Console.WriteLine();
            Console.WriteLine($"Prediction: {(prediction.Prediction ? "Elevated heart disease risk" : "Normal heart disease risk")} ");
            Console.WriteLine($"Probability: {prediction.Probability:P2} ");
        }
    }
}
