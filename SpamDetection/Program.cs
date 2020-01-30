using Microsoft.ML;
using System;
using System.IO;
using System.Linq;

namespace SpamDetection
{
    class Program
    {
        static string filePath = Path.Combine(Environment.CurrentDirectory, "spam.tsv");

        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<SpamInput>(filePath, hasHeader: false, separatorChar: '\t');

            var trainTestPartition = context.Data.TrainTestSplit(data, testFraction: 0.2);

            var pipeline = context.Transforms.CustomMapping<FromLabel, ToLabel>((inp, output) => { output.Label = inp.RawLabel.Equals("spam", StringComparison.InvariantCultureIgnoreCase); },
                contractName: "SpamClassification")
                .Append(context.Transforms.Text.FeaturizeText("Features", nameof(SpamInput.Message)))
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());


            Console.WriteLine("Performing cross validations...");

            var cvResults = context.BinaryClassification.CrossValidate(trainTestPartition.TrainSet, pipeline, numberOfFolds: 5);

            foreach (var result in cvResults)
            {
                Console.WriteLine($"Fold: {result.Fold}, AUC: {result.Metrics.AreaUnderRocCurve}");
            }

            Console.WriteLine($"Average AUC: {cvResults.Average(r => r.Metrics.AreaUnderRocCurve)}");
            Console.WriteLine();


            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainTestPartition.TrainSet);

            Console.WriteLine("Evaluating the model...");
            var predictions = model.Transform(trainTestPartition.TestSet);

            var metrics = context.BinaryClassification.Evaluate(predictions, "Label", "Score");

            // report the results
            Console.WriteLine($"  Accuracy:          {metrics.Accuracy:P2}");
            Console.WriteLine($"  Auc:               {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"  Auprc:             {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"  F1Score:           {metrics.F1Score:P2}");
            Console.WriteLine($"  LogLoss:           {metrics.LogLoss:0.##}");
            Console.WriteLine($"  LogLossReduction:  {metrics.LogLossReduction:0.##}");
            Console.WriteLine($"  PositivePrecision: {metrics.PositivePrecision:0.##}");
            Console.WriteLine($"  PositiveRecall:    {metrics.PositiveRecall:0.##}");
            Console.WriteLine($"  NegativePrecision: {metrics.NegativePrecision:0.##}");
            Console.WriteLine($"  NegativeRecall:    {metrics.NegativeRecall:0.##}");
            Console.WriteLine();

            // set up a prediction engine
            Console.WriteLine("Predicting spam probabilities for a sample messages...");
            var predictionEngine = context.Model.CreatePredictionEngine<SpamInput, SpamPrediction>(model);

            // create sample messages
            var messages = new SpamInput[] {
                new SpamInput() { Message = "Hi, wanna grab lunch together today?" },
                new SpamInput() { Message = "Win a Nokia, PSP, or €25 every week. Txt YEAHIWANNA now to join" },
                new SpamInput() { Message = "Home in 30 mins. Need anything from store?" },
                new SpamInput() { Message = "CONGRATS U WON LOTERY CLAIM UR 1 MILIONN DOLARS PRIZE" },
            };

            // make the prediction
            var myPredictions = from m in messages
                                select (Message: m.Message, Prediction: predictionEngine.Predict(m));

            // show the results
            foreach (var p in myPredictions)
                Console.WriteLine($"  [{p.Prediction.Probability:P2}] {p.Message}");
        }
    }
}
