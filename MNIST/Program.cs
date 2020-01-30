using BetterConsoleTables;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.IO;
using System.Linq;

namespace MNIST
{
    class Program
    {
        // filenames for data set
        private static string trainDataPath = Path.Combine(Environment.CurrentDirectory, "mnist_train.csv");
        private static string testDataPath = Path.Combine(Environment.CurrentDirectory, "mnist_test.csv");

        static void Main(string[] args)
        {
            var context = new MLContext();

            Console.WriteLine("Loading Data...");

            var colDef = new TextLoader.Column[] {
                new TextLoader.Column(nameof(Digit.PixelValues), DataKind.Single, 1, 784),
                new TextLoader.Column("Number", DataKind.Single, 0)
            };

            var trainDataView = context.Data.LoadFromTextFile(trainDataPath, colDef, hasHeader: true, separatorChar: ',');
            var testDataView = context.Data.LoadFromTextFile(testDataPath, colDef, hasHeader: true, separatorChar: ',');

            var pipeline = context.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(context.Transforms.Concatenate("Features", nameof(Digit.PixelValues)))
                .AppendCacheCheckpoint(context)
                .Append(context.MulticlassClassification.Trainers.OneVersusAll(context.BinaryClassification.Trainers.FastForest(),  "Label"))
                .Append(context.Transforms.Conversion.MapKeyToValue(outputColumnName: "Number", inputColumnName: "Label"));

            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainDataView);

            Console.WriteLine("Evaluating model...");
            var predictions = model.Transform(testDataView);

            var metrics = context.MulticlassClassification.Evaluate(predictions, labelColumnName:"Number", scoreColumnName:"Score");

            // show evaluation metrics
            Console.WriteLine($"Evaluation metrics");
            Console.WriteLine($"    MicroAccuracy:    {metrics.MicroAccuracy:0.###}");
            Console.WriteLine($"    MacroAccuracy:    {metrics.MacroAccuracy:0.###}");
            Console.WriteLine($"    LogLoss:          {metrics.LogLoss:#.###}");
            Console.WriteLine($"    LogLossReduction: {metrics.LogLossReduction:#.###}");
            Console.WriteLine();

            var digits = context.Data.CreateEnumerable<Digit>(testDataView, false).ToArray();

            var testDigits = new Digit[] {
                digits[215], // 0
                digits[202], // 1
                digits[199], // 2
                digits[200], // 3
                digits[198], // 4
                digits[207], // 5
                digits[201], // 6
                digits[220], // 7
                digits[226], // 8
                digits[235] // 9
            };

            var engine = context.Model.CreatePredictionEngine<Digit, DigitPrediction>(model);

            var table = new BetterConsoleTables.Table(TableConfiguration.Unicode());
            table.AddColumn("Digits");
            for (var i = 0; i < 10; i++)
                table.AddColumn($"P{i}");

            for (var i = 0; i < testDigits.Length; i++)
            {
                var prediction = engine.Predict(testDigits[i]);
                table.AddRow(
                    testDigits[i].Number,
                    prediction.Score[0].ToString("P2"),
                    prediction.Score[1].ToString("P2"),
                    prediction.Score[2].ToString("P2"),
                    prediction.Score[3].ToString("P2"),
                    prediction.Score[4].ToString("P2"),
                    prediction.Score[5].ToString("P2"),
                    prediction.Score[6].ToString("P2"),
                    prediction.Score[7].ToString("P2"),
                    prediction.Score[8].ToString("P2"),
                    prediction.Score[9].ToString("P2"));
            }

            // show results
            Console.WriteLine(table.ToString());
        }
    }
}
