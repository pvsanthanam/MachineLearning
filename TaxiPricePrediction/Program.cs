using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;

namespace TaxiPricePrediction
{
    class Program
    {
        static string DataPath = Path.Combine(Environment.CurrentDirectory, "yellow_tripdata_2018-12.csv");

        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            var loader = mlContext.Data.CreateTextLoader(new TextLoader.Options
            {
                HasHeader = true,
                Separators = new[] { ',' },
                Columns = new TextLoader.Column[] {
                    new TextLoader.Column("VendorID",DataKind.String, 0),
                    new TextLoader.Column("PassengerCount",DataKind.Single, 3),
                    new TextLoader.Column("TripDistance",DataKind.Single, 4),
                    new TextLoader.Column("RateCard",DataKind.String, 5),
                    new TextLoader.Column("PaymentType",DataKind.String, 9),
                    new TextLoader.Column("FareAmount",DataKind.Single, 10),
                    new TextLoader.Column("PickUpDateTime",DataKind.DateTime, 1),
                    new TextLoader.Column("DropOffDateTime",DataKind.DateTime, 2),
                }
            });

            // load the data
            Console.WriteLine("*** Loading Training Data ***");
            IDataView dataView = loader.Load(DataPath);
            Console.WriteLine("*** Done ***");

            dataView = mlContext.Data.FilterRowsByColumn(dataView, "FareAmount", lowerBound: 0);
            dataView = mlContext.Data.FilterRowsByColumn(dataView, "PassengerCount", lowerBound: 0);
            dataView = mlContext.Data.FilterRowsByColumn(dataView, "TripDistance", lowerBound: 0);


            // prepare the training data
            var trainingData = mlContext.Data.TrainTestSplit(dataView, 0.99);

            // create learning pipeline
            var pipeline1 = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(inputColumnName: "VendorID", outputColumnName: "EncodedVendorID"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(inputColumnName: "RateCard", outputColumnName: "EncodedRateCard"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(inputColumnName: "PaymentType", outputColumnName: "EncodedPaymentType"));

            var pipeline = pipeline1
                .Append(mlContext.Transforms.CustomMapping<TaxiTrip, TaxiTripTime>((inp, outp) =>
                {
                    float tripTime = 0;

                    if (inp.PickUpDateTime != DateTime.MinValue && inp.DropOffDateTime != DateTime.MinValue && inp.DropOffDateTime.Subtract(inp.PickUpDateTime).Minutes > 0)
                        tripTime = (float)inp.DropOffDateTime.Subtract(inp.PickUpDateTime).Minutes;

                    outp.TripTime = tripTime;
                }, contractName: "TripTime"))
                .Append(mlContext.Transforms.Concatenate("Features",
                "EncodedVendorID",//"VendorID",
                "PassengerCount",
                "TripDistance",
                "EncodedRateCard", //"RateCard",
                "EncodedPaymentType", //"PaymentType",
                "TripTime"))
                .Append(mlContext.Transforms.DropColumns("PickUpDateTime", "DropOffDateTime", "VendorID", "RateCard", "PaymentType"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Regression.Trainers.FastTree());

            //pipeline.Preview(dataView);

            // Train the model
            Console.WriteLine("*** Training the Model ***");
            var model = pipeline.Fit(trainingData.TrainSet);
            Console.WriteLine("*** Done ***");

            // get set of predictions
            Console.WriteLine("*** Evaluate the Model ***");
            var testData = model.Transform(trainingData.TestSet);
            Console.WriteLine("*** Done ***");

            // get the regression score
            var metrics = mlContext.Regression.Evaluate(testData, "Label", "Score");

            // print metrics
            Console.WriteLine();
            Console.WriteLine("Model Metrics:");
            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"MAE: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"MSE: {metrics.MeanSquaredError:#.##}");

            // make a single prediction
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripPrediction>(model);

            // prep a single taxi trip
            var trip = new TaxiTrip
            {
                VendorID = "1",
                PassengerCount = 1,
                RateCard = "1",
                TripDistance = 3.75f,
                PaymentType = "1",
                FareAmount = 0,
                PickUpDateTime = Convert.ToDateTime("2018-12-01 00:28:22"),
                DropOffDateTime = Convert.ToDateTime("2018-12-01 00:48:07")
            };

            var predictedFare = predictionEngine.Predict(trip);

            Console.WriteLine($"Predicted Price: {predictedFare.FareAmount:0.####}");
        }

        void GenerateHistogram(IEnumerable<TaxiTrip> data)
        {

        }

        #region Input/Output Model
        public class TaxiTripTime
        {
            public float TripTime { get; set; }
        }

        public class TaxiTrip
        {
            [LoadColumn(0)]
            public string VendorID { get; set; }
            [LoadColumn(3)]
            public float PassengerCount { get; set; }
            [LoadColumn(4)]
            public float TripDistance { get; set; }
            [LoadColumn(5)]
            public string RateCard { get; set; }
            [LoadColumn(9)]
            public string PaymentType { get; set; }
            [LoadColumn(10)]
            public float FareAmount { get; set; }
            [LoadColumn(1)]
            public DateTime PickUpDateTime { get; set; }
            [LoadColumn(2)]
            public DateTime DropOffDateTime { get; set; }
        }

        public class TaxiTripPrediction
        {
            [ColumnName("Score")]
            public float FareAmount { get; set; }
        }
        #endregion
    }
}
