using BetterConsoleTables;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using PLplot;
using System;
using System.IO;
using System.Linq;

namespace LoadingData
{
    class Program
    {
        static string FilePath = Path.Combine(Environment.CurrentDirectory, "california_housing.csv");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            IDataView dataView = mlContext.Data.LoadFromTextFile<HousingData>(FilePath, hasHeader: true, separatorChar: ',');

            dataView = mlContext.Data.FilterRowsByColumn(dataView, "MedianHouseValue", upperBound: 500000);


            GenerateHistogram(mlContext, dataView);

            var estimator = mlContext.Transforms.CustomMapping<HousingData, MedianHouseValue>((input, output) => { output.NormalizedMedianHouseValue = input.MedianHouseValue / 1000; },
                contractName: "MedianHouseValue");

            var estimator1 = estimator.Append(mlContext.Transforms.CustomMapping<HousingData, Rooms>((input, output) =>
            {
                //output.RoomsPerPerson = (float)Math.Log(((double)input.TotalRooms / (double)input.Population) + (double)1); 
                output.RoomsPerPerson = (input.TotalRooms / input.Population) + 1;
            }, contractName: "RoomsPerPerson"));

            //dataView = mlContext.Data.FilterRowsByColumn(dataView, "RoomsPerPerson", upperBound: 4);

            //GenerateHistogram1(mlContext, dataView);

            var estimator2 = estimator1.Append(mlContext.Transforms.NormalizeBinning(outputColumnName: "BinnedLongitude", inputColumnName: "Longitude", maximumBinCount: 10))
                .Append(mlContext.Transforms.NormalizeBinning(outputColumnName: "BinnedLatitude", inputColumnName: "Latitude", maximumBinCount: 10))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "EncodedLongitude", inputColumnName: "BinnedLongitude"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "EncodedLatitude", inputColumnName: "BinnedLatitude"));

            var estimator3 = estimator2.Append(mlContext.Transforms.CustomMapping<EncodedLocation, NormalizedLocation>(
                (input, output) =>
                {
                    output.Location = new float[input.EncodedLongitude.Length * input.EncodedLatitude.Length];
                    var ind = 0;
                    for (var i = 0; i < input.EncodedLongitude.Length; i++)
                        for (var j = 0; j < input.EncodedLatitude.Length; j++)
                            output.Location[ind++] = input.EncodedLongitude[i] * input.EncodedLatitude[j];
                }, contractName: "Location"))
                .Append(mlContext.Transforms.DropColumns("Longitude", "Latitude", "BinnedLongitude", "BinnedLatitude", "EncodedLongitude", "EncodedLatitude", "MedianHouseValue"));

            var model = estimator3.Fit(dataView);
            var transformedData = model.Transform(dataView);
            var preview = transformedData.Preview(10);

            WritePreview(preview);
        }

        private static void GenerateHistogram(MLContext context, IDataView data)
        {
            // get an array of housing data
            var houses = context.Data.CreateEnumerable<HousingData>(data, reuseRowObject: false).ToArray();

            // plot median house value by longitude
            var pl = new PLStream();
            pl.sdev("pngcairo");                // png rendering
            pl.sfnam("data.png");               // output filename
            pl.spal0("cmap0_alternate.pal");    // alternate color palette
            pl.init();
            pl.env(
                0, 10,                          // x-axis range
                0, 600000,                      // y-axis range
                AxesScale.Independent,          // scale x and y independently
                AxisBox.BoxTicksLabelsAxes);    // draw box, ticks, and num ticks
            pl.lab(
                "Median Income",                // x-axis label
                "Median House Value",           // y-axis label
                "House value by longitude");    // plot title
            pl.sym(
                houses.Select(h => (double)h.MedianIncome).ToArray(),
                houses.Select(h => (double)h.MedianHouseValue).ToArray(),
                (char)218
            );
            pl.eop();
        }

        private static void GenerateHistogram1(MLContext context, IDataView data)
        {
            // get an array of housing data
            var houses = context.Data.CreateEnumerable<HousingData>(data, reuseRowObject: false).ToArray();

            // plot median house value by longitude
            var pl = new PLStream();
            pl.sdev("pngcairo");                // png rendering
            pl.sfnam("data1.png");               // output filename
            pl.spal0("cmap0_alternate.pal");    // alternate color palette
            pl.init();
            pl.env(
                0, 10,                          // x-axis range
                0, 600000,                      // y-axis range
                AxesScale.Independent,          // scale x and y independently
                AxisBox.BoxTicksLabelsAxes);    // draw box, ticks, and num ticks
            pl.lab(
                "Rooms Per Person",                // x-axis label
                "Count",           // y-axis label
                "House value by longitude");    // plot title

            //var grouped = houses.Select(c => new { RoomsPerPerson = (double)c.TotalRooms / (double)c.Population, Data = c }).GroupBy(g => g.RoomsPerPerson)
            //    .Select(s => new { RoomsPerPerson = s.Key, Count = s.Sum().Count } );


            //pl.line(
            //    grouped.Select(h => (double)h.RoomsPerPerson).ToArray(),
            //    grouped.Select(h => (double)h.Count).ToArray()                
            //);
            //pl.eop();
        }

        /// <summary>
        /// Helper method to write the machine learning pipeline to the console.
        /// </summary>
        /// <param name="preview">The data preview to write.</param>
        public static void WritePreview(DataDebuggerPreview preview)
        {
            // set up a console table
            var table = new Table(
                TableConfiguration.Unicode(),
                preview.ColumnView.Select(c => new ColumnHeader(c.Column.Name)).ToArray());

            // fill the table with results
            foreach (var row in preview.RowView)
            {
                table.AddRow((from c in row.Values
                              select c.Value is VBuffer<float> ? "<vector>" : c.Value
                            ).ToArray());
            }

            // write the table
            Console.WriteLine(table.ToString());

            // set up a console table
            table = new Table(TableConfiguration.Unicode(), new ColumnHeader("Location"));

            // fill the table with results
            foreach (var row in preview.RowView)
            {
                foreach (var col in row.Values)
                {
                    if (col.Key == "Location")
                    {
                        var vector = (VBuffer<float>)col.Value;
                        table.AddRow(string.Concat(vector.DenseValues()));
                    }
                }
            }

            // write the table
            Console.WriteLine(table.ToString());
        }

        #region Input Model
        public class HousingData
        {
            [LoadColumn(0)]
            public float Longitude { get; set; }
            [LoadColumn(1)]
            public float Latitude { get; set; }
            [LoadColumn(2)]
            public float HousingMedianAge { get; set; }
            [LoadColumn(3)]
            public float TotalRooms { get; set; }
            [LoadColumn(4)]
            public float TotalBedRooms { get; set; }
            [LoadColumn(5)]
            public float Population { get; set; }
            [LoadColumn(6)]
            public float HouseHolds { get; set; }
            [LoadColumn(7)]
            public float MedianIncome { get; set; }
            [LoadColumn(8)]
            public float MedianHouseValue { get; set; }
            [LoadColumn(5)]
            public float RoomsPerPerson { get; set; }
        }

        //public class HousingDataFeatures : HousingData
        //{
        //    public float RoomsPerPerson { get; set; }
        //}

        public class Rooms
        {
            public float RoomsPerPerson { get; set; }
        }

        public class MedianHouseValue
        {
            public float NormalizedMedianHouseValue { get; set; }
        }

        public class EncodedLocation
        {
            public float[] EncodedLongitude { get; set; }
            public float[] EncodedLatitude { get; set; }
        }

        public class NormalizedLocation
        {
            public float[] Location { get; set; }
        }

        public class Location
        {
        }

        #endregion
    }
}
