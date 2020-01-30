using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace BinaryClassification
{
    /// <summary>
    /// The HeartData record holds one single heart data record.
    /// </summary>
    public class HeartData
    {
        [LoadColumn(0)] public float Age { get; set; }
        [LoadColumn(1)] public float Sex { get; set; }
        [LoadColumn(2)] public float Cp { get; set; }
        [LoadColumn(3)] public float TrestBps { get; set; }
        [LoadColumn(4)] public float Chol { get; set; }
        [LoadColumn(5)] public float Fbs { get; set; }
        [LoadColumn(6)] public float RestEcg { get; set; }
        [LoadColumn(7)] public float Thalac { get; set; }
        [LoadColumn(8)] public float Exang { get; set; }
        [LoadColumn(9)] public float OldPeak { get; set; }
        [LoadColumn(10)] public float Slope { get; set; }
        [LoadColumn(11)] public float Ca { get; set; }
        [LoadColumn(12)] public float Thal { get; set; }
        [LoadColumn(13)] public int RawLabel { get; set; }
    }

    /// <summary>
    /// The HeartPrediction class contains a single heart data prediction.
    /// </summary>
    public class HeartPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction;
        public float Probability;
        public float Score;
    }

    /// <summary>
    /// The FromLabel class is a helper class for a column transformation.
    /// </summary>
    public class FromLabel
    {
        public int RawLabel;
    }

    /// <summary>
    /// The ToLabel class is a helper class for a column transformation.
    /// </summary>
    public class ToLabel
    {
        public bool Label;
    }
}
