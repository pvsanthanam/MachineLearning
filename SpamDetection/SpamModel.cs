using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace SpamDetection
{
    public class SpamInput
    {
        [LoadColumn(0)]
        public string RawLabel { get; set; }
        [LoadColumn(1)]
        public string Message { get; set; }
    }

    public class SpamPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsSpam { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }

    /// <summary>
    /// This class describes which input columns we want to transform.
    /// </summary>
    public class FromLabel
    {
        public string RawLabel { get; set; }
    }

    /// <summary>
    /// This class describes what output columns we want to produce.
    /// </summary>
    public class ToLabel
    {
        public bool Label { get; set; }
    }
}
