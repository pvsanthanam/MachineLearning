using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MNIST
{
    /// <summary>
    /// The Digit class represents one mnist digit.
    /// </summary>
    class Digit
    {
        [ColumnName("PixelValues")]
        [VectorType(784)]
        public float[] PixelValues;

        [LoadColumn(0)]
        public float Number;
    }

    /// <summary>
    /// The DigitPrediction class represents one digit prediction.
    /// </summary>
    class DigitPrediction
    {
        [ColumnName("Score")]
        public float[] Score;
    }
}
