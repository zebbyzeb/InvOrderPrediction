using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace InvOrderPrediction.Models
{
    public class QtyPrediction
    {
        [ColumnName("Score")]
        public float Qty { get; set; }
    }
}
