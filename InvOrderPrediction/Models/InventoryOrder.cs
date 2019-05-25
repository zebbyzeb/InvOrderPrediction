using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using System.Text;

namespace InvOrderPrediction.Models
{
    public class InventoryOrder
    {
        public float OrderDate { get; set; }
        [ColumnName("Label")]
        public float Qty { get; set; }
    }
}
