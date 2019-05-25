using Microsoft.ML;
using System;
using InvOrderPrediction.Models;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace InvOrderPrediction
{
    class Program
    {
        public class HouseData
        {
            public float Size { get; set; }
            public float Price { get; set; }
        }

        public class Prediction
        {
            [ColumnName("Score")]
            public float Price { get; set; }
        }

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // 1. Import or create training data
            HouseData[] houseData = {
               new HouseData() { Size = 1.1F, Price = 1.2F },
               new HouseData() { Size = 1.9F, Price = 2.3F },
               new HouseData() { Size = 2.8F, Price = 3.0F },
               new HouseData() { Size = 3.4F, Price = 3.7F } };
            IDataView trainingData = mlContext.Data.LoadFromEnumerable(houseData);

            // 2. Specify data preparation and model training pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Size" })
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

            // 3. Train model
            var model = pipeline.Fit(trainingData);

            // 4. Make a prediction
            var size = new HouseData() { Size = 2.5F };
            var price = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(size);

            Console.WriteLine($"Predicted price for size: {size.Size * 1000} sq ft= {price.Price * 100:C}k");

            //static void Main(string[] args)
            //{
            //    MLContext mlContext = new MLContext();
            //    InventoryOrder[] inventoryOrderList = new InventoryOrder[] {

            //    new InventoryOrder() { OrderDate = new DateTime(2019, 01, 01).Ticks, Qty = 5 },
            //    new InventoryOrder() { OrderDate = new DateTime(2019, 01, 15).Ticks, Qty = 7 },
            //    new InventoryOrder() { OrderDate = new DateTime(2019, 02, 01).Ticks, Qty = 3 },
            //    new InventoryOrder() { OrderDate = new DateTime(2019, 02, 14).Ticks, Qty = 10 },
            //    new InventoryOrder() { OrderDate = new DateTime(2019, 03, 20).Ticks, Qty = 7 }
            //    };

            //    //InventoryOrder[] inventoryOrderListTest = new InventoryOrder[] {

            //    //new InventoryOrder() { OrderDate = new DateTime(2019, 01, 01), Qty = 15 },
            //    //new InventoryOrder() { OrderDate = new DateTime(2019, 01, 15), Qty = 17 },
            //    //new InventoryOrder() { OrderDate = new DateTime(2019, 02, 01), Qty = 13 },
            //    //new InventoryOrder() { OrderDate = new DateTime(2019, 02, 14), Qty = 20 },
            //    //new InventoryOrder() { OrderDate = new DateTime(2019, 03, 20), Qty = 17 }
            //    //};

            //    var model = Train(mlContext, inventoryOrderList);
            //    //Evaluate(mlContext, model, inventoryOrderListTest);
            //    TestSinglePrediction(mlContext, model);
            //}
        }
        private static ITransformer Train(MLContext mlContext, InventoryOrder[] inventoryOrderList)
        {
            IDataView data = mlContext.Data.LoadFromEnumerable(inventoryOrderList);
            //var pipeline = mlContext.Transforms.Conversion.ConvertType("Label", "Qty", DataKind.Single)
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "OrderDateEncoded", inputColumnName: "OrderDate"))
                var pipeline =
                //mlContext.Transforms.Conversion.ConvertType("OrderDate", null, DataKind.Single)
                //.Append(mlContext.Transforms.Conversion.ConvertType("Label", null, DataKind.Single))
                mlContext.Transforms.Concatenate("Features", new[] { "OrderDate" })
                
                .Append(mlContext.Regression.Trainers.FastTree());
            //.Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));
            var model = pipeline.Fit(data);
            return model;            
        }

        private static void Evaluate(MLContext mlContext, ITransformer model, InventoryOrder[] inventoryOrderListTest)
        {
            IDataView data = mlContext.Data.LoadFromEnumerable(inventoryOrderListTest);
            var predictions = model.Transform(data);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {

            var predictionFunction = mlContext.Model.CreatePredictionEngine<InventoryOrder, QtyPrediction>(model);

            var inventoryOrderSample = new InventoryOrder() { OrderDate = new DateTime(2019, 04, 20).Ticks};
            var prediction = predictionFunction.Predict(inventoryOrderSample);

            Console.WriteLine($"Predicted fare: {prediction.Qty}, actual fare: 15.5");

        }
    }
}
