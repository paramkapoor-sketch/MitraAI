using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace HomePricesML
{

    public class ProcessedHouseData
    {
        public float OverallQual { get; set; }
        public float TotalSquareFeet { get; set; }
        public float TotalBath { get; set; }
        public float GrLivArea { get; set; }
        public float SqFtPerRoom { get; set; }
        public float GarageCars { get; set; }
        public float YearBuilt { get; set; }
        public float SalePrice { get; set; }
    }

    public class HousePrediction
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 10);

            try
            {
                // Load data with explicit column mapping for problematic column names
                var loader = mlContext.Data.CreateTextLoader(
                    columns: new[]
                    {
                        new TextLoader.Column("OverallQual", DataKind.Single, 17),
                        new TextLoader.Column("BsmtFinSF1", DataKind.Single, 34),
                        new TextLoader.Column("BsmtFinSF2", DataKind.Single, 36),
                        new TextLoader.Column("FirstFlrSF", DataKind.Single, 43),
                        new TextLoader.Column("SecondFlrSF", DataKind.Single, 44),
                        new TextLoader.Column("TotalBsmtSF", DataKind.Single, 38),
                        new TextLoader.Column("FullBath", DataKind.Single, 49),
                        new TextLoader.Column("HalfBath", DataKind.Single, 50),
                        new TextLoader.Column("BsmtFullBath", DataKind.Single, 47),
                        new TextLoader.Column("BsmtHalfBath", DataKind.Single, 48),
                        new TextLoader.Column("OpenPorchSF", DataKind.Single, 67),
                        new TextLoader.Column("ThreeSsnPorch", DataKind.Single, 69),
                        new TextLoader.Column("EnclosedPorch", DataKind.Single, 68),
                        new TextLoader.Column("ScreenPorch", DataKind.Single, 70),
                        new TextLoader.Column("WoodDeckSF", DataKind.Single, 66),
                        new TextLoader.Column("GrLivArea", DataKind.Single, 46),
                        new TextLoader.Column("TotRmsAbvGrd", DataKind.Single, 54),
                        new TextLoader.Column("KitchenAbvGr", DataKind.Single, 52),
                        new TextLoader.Column("GarageCars", DataKind.Single, 61),
                        new TextLoader.Column("YearBuilt", DataKind.Single, 19),
                        new TextLoader.Column("SalePrice", DataKind.Single, 80)
                    },
                    hasHeader: true,
                    separatorChar: ','
                );

                var dataPath = "../data/train.csv";
                var rawData = loader.Load(dataPath);

                Console.WriteLine("Data loaded successfully.");

                // Transform to enumerable for processing
                var houseDataEnumerable = mlContext.Data.CreateEnumerable<HouseRawData>(rawData, reuseRowObject: false);

                // Feature engineering - convert to ProcessedHouseData
                var processedData = houseDataEnumerable
                    .Select(house => new ProcessedHouseData
                    {
                        OverallQual = house.OverallQual,
                        TotalSquareFeet = house.BsmtFinSF1 + house.BsmtFinSF2 + house.FirstFlrSF + house.SecondFlrSF + house.TotalBsmtSF,
                        TotalBath = house.FullBath + (0.5f * house.HalfBath) + house.BsmtFullBath + (0.5f * house.BsmtHalfBath),
                        GrLivArea = house.GrLivArea,
                        SqFtPerRoom = house.GrLivArea / Math.Max(1, house.TotRmsAbvGrd + house.FullBath + house.HalfBath + house.KitchenAbvGr),
                        GarageCars = house.GarageCars,
                        YearBuilt = house.YearBuilt,
                        SalePrice = house.SalePrice
                    })
                    .Where(house => !float.IsNaN(house.SqFtPerRoom) && !float.IsInfinity(house.SqFtPerRoom) && house.SalePrice > 0)
                    .ToArray();

                Console.WriteLine($"Processed {processedData.Length} records.");

                // Convert back to IDataView
                var processedDataView = mlContext.Data.LoadFromEnumerable(processedData);

                // Split data
                var splitData = mlContext.Data.TrainTestSplit(processedDataView, testFraction: 0.2, seed: 10);

                // Define pipeline
                var pipeline = mlContext.Transforms.Concatenate("Features", 
                        "OverallQual", "TotalSquareFeet", "TotalBath", "GrLivArea", 
                        "SqFtPerRoom", "GarageCars", "YearBuilt")
                    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "SalePrice", maximumNumberOfIterations: 100));

                // Train model
                Console.WriteLine("Training model...");
                var model = pipeline.Fit(splitData.TrainSet);

                // Evaluate model
                var predictions = model.Transform(splitData.TestSet);
                var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "SalePrice");

                Console.WriteLine($"R-squared: {metrics.RSquared:F4}");
                Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError:F2}");
                Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError:F2}");

                // Save model
                mlContext.Model.Save(model, processedDataView.Schema, "model.zip");
                Console.WriteLine("Model saved as model.zip");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }
    }

    public class HouseRawData
    {
        public float OverallQual { get; set; }
        public float BsmtFinSF1 { get; set; }
        public float BsmtFinSF2 { get; set; }
        public float FirstFlrSF { get; set; }
        public float SecondFlrSF { get; set; }
        public float TotalBsmtSF { get; set; }
        public float FullBath { get; set; }
        public float HalfBath { get; set; }
        public float BsmtFullBath { get; set; }
        public float BsmtHalfBath { get; set; }
        public float OpenPorchSF { get; set; }
        public float ThreeSsnPorch { get; set; }
        public float EnclosedPorch { get; set; }
        public float ScreenPorch { get; set; }
        public float WoodDeckSF { get; set; }
        public float GrLivArea { get; set; }
        public float TotRmsAbvGrd { get; set; }
        public float KitchenAbvGr { get; set; }
        public float GarageCars { get; set; }
        public float YearBuilt { get; set; }
        public float SalePrice { get; set; }
    }
}