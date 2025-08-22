using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace HomePricesDeploy
{
    public class HousePredictionInput
    {
        public float OverallQual { get; set; }
        public float TotalSquareFeet { get; set; }
        public float TotalBath { get; set; }
        public float GrLivArea { get; set; }
        public float SqFtPerRoom { get; set; }
        public float GarageCars { get; set; }
        public float YearBuilt { get; set; }
    }

    public class HousePredictionOutput
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }

    public class HomePricePredictor
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _model;
        private readonly PredictionEngine<HousePredictionInput, HousePredictionOutput> _predictionEngine;

        public HomePricePredictor(string modelPath)
        {
            _mlContext = new MLContext();
            _model = _mlContext.Model.Load(modelPath, out var modelInputSchema);
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<HousePredictionInput, HousePredictionOutput>(_model);
        }

        public string PredictPrice(float overallQual, float grLivArea, float garageCars, 
                                 float totalSquareFeet, float totalBath, float sqFtPerRoom, float yearBuilt)
        {
            var input = new HousePredictionInput
            {
                OverallQual = overallQual,
                TotalSquareFeet = totalSquareFeet,
                TotalBath = totalBath,
                GrLivArea = grLivArea,
                SqFtPerRoom = sqFtPerRoom,
                GarageCars = garageCars,
                YearBuilt = yearBuilt
            };

            var prediction = _predictionEngine.Predict(input);
            return $"${prediction.Price:N2}";
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                var predictor = new HomePricePredictor("model.zip");
                
                // Since Gradio.NET is experimental (v0.0.6), we'll create a fallback solution
                Console.WriteLine("Gradio.NET is experimental. Creating ASP.NET Core alternative...");
                
                // For now, test the prediction engine
                var testPrice = predictor.PredictPrice(7, 1710, 2, 2566, 2.5f, 214f, 2003);
                Console.WriteLine($"Test prediction: {testPrice}");
                
                Console.WriteLine("Model loaded successfully. Run the ASP.NET Core version instead.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }
}