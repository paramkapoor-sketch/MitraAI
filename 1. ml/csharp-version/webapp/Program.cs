using Microsoft.ML;
using Microsoft.ML.Data;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllers();
builder.Services.AddSingleton<HomePricePredictor>();

var app = builder.Build();

// Configure the HTTP request pipeline.
app.UseDefaultFiles();
app.UseStaticFiles();
app.MapControllers();

app.Run();

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

public class PredictionRequest
{
    public float OverallQual { get; set; }
    public float GrLivArea { get; set; }
    public float GarageCars { get; set; }
    public float TotalSquareFeet { get; set; }
    public float TotalBath { get; set; }
    public float SqFtPerRoom { get; set; }
    public float YearBuilt { get; set; }
}

public class PredictionResponse
{
    public string PredictedPrice { get; set; } = "";
    public bool Success { get; set; }
    public string? Error { get; set; }
}

public class HomePricePredictor
{
    private readonly MLContext _mlContext;
    private readonly ITransformer _model;
    private readonly PredictionEngine<HousePredictionInput, HousePredictionOutput> _predictionEngine;

    public HomePricePredictor()
    {
        _mlContext = new MLContext();
        
        try
        {
            _model = _mlContext.Model.Load("../model.zip", out var modelInputSchema);
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<HousePredictionInput, HousePredictionOutput>(_model);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load model: {ex.Message}", ex);
        }
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