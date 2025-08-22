# Home Price Prediction - C# ML.NET

Python to C# conversion using ML.NET and ASP.NET Core with Gradio-like web interface.

## Quick Start

### 1. Install .NET
```bash
brew install --cask dotnet-sdk
```

### 2. Train Model
```bash
dotnet run --project HomePricesML.csproj
```
Creates `model.zip` 

### 3. Run Web App
```bash
dotnet run --project HomePricesWebApp.csproj
```
Opens http://localhost:5000

## What You Get

- **ML Training**: R-squared 0.8121, processes `../data/train.csv`
- **Web Interface**: Gradio-like sliders for 7 house features
- **REST API**: `/api/prediction/predict` endpoint
- **Production Ready**: ASP.NET Core deployment

## Files

- `HomePricesTrain.cs` - Training script (ML.NET)
- `HomePricesWebApp.csproj` - Web app
- `Program.cs` - API backend  
- `wwwroot/index.html` - Frontend

Perfect Python Gradio alternative for .NET!