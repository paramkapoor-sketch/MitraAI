# Home Price Prediction - C# Deployment

This is a C# conversion of the Python Gradio deployment app using ASP.NET Core Web API + HTML frontend.

## Features

✅ **Web Interface**: Beautiful, responsive interface similar to Gradio
✅ **Real-time Sliders**: Interactive sliders for all input parameters  
✅ **REST API**: JSON-based prediction endpoint
✅ **Model Loading**: Loads the trained ML.NET model (model.zip)
✅ **Error Handling**: Graceful error handling and validation
✅ **Production Ready**: Built with ASP.NET Core for production deployment

## Quick Start

### Prerequisites
1. **.NET SDK 9.0+**: `brew install --cask dotnet-sdk`
2. **Trained Model**: Run training first to generate `model.zip`

### Option 1: Run with Script
```bash
./run-webapp.sh
```

### Option 2: Manual Setup
```bash
# Ensure model exists
dotnet run --project HomePricesML.csproj  # Train model first

# Run web app
dotnet restore HomePricesWebApp.csproj
dotnet run --project HomePricesWebApp.csproj
```

### Access the App
- **Web Interface**: http://localhost:5000
- **API Health Check**: http://localhost:5000/api/prediction/health
- **Prediction API**: POST to `/api/prediction/predict`

## API Usage

### Prediction Endpoint
```bash
curl -X POST http://localhost:5000/api/prediction/predict \
  -H "Content-Type: application/json" \
  -d '{
    "overallQual": 7,
    "grLivArea": 1710,
    "garageCars": 2,
    "totalSquareFeet": 2566,
    "totalBath": 2.5,
    "sqFtPerRoom": 214,
    "yearBuilt": 2003
  }'
```

### Response Format
```json
{
  "predictedPrice": "$208,500.00",
  "success": true,
  "error": null
}
```

## Interface Features

The web interface provides:

1. **Overall Quality** (1-10): Material and finish quality
2. **Above Grade Living Area** (500-4000 sq ft): Main living space
3. **Garage Cars** (0-4): Number of car capacity
4. **Total Square Feet** (1000-4000): Complete house area
5. **Total Bathrooms** (1-6): Full + half bathrooms combined
6. **Square Feet per Room** (500-3000): Room size efficiency
7. **Year Built** (1900-2024): Construction year

## Gradio.NET Alternative

This implementation provides a **production-ready alternative** to Gradio.NET because:

- **Gradio.NET** is experimental (v0.0.6) and may have stability issues
- **ASP.NET Core** is mature, well-documented, and production-tested
- **Full Control**: Complete customization of UI and API behavior
- **Deployment Ready**: Easy to deploy to Azure, AWS, or any cloud provider

## File Structure

```
├── HomePricesWebApp.csproj    # Main web project
├── Program.cs                 # Web app startup and ML services
├── Controllers/
│   └── PredictionController.cs # API endpoint
├── wwwroot/
│   └── index.html            # Web interface (Gradio-like UI)
└── run-webapp.sh             # Setup and run script
```

## Deployment Options

### Local Development
```bash
dotnet run --project HomePricesWebApp.csproj
```

### Production Deployment
```bash
dotnet publish -c Release
# Deploy the publish folder to your server
```

### Docker (Optional)
```dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:9.0
COPY ./publish /app
WORKDIR /app
EXPOSE 80
ENTRYPOINT ["dotnet", "HomePricesWebApp.dll"]
```

## Comparison with Python Gradio

| Feature | Python Gradio | C# ASP.NET Core |
|---------|---------------|-----------------|
| Setup Complexity | Very Simple | Simple |
| UI Customization | Limited | Full Control |
| Production Ready | Good | Excellent |
| Performance | Good | Excellent |
| Deployment | Easy | Easy |
| Scalability | Limited | High |
| Enterprise Support | Community | Microsoft |

## Next Steps

1. **Deploy to Cloud**: Use Azure App Service, AWS, or other cloud providers
2. **Add Authentication**: Implement user login/registration
3. **Database Integration**: Store predictions and user data
4. **Monitoring**: Add Application Insights or similar monitoring
5. **CI/CD**: Set up automated deployment pipelines