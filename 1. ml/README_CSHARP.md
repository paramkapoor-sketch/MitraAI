# Home Prices ML - C# Version

This is a C# conversion of the Python home prices prediction model using ML.NET.

## Prerequisites

1. Install .NET SDK 8.0 or later:
   - **macOS**: `brew install --cask dotnet-sdk`
   - **Direct download**: https://dotnet.microsoft.com/download

2. Ensure you have the training data file at `../data/train.csv`

## Setup and Run

### Option 1: Using the setup script
```bash
./setup.sh
```

### Option 2: Manual setup
```bash
# Restore packages
dotnet restore HomePricesML.csproj

# Build the project
dotnet build HomePricesML.csproj

# Run the project
dotnet run --project HomePricesML.csproj
```

## What the Code Does

The C# version replicates the Python functionality:

1. **Data Loading**: Loads housing data from CSV
2. **Feature Engineering**: 
   - Creates `TotalSquareFeet` (sum of basement and floor areas)
   - Creates `TotalBath` (weighted sum of all bathrooms)
   - Creates `SqFtPerRoom` (living area per room ratio)
3. **Model Training**: Uses ML.NET's SDCA regression trainer
4. **Model Evaluation**: Shows R-squared, MAE, and RMSE metrics
5. **Model Saving**: Saves trained model as `model.zip`

## Key Differences from Python Version

- Uses ML.NET instead of scikit-learn
- Model saved as `.zip` instead of `.pkl`
- Uses SDCA trainer instead of LinearRegression (similar performance)
- Includes data normalization for better convergence

## Output

The program will display:
- R-squared score (coefficient of determination)
- Mean Absolute Error
- Root Mean Squared Error
- Confirmation of model save