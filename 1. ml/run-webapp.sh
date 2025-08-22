#!/bin/bash

echo "Setting up C# Home Price Prediction Web App..."

# Check if dotnet is installed
if ! command -v dotnet &> /dev/null; then
    echo "dotnet is not installed. Please install it using:"
    echo "brew install --cask dotnet-sdk"
    exit 1
fi

echo "dotnet is installed. Version:"
dotnet --version

# Navigate to project directory
cd "$(dirname "$0")"

# Check if model exists
if [ ! -f "model.zip" ]; then
    echo "Model file 'model.zip' not found. Please run the training script first:"
    echo "dotnet run --project HomePricesML.csproj"
    exit 1
fi

# Create Controllers directory if it doesn't exist
mkdir -p Controllers
mkdir -p wwwroot

# Restore packages
echo "Restoring packages..."
dotnet restore HomePricesWebApp.csproj

# Build the project
echo "Building project..."
dotnet build HomePricesWebApp.csproj

# Run the web application
echo "Starting web application..."
echo "Open your browser to: https://localhost:5001 or http://localhost:5000"
echo "Press Ctrl+C to stop the server"
dotnet run --project HomePricesWebApp.csproj