#!/bin/bash

echo "Setting up C# ML.NET environment for Home Prices prediction..."

# Check if dotnet is installed
if ! command -v dotnet &> /dev/null; then
    echo "dotnet is not installed. Please install it using one of these methods:"
    echo "1. Using Homebrew: brew install --cask dotnet-sdk"
    echo "2. Download from: https://dotnet.microsoft.com/download"
    echo ""
    echo "After installation, run this script again."
    exit 1
fi

echo "dotnet is installed. Version:"
dotnet --version

# Navigate to project directory
cd "$(dirname "$0")"

# Restore packages
echo "Restoring packages..."
dotnet restore HomePricesML.csproj

# Build the project
echo "Building project..."
dotnet build HomePricesML.csproj

echo "Setup complete! To run the project:"
echo "dotnet run --project HomePricesML.csproj"