#!/bin/bash

echo "Starting Home Price Prediction Web App..."

# Check if model exists
if [ ! -f "model.zip" ]; then
    echo "❌ Model file 'model.zip' not found!"
    echo "Please train the model first:"
    echo "  dotnet run"
    exit 1
fi

echo "✅ Model found. Starting web application..."
cd webapp/
echo "🌐 Opening web interface at http://localhost:5002"
echo "Press Ctrl+C to stop the server"
dotnet run --urls "http://localhost:5002"
