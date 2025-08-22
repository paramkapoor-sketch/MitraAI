# Home Price Prediction - ML Project

This project contains both Python and C# implementations of a home price prediction model.

## Project Structure

```
├── csharp-version/          # ✨ C# ML.NET Implementation
│   ├── README.md           # Complete C# setup instructions  
│   ├── HomePricesTrain.cs  # Training script
│   ├── HomePricesML.csproj # Training project
│   └── webapp/             # Web application
│       ├── Program.cs      # API backend
│       ├── Controllers/    # REST endpoints
│       └── wwwroot/        # Gradio-like interface
├── 0_home_prices_train.py  # Original Python training
├── 1_home_prices_deploy.py # Original Python Gradio app
└── ../data/train.csv       # Training data
```

## Quick Start

### Python Version (Original)
```bash
python 0_home_prices_train.py   # Train model
python 1_home_prices_deploy.py  # Launch Gradio app
```

### C# Version (Converted)
```bash
cd csharp-version/
# See README.md for full instructions
```

## C# Advantages

- **Better Performance**: ML.NET optimized for .NET
- **Production Ready**: ASP.NET Core web framework
- **Enterprise Support**: Full Microsoft ecosystem
- **Type Safety**: Strong typing throughout

The C# version provides equivalent functionality with a beautiful web interface similar to Gradio!