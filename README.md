# Google Trends Quantile Forecaster

This application creates quantile forecasts for the popularity of a series of keywords using Long short-term memory (LSTM), an artificial recurrent neural network (RNN) architecture which is capable of learning long-term dependencies.

## New Flask API Structure

The application has been refactored into a modern Flask API with the following features:

- **RESTful API endpoints** for trends data, model training, and predictions
- **Modular architecture** with separate services for different functionalities
- **Input validation** and error handling
- **MLflow integration** for model tracking and versioning
- **Rate limiting** and CORS support
- **Comprehensive logging** and monitoring

## API Endpoints

- `GET /health` - Health check
- `GET /api/health` - API health check
- `POST /api/trends` - Get Google Trends data for a keyword
- `POST /api/models/train` - Train a new LSTM model
- `POST /api/models/{model_id}/predict` - Generate predictions using a trained model
- `GET /api/models/{model_id}` - Get model information
- `GET /api/models` - List all available models

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Run the application:**
   ```bash
   python run.py
   ```

4. **Test the API:**
   ```bash
   curl http://localhost:5000/health
   ```

## Development

- **Run tests:** `pytest`
- **Project structure:** See `docs/DETAILED_IMPLEMENTATION_PLAN.md`
- **API specification:** See `docs/API_SPECIFICATION.md`

## Original Functionality

The original command-line functionality is preserved in the legacy files:
- `pytrends_driver.py` - Main driver script
- `data_transform.py` - Data preprocessing
- `util.py` - Utility functions

## Documentation

Comprehensive documentation is available in the `docs/` directory:
- Implementation plan and architecture
- API specifications and examples
- Setup and deployment guides
- Testing strategies
