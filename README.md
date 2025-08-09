# Google Trends Quantile Forecaster

This application creates quantile forecasts for the popularity of a series of keywords using state-of-the-art time series forecasting models from the Python Darts library, including LSTM, TCN, Transformer, Prophet, ARIMA, and more.

## API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `GET /api/health` - API health check
- `POST /api/trends` - Get Google Trends data for keywords
- `POST /api/trends/summary` - Get trends summary with statistical analysis
- `POST /api/trends/compare` - Compare multiple keywords

### Model Management
- `POST /api/models/train` - Train a new Darts model with evaluation
- `GET /api/models/{model_id}/evaluate` - Get model evaluation metrics
- `POST /api/models/{model_id}/predict` - Generate predictions with accuracy reporting
- `GET /api/models/{model_id}` - Get model information
- `GET /api/models` - List all available models
- `POST /api/models/compare` - Compare multiple models

### Cache Management
- `POST /api/trends/cache/clear` - Clear trends cache
- `GET /api/trends/cache/stats` - Get cache statistics

## Architecture Overview

The application is organized by major dependencies for better maintainability:

### Core Structure
```
app/
├── models/
│   ├── darts/           # Darts-specific data models
│   │   ├── darts_models.py
│   │   └── training_request.py
│   ├── pytrends/        # Google Trends data models
│   │   └── pytrend_model.py
│   └── prediction_model.py  # Legacy models (to be removed)
├── services/
│   ├── darts/           # Darts forecasting services
│   │   ├── training_service.py
│   │   ├── evaluation_service.py
│   │   └── prediction_service.py
│   ├── pytrends/        # Google Trends services
│   │   └── trends_service.py
│   ├── model_service.py     # Legacy services (to be removed)
│   └── prediction_service.py # Legacy services (to be removed)
└── utils/
    ├── error_handlers.py
    └── validators.py
```

### Test Structure
```
tests/
├── unit/
│   ├── darts/           # Darts service tests
│   ├── pytrends/        # Google Trends tests
│   └── [other tests]
└── integration/
    └── test_trends_api.py
```

## New Darts-Based Architecture

The application has been enhanced with the Python Darts library for robust time series forecasting:

- **Multiple Model Types**: LSTM, TCN, Transformer, Prophet, ARIMA, Exponential Smoothing, Random Forest
- **Proper Evaluation**: Built-in train/test splits with comprehensive holdout evaluation
- **Confidence Intervals**: Probabilistic forecasting with uncertainty quantification
- **Model Comparison**: Easy comparison between different model types
- **Production Ready**: Well-tested library with active maintenance

## Quick Start

### Local Development

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

### Docker Development

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Test the application:**
   ```bash
   curl http://localhost:5000/health
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f
   ```

### Production Deployment

For production deployment on AWS EC2, see the [Deployment Guide](docs/DEPLOYMENT_GUIDE.md).

**Quick EC2 Setup:**
```bash
# Clone repository on EC2
git clone <repository-url>
cd google_trends_forecaster

# Run setup scripts
chmod +x scripts/deployment/*.sh
./scripts/deployment/setup_ec2.sh
./scripts/deployment/deploy_app.sh
./scripts/deployment/security_setup.sh setup
```

## Training a Model with Darts

### Example: Train an LSTM Model
```bash
curl -X POST http://localhost:5000/api/models/train \
  -H "Content-Type: application/json" \
  -d '{
    "keyword": "artificial intelligence",
    "time_series_data": [70, 75, 80, 85, 90, 88, 92, 95, 89, 87, ...],
    "dates": ["2023-01-01", "2023-01-08", "2023-01-15", "2023-01-22", ...],
    "model_type": "lstm",
    "train_test_split": 0.8,
    "forecast_horizon": 25,
    "model_parameters": {
      "input_chunk_length": 12,
      "n_epochs": 100
    }
  }'
```

### Example: Get Model Evaluation
```bash
curl -X GET http://localhost:5000/api/models/{model_id}/evaluate
```

### Example: Generate Forecast with Accuracy
```bash
curl -X POST http://localhost:5000/api/models/{model_id}/predict \
  -H "Content-Type: application/json" \
  -d '{
    "forecast_horizon": 25
  }'
```

## Development

- **Run tests:** `pytest`
- **Run unit tests:** `python -m pytest tests/unit/ -v`
- **Run darts tests:** `python -m pytest tests/unit/darts/ -v`
- **Run pytrends tests:** `python -m pytest tests/unit/pytrends/ -v`
- **Run integration tests:** `python -m pytest tests/integration/ -v`
- **Project structure:** See the implementation plan for detailed architecture
- **API specification:** See the API endpoints documentation

## Key Features

### Comprehensive Model Evaluation
- **Train/Test Split**: Configurable split ratios with holdout evaluation
- **Multiple Metrics**: MAE, RMSE, MAPE, directional accuracy, confidence intervals
- **Model Comparison**: Compare different model types for the same keyword
- **Performance Tracking**: Training time, memory usage, and scalability metrics

### Advanced Forecasting
- **Confidence Intervals**: Probabilistic forecasting with uncertainty quantification
- **Multiple Horizons**: Configurable forecast periods
- **Model Persistence**: Save and load trained models
- **MLflow Integration**: Model tracking and versioning

### Robust API Design
- **Input Validation**: Comprehensive validation for all inputs
- **Error Handling**: Detailed error messages and proper HTTP status codes
- **Rate Limiting**: Protection against API abuse
- **Caching**: Efficient caching for trends data and model predictions

## Model Types Supported

1. **LSTM**: Long Short-Term Memory networks for complex temporal patterns
2. **TCN**: Temporal Convolutional Networks for efficient sequence modeling
3. **Transformer**: Attention-based models for long-range dependencies
4. **Prophet**: Facebook's forecasting tool for trend and seasonality
5. **ARIMA**: Classical statistical forecasting method
6. **Exponential Smoothing**: Simple but effective smoothing techniques
7. **Random Forest**: Ensemble method for time series forecasting
8. **N-BEATS**: Neural basis expansion analysis for interpretable time series
9. **TFT**: Temporal Fusion Transformers for multivariate forecasting
10. **GRU**: Gated Recurrent Units for efficient sequence modeling
11. **AutoARIMA**: Automatic ARIMA model selection
12. **AutoETS**: Automatic Exponential Smoothing model selection
13. **AutoTheta**: Automatic Theta model selection
14. **AutoCES**: Automatic Complex Exponential Smoothing model selection

## Performance Characteristics

- **Training Time**: 30 seconds to 5 minutes depending on model type and data size
- **Prediction Speed**: Sub-second predictions for most models
- **Memory Usage**: Efficient memory management with model caching
- **Scalability**: Supports multiple concurrent training and prediction requests

## Current Status

✅ **Phase 1 Complete**: Foundation setup with Darts data models and backward compatibility
✅ **Phase 2 Complete**: Core Darts services implementation with comprehensive testing (245 tests passing)
✅ **Phase 3 Complete**: Refactoring - Organizing code by major dependencies
✅ **Phase 4 Complete**: API endpoints and integration testing
✅ **Phase 5 Complete**: Cleanup and optimization
✅ **Phase 6 Complete**: Integration test updates
✅ **Phase 7 Complete**: Deployment preparation with Docker and EC2 support

## Next Steps

1. **Production Deployment**: Deploy to AWS EC2 using the provided scripts
2. **Performance Optimization**: Monitor and optimize based on real usage
3. **Scaling**: Consider load balancing and auto-scaling for high traffic
4. **Monitoring**: Set up comprehensive monitoring and alerting

## Deployment

The application is ready for production deployment with:

- **Docker Containerization**: Complete Docker setup with optimized images
- **EC2 Deployment Scripts**: Automated deployment and configuration
- **Security Configuration**: Firewall, SSL, and monitoring setup
- **Health Monitoring**: Automated health checks and alerting
- **Documentation**: Comprehensive deployment guide

See [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) for detailed instructions.

## Documentation

- **Implementation Checklist**: `docs/IMPLEMENTATION_CHECKLIST.md` - Complete project phases and progress
- **Deployment Guide**: `docs/DEPLOYMENT_GUIDE.md` - Production deployment instructions
- **API Endpoints**: See API documentation in the main README
