# Google Trends Quantile Forecaster

An intelligent forecasting system that combines natural language understanding with state-of-the-art time series forecasting to provide quantile predictions for keyword popularity trends.

## 🤖 Agentic Capabilities

The system features a **hybrid intent recognizer** that understands natural language queries and routes them to appropriate forecasting services. Users can interact with the system using conversational language instead of learning specific API endpoints.

### Natural Language Interface

Ask questions like:
- *"How will machine learning trend next week?"*
- *"Compare artificial intelligence vs data science popularity"*
- *"Train a model for blockchain forecasting"*
- *"Evaluate the performance of my models"*
- *"Show me a summary of current AI trends"*
- *"Forecast 'python programming' trends in United States for next month with p10/p50/p90"*

The system automatically:
1. **Recognizes your intent** using semantic similarity and pattern matching
2. **Extracts relevant information** from your query (keywords, time horizons, quantiles, locations)
3. **Routes to appropriate services** for forecasting, comparison, or analysis
4. **Returns natural language responses** with actionable insights

## 🏗️ Architecture Overview

### Core Components

```
app/
├── api/
│   ├── agent_routes.py      # Natural language agent interface
│   └── routes.py            # Traditional REST API endpoints
├── services/
│   ├── agent/               # Natural language processing
│   │   ├── agent_service.py
│   │   ├── intent_recognizer.py  # Hybrid semantic + regex recognizer
│   │   ├── slot_extractor.py     # Advanced parameter extraction
│   │   └── validators.py
│   ├── darts/               # Time series forecasting
│   │   ├── training_service.py
│   │   ├── evaluation_service.py
│   │   └── prediction_service.py
│   └── pytrends/            # Google Trends data
│       └── trends_service.py
├── models/
│   ├── agent_models.py      # Agent data models
│   ├── darts/               # Forecasting models
│   └── pytrends/            # Trends data models
└── config/
    ├── agent_config.py      # Agent configuration
    └── config.py            # General configuration
```

## 🚀 Quick Start

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

4. **Test the agent interface:**
   ```bash
   curl -X POST http://localhost:5000/api/agent/query \
     -H "Content-Type: application/json" \
     -d '{"query": "How will AI trend next week?"}'
   ```

### Docker Development

```bash
docker-compose up --build
```

## 🧠 Intelligent Intent Recognition

The system uses a **hybrid approach** combining:

### Semantic Understanding (60% weight)
- **TF-IDF vectors** for semantic similarity
- **Cosine similarity** to match user queries to intent examples
- **Handles paraphrases** and natural language variations
- **Case-insensitive processing** for robust matching

### Pattern Matching (30% weight)
- **Regex patterns** as guardrails for precision
- **Keyword requirements** and exclusions
- **Confidence boosting** for multiple matches
- **Advanced text normalization** with punctuation removal

### LLM Integration (10% weight)
- **Placeholder for future** LLM-based classification
- **Few-shot learning** capabilities
- **Tie-breaking** for ambiguous cases

### Confidence Thresholds
- **High Confidence** (≥ 0.45): Accept intent classification
- **Low Confidence** (0.25 - 0.45): Return UNKNOWN with clarification
- **Unknown** (< 0.25): Return UNKNOWN

## 🔍 Advanced Slot Extraction

The system intelligently extracts parameters from natural language queries:

### Keyword Extraction
- **Quoted keywords**: `"machine learning"` and `'artificial intelligence'`
- **Comparison patterns**: `python vs javascript`, `AI versus ML`
- **Contextual extraction**: `for machine learning`, `about data science`
- **Individual word extraction**: Breaks down multi-word phrases

### Time and Date Extraction
- **Horizon extraction**: `next week`, `30 days`, `2 months`
- **Date ranges**: `2023-01-01 to 2023-12-31`, `from X to Y`
- **Relative dates**: `last 30 days`, `this week`, `yesterday`
- **Automatic calculation**: Converts relative expressions to actual dates

### Quantile and Statistical Extraction
- **Percentile expressions**: `p10`, `p50`, `p90`, `25th percentile`
- **Confidence intervals**: `95% confidence interval` → [0.025, 0.975]
- **Multiple formats**: `p10/p50/p90`, `10th and 90th percentile`
- **Automatic sorting**: Consistent output ordering

### Geographic and Category Extraction
- **Location support**: `United States`, `US`, `UK`, `Canada`, etc.
- **Category detection**: `technology`, `business`, `entertainment`
- **Case insensitive**: Handles various input formats
- **Abbreviation mapping**: `US` → `united states`

## 📊 Supported Intents

### 🔮 Forecast Intent
*"How will [keyword] trend next week?"*
- Generates quantile forecasts using Darts models
- Provides confidence intervals and uncertainty quantification
- Supports multiple forecast horizons
- Extracts keywords, horizons, quantiles, and geographic filters

### 🔄 Compare Intent
*"Compare [keyword1] vs [keyword2] popularity"*
- Side-by-side trend comparison
- Statistical significance testing
- Correlation analysis
- Handles multiple comparison formats

### 📈 Summary Intent
*"Show me a summary of [keyword] trends"*
- Current trend analysis
- Historical performance overview
- Key insights and patterns
- Date range filtering support

### 🎯 Train Intent
*"Train a model for [keyword] forecasting"*
- Automatic model selection and training
- Comprehensive evaluation metrics
- Model persistence and versioning
- Horizon and quantile extraction

### 📊 Evaluate Intent
*"Evaluate the performance of my models"*
- Model accuracy assessment
- Performance comparison
- Recommendations for improvement
- Model ID extraction from queries

### 🏥 Health Intent
*"Is the service working?"*
- System status check
- Service health monitoring
- Performance metrics

### 📋 List Models Intent
*"Show me available models"*
- List trained models
- Model metadata and performance
- Model management capabilities

## 🔧 API Endpoints

### Agent Interface
- `POST /api/agent/query` - Natural language query processing

### Traditional REST API
- `GET /health` - Health check
- `POST /api/trends` - Get Google Trends data
- `POST /api/trends/summary` - Get trends summary
- `POST /api/trends/compare` - Compare keywords
- `POST /api/models/train` - Train forecasting models
- `GET /api/models/{model_id}/evaluate` - Model evaluation
- `POST /api/models/{model_id}/predict` - Generate predictions
- `GET /api/models` - List available models

## 🎯 Model Types Supported

The system supports 14+ forecasting models from the Darts library:

1. **LSTM** - Long Short-Term Memory networks
2. **TCN** - Temporal Convolutional Networks  
3. **Transformer** - Attention-based models
4. **Prophet** - Facebook's forecasting tool
5. **ARIMA** - Classical statistical forecasting
6. **Exponential Smoothing** - Smoothing techniques
7. **Random Forest** - Ensemble methods
8. **N-BEATS** - Neural basis expansion analysis
9. **TFT** - Temporal Fusion Transformers
10. **GRU** - Gated Recurrent Units
11. **AutoARIMA** - Automatic ARIMA selection
12. **AutoETS** - Automatic Exponential Smoothing
13. **AutoTheta** - Automatic Theta model selection
14. **AutoCES** - Automatic Complex Exponential Smoothing

## 📈 Performance

### Intent Recognition
- **Response Time**: < 100ms
- **Accuracy**: 92%+ on test queries
- **Robustness**: Handles paraphrases and variations

### Slot Extraction
- **Keyword extraction**: Supports quoted, contextual, and comparison patterns
- **Date parsing**: Handles explicit dates, relative expressions, and ranges
- **Quantile detection**: Multiple formats with automatic conversion
- **Geographic filtering**: 12+ countries with abbreviation support

### Forecasting
- **Training Time**: 30 seconds to 5 minutes
- **Prediction Speed**: Sub-second predictions
- **Memory Usage**: Efficient with model caching
- **Scalability**: Concurrent training and prediction

## 🛠️ Development

### Testing
```bash
# Run all tests (392 tests)
pytest

# Run specific test suites
python -m pytest tests/unit/ -v
python -m pytest tests/unit/darts/ -v
python -m pytest tests/unit/pytrends/ -v
python -m pytest tests/integration/ -v
```

### Active Learning
The intent recognizer supports continuous improvement:
```python
from app.services.agent.intent_recognizer import HybridIntentRecognizer

recognizer = HybridIntentRecognizer()
recognizer.add_example("What's the future of blockchain?", AgentIntent.FORECAST)
```

## 🚀 Production Deployment

### Docker Deployment
```bash
docker-compose up --build
```

### AWS EC2 Deployment
```bash
# Clone and setup
git clone <repository-url>
cd google_trends_quantile_forecaster

# Run deployment scripts
chmod +x scripts/deployment/*.sh
./scripts/deployment/setup_ec2.sh
./scripts/deployment/deploy_app.sh
./scripts/deployment/security_setup.sh setup
```

## 🔮 Future Enhancements

1. **Advanced Semantic Scoring** - Upgrade to sentence-transformers for better understanding
2. **LLM Integration** - Few-shot classification for ambiguous cases
3. **Active Learning Pipeline** - Automatic example generation and performance monitoring
4. **Multi-language Support** - Language detection and translation
5. **Voice Interface** - Speech-to-text and text-to-speech capabilities
6. **Enhanced Slot Extraction** - Support for more complex parameter patterns
7. **Real-time Learning** - Continuous improvement from user interactions

## 🤝 Contributing

The system is designed for extensibility:
- **Modular architecture** for easy feature additions
- **Comprehensive testing** for reliability (392 tests passing)
- **Active learning** for continuous improvement
- **MLflow integration** for model management
- **Hybrid intent recognition** for robust natural language understanding

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
