# Trend Forecasting Agent

An intelligent forecasting system that combines natural language understanding with state-of-the-art time series forecasting to provide quantile predictions for keyword popularity trends.

## 🤖 Agentic Interface with LangGraph

The system features a **powerful agentic interface** built with LangGraph that understands natural language queries and orchestrates complex workflows. Users can interact with the system using conversational language, making it accessible to non-technical users while still supporting traditional API endpoints for developers.

### Natural Language Interface

Ask questions like:
- *"How will machine learning trend next week?"*
- *"Compare artificial intelligence vs data science popularity"*
- *"Train a model for blockchain forecasting"*
- *"Evaluate the performance of my models"*
- *"Show me a summary of current AI trends"*
- *"Forecast 'python programming' trends in United States for next month with p10/p50/p90"*

The system automatically:
1. **Recognizes your intent** using advanced natural language processing
2. **Extracts relevant information** from your query (keywords, time horizons, quantiles, locations)
3. **Orchestrates complex workflows** using LangGraph for reliable execution
4. **Returns natural language responses** with actionable insights

### Dual Interface Support

- **Conversational Interface**: Perfect for non-technical users who want to interact naturally
- **Traditional REST API**: Available for developers who prefer structured endpoints

## 🏗️ Architecture Overview

### Core Components

```
app/
├── agent_graph/                  # LangGraph orchestration
│   ├── graph.py                  # Main workflow graph
│   ├── nodes.py                  # Individual processing nodes
│   ├── state.py                  # State management
│   └── service_client.py         # Service integration
├── api/
│   ├── agent_routes.py           # LangGraph agent interface
│   └── routes.py                 # Traditional REST API endpoints
├── services/
│   ├── agent/                    # LangGraph-based agent service
│   │   ├── agent_service.py
│   │   ├── intent_recognizer.py
│   │   ├── slot_extractor.py
│   │   └── validators.py
│   ├── darts/                    # Time series forecasting
│   │   ├── training_service.py
│   │   ├── evaluation_service.py
│   │   └── prediction_service.py
│   └── pytrends/                 # Google Trends data
│       └── trends_service.py
├── models/
│   ├── agent_models.py           # Agent data models
│   ├── darts/                    # Forecasting models
│   └── pytrends/                 # Trends data models
├── utils/
│   ├── text_normalizer.py        # Advanced text processing
│   ├── error_handlers.py         # Error management
│   ├── monitoring_service.py     # System monitoring
│   └── security_service.py       # Security and audit
└── config/
    ├── agent_config.py           # Agent configuration
    └── config.py                 # General configuration
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
   curl -X POST http://localhost:5000/agent/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "How will AI trend next week?"}'
   ```

### Docker Development

```bash
docker-compose up --build
```

## 🔌 HTTP Client Usage

The system supports both in-process and HTTP clients for service communication:

### In-Process Client (Default)
```python
from app.agent_graph.service_client import InProcessForecasterClient
from app.services.forecaster_interface import ForecasterServiceInterface

service = ForecasterServiceInterface()
client = InProcessForecasterClient(service)
result = client.health()
```

### HTTP Client
```python
from app.agent_graph.service_client import HTTPForecasterClient

client = HTTPForecasterClient("http://localhost:5000")
result = client.health()
```

### Configuration
HTTP client behavior can be configured via environment variables:
- `FORECASTER_API_URL`: Base URL for the API
- `FORECASTER_API_KEY`: API key for authentication
- `HTTP_TIMEOUT`: Request timeout in seconds
- `HTTP_MAX_RETRIES`: Maximum number of retries
- `HTTP_LOG_REQUESTS`: Enable request logging

## 🧠 LangGraph-Powered Intelligence

The system uses **LangGraph** for robust workflow orchestration:

### Workflow Orchestration
- **State Management**: Maintains context throughout complex workflows
- **Error Handling**: Graceful failure recovery and user-friendly error messages
- **Parallel Processing**: Efficient handling of multiple operations
- **Validation**: Comprehensive input validation at each step

### Processing Pipeline
1. **Query Normalization**: Standardizes input for consistent processing
2. **Intent Recognition**: Identifies user intent with high accuracy
3. **Slot Extraction**: Extracts parameters (keywords, dates, quantiles)
4. **Planning**: Determines optimal execution strategy
5. **Execution**: Orchestrates service calls and data processing
6. **Response Formatting**: Delivers natural language responses

## 🔍 Advanced Slot Extraction

The system intelligently extracts parameters from natural language queries:

### Keyword Extraction
- **Quoted keywords**: `"machine learning"` and `'artificial intelligence'`
- **Comparison patterns**: `python vs javascript`, `AI versus ML`
- **Contextual extraction**: `for machine learning`, `about data science`
- **Individual word extraction**: Breaks down multi-word phrases

### Time and Date Extraction
- **Horizon extraction**: `next week`, `30 days`, `2 months`
- **Date ranges**: `2024-01-01 to 2024-12-31`, `from X to Y`, `between X and Y`
- **Relative dates**: `last 30 days`, `this week`, `yesterday`
- **Automatic calculation**: Converts relative expressions to actual dates

### Quantile and Statistical Extraction
- **Percentile expressions**: `p10`, `p50`, `p90`, `25th percentile`
- **Percentage notation**: `90%`, `90 percent`
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
- `POST /agent/ask` - Natural language query processing
- `GET /agent/health` - Agent health check
- `GET /agent/capabilities` - Get supported capabilities

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

The system supports 13+ forecasting models from the Darts library:

1. **LSTM** - Long Short-Term Memory networks
2. **TCN** - Temporal Convolutional Networks  
3. **Transformer** - Attention-based models
4. **ARIMA** - Classical statistical forecasting
5. **Exponential Smoothing** - Smoothing techniques
6. **Random Forest** - Ensemble methods
7. **N-BEATS** - Neural basis expansion analysis
8. **TFT** - Temporal Fusion Transformers
9. **GRU** - Gated Recurrent Units
10. **AutoARIMA** - Automatic ARIMA selection
11. **AutoETS** - Automatic Exponential Smoothing
12. **AutoTheta** - Automatic Theta model selection
13. **AutoCES** - Automatic Complex Exponential Smoothing

## 🛡️ Advanced Infrastructure Features

### Text Normalization System
- **Dual-view normalization**: Loose (preserves case/punctuation) and strict (casefolded/trimmed)
- **Unicode support**: Full-width digits, emoji handling, link protection
- **Caching system**: LRU cache with configurable size and statistics
- **Performance optimization**: Idempotent operations with fast-path caching

### Reliability & Resilience
- **Circuit breaker patterns**: Automatic failure detection and recovery
- **Retry mechanisms**: Exponential backoff with jitter
- **Rate limiting**: Token bucket algorithm with multi-tenant support
- **Error handling**: Structured error responses with debugging information

### Monitoring & Observability
- **Request tracking**: Unique request IDs and context management
- **Structured logging**: Comprehensive audit trails and debugging
- **Health checks**: System and component health monitoring
- **Metrics collection**: Performance and usage analytics

### Security Features
- **Data redaction**: Automatic sensitive data masking in logs
- **Audit logging**: Comprehensive security event tracking
- **Input validation**: Robust sanitization and validation
- **Rate limiting**: Protection against abuse and overload

## 📈 Performance

### Intent Recognition
- **Response Time**: < 100ms
- **Accuracy**: 92%+ on test queries
- **Robustness**: Handles paraphrases and variations
- **Hybrid scoring**: Semantic, regex, and LLM ensemble

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

### Text Processing
- **Normalization speed**: < 1ms per query with caching
- **Cache hit rate**: 85%+ for repeated queries
- **Memory efficiency**: Optimized data structures and algorithms

## 🛠️ Development

### Testing
```bash
# Run all tests
pytest

# Run specific test suites
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v

# Test coverage
python -m pytest --cov=app tests/
```

### Active Learning
The intent recognizer supports continuous improvement:
```python
from app.services.agent.intent_recognizer import IntentRecognizer

recognizer = IntentRecognizer()
recognizer.add_example("What's the future of blockchain?", AgentIntent.FORECAST)
```

### Text Normalizer Features
```python
from app.utils.text_normalizer import normalize_views, TextNormalizer

# Dual-view normalization
loose, strict, stats = normalize_views("Hello, World! 🔥")

# Caching and statistics
normalizer = TextNormalizer()
cache_info = normalizer.get_cache_info()
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
cd trend_forecasting_agent

# Run deployment scripts
chmod +x scripts/deployment/*.sh
./scripts/deployment/setup_ec2.sh
./scripts/deployment/deploy_app.sh
./scripts/deployment/security_setup.sh setup
```

## 🌐 Web Interface

The system includes a complete Angular web application:

### Features
- **Agent Chat Interface**: Natural language interaction
- **API Testing Tool**: Direct API endpoint testing
- **Real-time Updates**: Live data and status monitoring
- **Responsive Design**: Mobile and desktop optimized

### Development
```bash
cd ui/trend-forecasting-ui
npm install
ng serve
```

## 🔮 Recent Enhancements

### Text Normalizer System (Phase A & B)
- ✅ **Advanced text processing** with dual-view normalization
- ✅ **Unicode digit normalization** and emoji handling
- ✅ **Link protection** during edge trimming
- ✅ **Caching system** with LRU cache and statistics
- ✅ **Enhanced slot extraction** with optimized performance

### Intent Recognition Optimization (Phase C1)
- ✅ **Double normalization avoidance** for improved performance
- ✅ **Hybrid scoring system** with ensemble methods
- ✅ **Dynamic weight redistribution** for failed components
- ✅ **Comprehensive test coverage** with 29 intent recognizer tests

### Infrastructure Improvements
- ✅ **Circuit breaker patterns** for resilience
- ✅ **Advanced retry mechanisms** with exponential backoff
- ✅ **Multi-tenant rate limiting** with cost tracking
- ✅ **Comprehensive monitoring** and security services

## 🤝 Contributing

The system is designed for extensibility:
- **Modular architecture** for easy feature additions
- **Comprehensive testing** for reliability
- **Active learning** for continuous improvement
- **MLflow integration** for model management
- **LangGraph orchestration** for robust workflow management

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
