# Stock Recommendation System Backend

This backend service provides multiple AI analysis and explainability features for the stock recommendation system, including FinBERT sentiment analysis, Mistral large language model investment recommendations, and semantic text analysis.

## Main Features

- FinBERT Financial Domain Sentiment Analysis
  - Integrated Gradients explainability analysis: Calculates each word's contribution to sentiment prediction
  - Self-Attention weight analysis: Visualizes the connection strength between words
  - Single text rapid analysis API: Provides real-time sentiment evaluation for financial texts

- Mistral Large Language Model Investment Recommendations
  - Tree-SHAP-based explainability analysis: Quantifies the impact of different financial indicators on investment advice
  - Asynchronous task processing mechanism: Handles long-running analysis requests
  - Structured financial data analysis: Evaluates key metrics such as revenue growth, net income, debt-to-equity ratio, etc.

- Semantic Text Analysis
  - Sentence semantic centrality calculation: Identifies the most representative sentences in a text
  - Financial domain keyword extraction: Highlights technical terms and their importance
  - Text importance scoring: Comprehensively evaluates sentence relevance to the overall document

## System Requirements

- Python 3.8+
- At least 4GB memory (8GB+ recommended)
- Supports CPU operation mode (no GPU required)

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

Using a virtual environment is optional but recommended to avoid package conflicts:

```bash
# Optional: Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate.bat  # Windows

# Then install dependencies
pip install -r requirements.txt
```

## Environment Variable Configuration

Create a `.env` file or directly set the following environment variables:

```
# Mistral API Configuration
MISTRAL_API_KEY=your_mistral_api_key
HF_API_URL=https://api-inference.huggingface.co/models
MISTRAL_MODEL=mistralai/Mistral-7B-Instruct-v0.1
```

## Running the Service

### Windows System (Recommended Method)

#### For Systems Without GPU (or when not using GPU)

Use the provided CPU mode batch script (works with both base Python or virtual environment):

```bash
# CPU mode startup (recommended for most users)
start_cpu.bat
```

This script automatically sets the appropriate environment variables and starts the service in CPU mode, using your current Python environment (whether it's base Python or a virtual environment).

#### For Systems With GPU

If you have a compatible NVIDIA GPU and want to use it for faster inference:

```bash
# GPU mode startup
start_backend.bat
```

### Other Startup Methods

```bash
# Manual startup (will use whatever device is available)
python app.py
```

By default, the service will run on http://localhost:5000

## Troubleshooting Dependency Issues

If you encounter dependency compatibility issues during installation or when running the application, here are some solutions:

### Common Problems and Solutions

1. **PyTorch Installation Fails**:
   - Try installing PyTorch separately first, following the specific instructions for your system from the [official PyTorch website](https://pytorch.org/get-started/locally/)
   - For example: `pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu` (for CPU-only version)

2. **Version Conflicts**:
   - If you see "cannot satisfy requirement" errors, try installing packages one by one:
     ```bash
     pip install flask==2.3.3 flask-cors==4.0.0
     pip install transformers==4.36.2
     pip install torch==2.2.0
     # and so on with other packages
     ```

3. **CUDA Version Mismatch**:
   - If you have GPU but see "CUDA version mismatch" errors, install the specific PyTorch version compatible with your CUDA:
     ```bash
     # For CUDA 11.8 example
     pip install torch==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
     ```

4. **Memory Issues During Installation**:
   - For systems with limited RAM, install memory-intensive packages individually:
     ```bash
     pip install transformers==4.36.2
     pip install sentence-transformers==2.2.2
     ```

### Creating a Minimal Environment

If you're having persistent issues, try this minimal setup that should work on most systems:

```bash
# Create a separate environment for testing
python -m venv minimal_env
source minimal_env/bin/activate  # or minimal_env\Scripts\activate.bat on Windows

# Install minimal requirements
pip install flask==2.3.3 flask-cors==4.0.0
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.30.2
pip install pandas matplotlib scikit-learn
pip install python-dotenv requests
pip install sentence-transformers==2.2.2

# Start in CPU mode
python app.py
```

### Last Resort: Using Docker

If dependency issues persist, using Docker is the most reliable solution as it provides an isolated environment with all dependencies pre-configured:

```bash
docker build -t stock-recommendation-backend .
docker run -p 5000:5000 stock-recommendation-backend
```

## Technical Implementation Details

### 1. FinBERT Sentiment Analysis

The FinBERT model is specifically trained for financial domain texts, capable of understanding professional terminology in financial reports, news, and analyst comments. Our implementation includes:

- **Integrated Gradients**: Quantifies each word's contribution to the final sentiment prediction by comparing the gradient path between a baseline input (zero embeddings) and the actual input
- **Self-Attention Analysis**: Extracts the attention matrix from the last layer of the Transformer, showing the connection strength between words
- **Fallback Mechanism**: Automatically switches to the perturbation method as an alternative explanation approach when the primary explanation method fails

### 2. Mistral Financial Analysis

Utilizes the Mistral-7B-Instruct large language model to analyze a company's financial condition and provide investment recommendations:

- **Perturbation Sample Generation**: Creates a sample space by controlling changes to financial indicator values
- **XGBoost Proxy Model**: Serves as an explainable alternative to the Mistral model for applying TreeSHAP analysis
- **Contribution Value Visualization**: Quantifies the impact of indicators such as revenue growth, net income growth, and current ratio on the final recommendation

### 3. Semantic Text Analysis

Performs in-depth analysis of investment recommendation texts to identify key information:

- **Vectorized Semantic Similarity**: Uses sentence-transformers to calculate similarity and centrality between sentences
- **Professional Vocabulary Weighting**: Assigns higher weights to sentences containing key terms based on a financial domain dictionary
- **Lightweight Implementation**: Designed to run efficiently even in CPU environments

## API Endpoints

### 1. FinBERT Sentiment Analysis API

#### POST /api/explain-single-text

Analyzes the sentiment of a single text and provides explainability data

**Request Format**:
```json
{
  "text": "The company reported strong growth in Q3, with earnings exceeding expectations."
}
```

**Response Format**:
```json
{
  "sentiment": "Positive",
  "sentimentScore": 0.75,
  "explainability": {
    "tokens": ["strong", "growth", "exceeding", "expectations"],
    "importanceValues": [0.85, 0.76, 0.68, 0.65],
    "attentionMatrix": [...],
    "importanceError": null
  },
  "methodInfo": {
    "method": "IntegratedGradients",
    "description": "Using integrated gradients to calculate feature importance"
  }
}
```

### 2. Mistral Model Recommendation API

#### POST /api/explain-mistral/treeshap

Analyzes financial data and provides explainability analysis for investment recommendations

**Request Format**:
```json
{
  "financialData": {
    "revenueGrowth": 15.2,
    "netIncomeGrowth": 10.5,
    "currentRatio": 1.8,
    "debtToEquity": 0.6,
    "returnOnEquity": 12.3,
    "peRatio": 18.5
  }
}
```

#### POST /api/explain-mistral/treeshap-async

Asynchronous version of the financial data analysis API, suitable for longer-running requests

**Response Format**:
```json
{
  "taskId": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### GET /api/tasks/<task_id>

Retrieves the status and results of an asynchronous task

**Response Format**:
```json
{
  "status": "completed",
  "progress": {
    "progress": 100,
    "message": "Task completed"
  },
  "result": {
    "shapValues": {...},
    "baseValue": 0.0,
    "finalValue": 0.65
  }
}
```

### 3. Semantic Text Analysis API

#### POST /api/analyze-text

Analyzes important sentences and keywords in text

**Request Format**:
```json
{
  "text": "The financial report details...",
  "section": "summary"
}
```

**Response Format**:
```json
{
  "success": true,
  "section": "summary",
  "sentences": [
    {
      "text": "The company reported strong financial results.",
      "score": 2.5,
      "centrality": 0.8,
      "keyword_weight": 0.7
    }
  ],
  "importantWords": [
    {"word": "financial", "importance": 0.9},
    {"word": "strong", "importance": 0.8}
  ]
}
```

### 4. Health Check API

#### GET /health

Checks the backend service status

**Response Format**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "tokenizer_loaded": true,
  "explainer_available": true,
  "device": "cpu",
  "gpu_available": false
}
```

## Performance Considerations

- **CPU Mode**: The system is designed to run in pure CPU environments, but some AI model inference may be slower
- **Memory Usage**: Loading models for the first time will consume significant memory; ensure your system has sufficient available memory
- **API Limitations**: Calls to external APIs (such as Hugging Face) may be subject to rate limits; the system implements retry and degradation strategies

## Docker Support

You can run this service using a Docker container:

```bash
docker build -t stock-recommendation-backend .
docker run -p 5000:5000 stock-recommendation-backend
```

## Main Component Description

- **app.py**: Main application server and API routing, implements Flask service and asynchronous task management
- **finbert_explainer.py**: FinBERT model and explainability analysis implementation, including integrated gradients and attention visualization
- **mistral_treeshap_explainer.py**: Mistral model and Tree-SHAP analysis implementation, including sample generation and proxy model training
- **semantic_text_analyzer.py**: Semantic text analysis functionality implementation, used to extract key sentences and terms
- **financial_domain_data.py**: Financial domain professional vocabulary data, supporting terminology weight configuration for semantic analysis

## System Integration

This backend service provides complete analysis functionality for the frontend, forming the core part of the stock recommendation system:
- Sentiment analysis module evaluates news and social media data
- Financial analysis module evaluates company fundamentals and financial indicators
- Strategy analysis module evaluates technical indicators and backtesting results

The results of these three modules are combined to generate the final investment recommendation.

## Technical References

- FinBERT: [https://huggingface.co/yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone)
- Tree-SHAP: [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
- Mistral: [https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) 