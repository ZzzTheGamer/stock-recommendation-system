# Stock Recommendation System

A comprehensive financial analysis and stock recommendation platform with explainable AI features. This system combines sentiment analysis of financial news, fundamental financial metrics evaluation, and technical trading strategy analysis to provide holistic investment recommendations.

## Overview

The Stock Recommendation System consists of two main components:

1. **Frontend** - Next.js React application that provides the user interface
2. **Backend** - Flask-based API server that powers the AI analysis components

The system analyzes stocks through multiple perspectives:
- **Sentiment Analysis**: Evaluates news and social media content using FinBERT
- **Financial Analysis**: Assesses company fundamentals using Mistral LLM and TreeSHAP
- **Strategy Analysis**: Tests trading strategies with historical price data
- **Comprehensive Recommendation**: Combines all analyses to provide an explainable investment suggestion

## Key Features

### Frontend Features
- Interactive stock price chart with historical data
- Real-time stock information display
- Financial news aggregation and sentiment analysis
- Fundamental financial metrics visualization
- Technical analysis and strategy backtesting
- Explainable AI elements that clarify recommendation reasoning

### Backend Features
- FinBERT sentiment analysis with Integrated Gradients and Self-Attention explainability
- Mistral large language model financial analysis with TreeSHAP explainability
- Semantic text analysis for highlighting key information
- Asynchronous task processing for complex analyses
- Comprehensive API for all analytical functions

## System Architecture

```
stock-recommendation-system/
├── backend/                 # Python Flask backend
│   ├── app.py               # Main Flask application
│   ├── finbert_explainer.py # FinBERT sentiment analysis
│   ├── mistral_treeshap_explainer.py # Financial analysis with Mistral
│   ├── semantic_text_analyzer.py # Text analysis utilities
│   ├── financial_domain_data.py # Financial domain vocabulary
│   ├── requirements.txt     # Python dependencies
│   ├── start_cpu.bat        # Script to start backend in CPU mode
│   └── start_backend.bat    # Script to start backend with GPU
│
├── src/                     # Frontend Next.js source
│   ├── app/                 # Next.js App Router
│   │   ├── layout.tsx       # Main layout
│   │   ├── page.tsx         # Main page
│   │   └── globals.css      # Global styles
│   ├── components/          # React components
│   └── services/            # API services
│
├── components/              # Additional React components
├── public/                  # Static assets
├── package.json             # Frontend dependencies
├── next.config.js           # Next.js configuration
├── tailwind.config.js       # Tailwind CSS configuration
└── start.bat                # Script to start frontend
```

## Prerequisites

- **Node.js** (v14+) for frontend
- **Python** (3.8+) for backend
- At least 4GB RAM (8GB+ recommended)
- Internet connection for API calls

## Installation and Setup

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd stock-recommendation-system/backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the backend directory with the following content:
   ```
   # Mistral API Configuration
   MISTRAL_API_KEY=your_mistral_api_key
   HF_API_URL=https://api-inference.huggingface.co/models
   MISTRAL_MODEL=mistralai/Mistral-7B-Instruct-v0.1
   ```

### Frontend Setup

1. Navigate to the project root directory:
   ```bash
   cd stock-recommendation-system
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
   NEWS_API_KEY=your_news_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Running the Application

### Step 1: Start the Backend Server

Navigate to the backend directory and run:

```bash
# For systems without GPU (recommended for most users)
start_cpu.bat

# OR if you have a compatible GPU
start_backend.bat
```

The backend server will start on http://localhost:5000

### Step 2: Start the Frontend Application

In a new terminal window, navigate to the project root directory and run:

```bash
npm run dev
```

The frontend application will be available at http://localhost:3000

### Using the Application

1. Open your browser and navigate to http://localhost:3000
2. Enter a stock symbol (e.g., AAPL, MSFT, GOOGL) in the search box
3. View the stock information, news, and analysis tabs
4. Run the various analyses by clicking on their respective tabs
5. View the final recommendation that combines all analyses

## Troubleshooting

### Backend Issues

If you encounter dependency compatibility issues with the backend:

1. **PyTorch Installation Issues**: Try installing PyTorch separately using:
   ```bash
   pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Version Conflicts**: Install packages individually:
   ```bash
   pip install flask==2.3.3 flask-cors==4.0.0
   pip install transformers==4.36.2
   pip install torch==2.2.0
   # Continue with other packages
   ```

3. **Minimal Working Setup**:
   ```bash
   pip install flask==2.3.3 flask-cors==4.0.0
   pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
   pip install transformers==4.30.2
   pip install pandas matplotlib scikit-learn
   pip install python-dotenv requests
   pip install sentence-transformers==2.2.2
   ```

### Frontend Issues

1. **Next.js Build Errors**: Try clearing the Next.js cache:
   ```bash
   rm -rf .next
   npm run dev
   ```

2. **API Connection Issues**: Ensure the backend is running before starting the frontend

3. **Missing Environment Variables**: Verify all API keys are correctly set in the .env file

## System Components and Technical Details

### Backend Components

- **FinBERT Sentiment Analysis**: Uses the FinBERT model to analyze financial text sentiment with Integrated Gradients and Self-Attention explainability
- **Mistral Financial Analysis**: Leverages Mistral-7B-Instruct LLM with TreeSHAP explainability to evaluate financial metrics
- **Semantic Text Analysis**: Employs sentence similarity and domain-specific weighting to identify important information
- **Asynchronous Task Processing**: Handles long-running analyses without blocking the API

### Frontend Components

- **Stock Information Module**: Displays core stock data and interactive price charts
- **News Analysis Module**: Shows news articles with sentiment analysis results
- **Financial Metrics Module**: Visualizes key financial indicators with explanations
- **Strategy Backtesting Module**: Tests and visualizes technical trading strategies
- **Recommendation Engine**: Combines all analyses to produce a final investment recommendation

## Docker Support

For containerized deployment:

```bash
# Build and run backend container
cd backend
docker build -t stock-recommendation-backend .
docker run -p 5000:5000 stock-recommendation-backend

# Build and run frontend container
cd ..
docker build -t stock-recommendation-frontend .
docker run -p 3000:3000 stock-recommendation-frontend
```

Or use docker-compose to run both services:

```bash
docker-compose up
```

## Technical References

- FinBERT: [https://huggingface.co/yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone)
- Tree-SHAP: [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
- Mistral: [https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- Next.js: [https://nextjs.org/](https://nextjs.org/)
- Flask: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/) 