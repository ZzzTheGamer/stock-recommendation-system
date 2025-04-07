"""
Financial domain vocabulary and weight configuration
Contains financial indicator terms and important prompt words for text analysis
"""

# Financial indicators and their weights
financial_indicators = {
    # Basic financial metrics
    "revenue growth": 0.7, "net income": 0.8, "revenue": 0.6, "sales": 0.6,
    "profit": 0.7, "return rate": 0.7, "return": 0.6, "earnings": 0.7, 
    "net income growth": 0.8, "performance": 0.6, "financial performance": 0.7,
    
    # Liquidity and debt indicators
    "debt": 0.6, "equity": 0.6, "liquidity": 0.5, "ratio": 0.4,
    "liability": 0.6, "debt to equity": 0.7, "PE ratio": 0.7, "current ratio": 0.7,
    "debt financing": 0.7, "liquidity position": 0.7, "debt to equity ratio": 0.8,
    
    # Investment return indicators
    "shareholder": 0.5, "capital": 0.6, "ROE": 0.8, "investment": 0.5,
    "assets": 0.5, "financing": 0.5, "stock price": 0.5, "valuation": 0.5,
    "return on equity": 0.8, "generating profits": 0.7, "industry": 0.5,
    "operations": 0.4, "major player": 0.5,
    
    # Number markers (indicating potential numerical information)
    "%": 0.4, "billion": 0.3, "million": 0.3, "dollars": 0.3
}

# Prompt words and their weights
prompt_words = {
    # Positive evaluation words
    "significant": 0.5, "strong": 0.5, "growth": 0.5, "increase": 0.4,
    "positive": 0.5, "stable": 0.5, "excellent": 0.5, "advantage": 0.5,
    "good": 0.4, "outstanding": 0.4, "superior": 0.4, "higher than": 0.4,
    "robust": 0.6, "major": 0.5, "primarily": 0.3,
    
    # Negative evaluation words
    "decline": 0.5, "decrease": 0.4, "reduce": 0.4, "deteriorate": 0.5,
    "risk": 0.6, "challenge": 0.5, "problem": 0.4, "difficulty": 0.4,
    "concern": 0.5, "weaker than": 0.4, "lower than": 0.4, "disadvantage": 0.5,
    "tight": 0.5, "heavily": 0.5, "reliant on": 0.4, "somewhat": 0.3,
    
    # Recommendation/prediction words
    "recommend": 0.7, "suggest": 0.7, "expect": 0.6, "predict": 0.6,
    "may": 0.4, "should": 0.6, "need": 0.5, "opportunity": 0.6,
    "outlook": 0.6, "prospect": 0.6, "trend": 0.5, "development": 0.4,
    "indicates": 0.6, "suggests": 0.6
}

# Special phrases and their weights
special_phrases = {
    "significant growth": 0.8,
    "major decline": 0.8,
    "strong growth": 0.8,
    "high return rate": 0.8,
    "low debt ratio": 0.7,
    "high risk": 0.7,
    "strongly recommend": 0.9,
    "not recommended": 0.9,
    "hold position": 0.7,
    "robust performance": 0.8,
    "significant profits": 0.8,
    "tight liquidity": 0.7,
    "heavily reliant": 0.7,
    "strong return": 0.8,
    "current ratio": 0.7,
    "return on equity": 0.8,
    "debt to equity ratio": 0.8
} 