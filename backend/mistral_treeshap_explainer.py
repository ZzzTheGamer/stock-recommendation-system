import random
import time
import json
import logging
import numpy as np
import xgboost as xgb
import shap
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler
import os
import requests
import re
import pandas as pd
from dotenv import load_dotenv

# Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables
HF_API_URL = os.environ.get("HF_API_URL", "https://api-inference.huggingface.co/models")
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
HF_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MAX_API_RETRIES = 3
API_RETRY_DELAY = 2
# Increase API timeout setting
API_TIMEOUT = 60  # API call timeout (seconds)

# Financial indicator display names in Chinese
FEATURE_DISPLAY_NAMES = {
    "revenueGrowth": "Revenue Growth",
    "netIncomeGrowth": "Net Income Growth",
    "currentRatio": "Current Ratio",
    "debtToEquity": "Debt to Equity",
    "returnOnEquity": "Return on Equity",
    "peRatio": "P/E Ratio"
}

class MistralTreeSHAPExplainer:
    """
    Mistral model recommendation explainer combining tree model and SHAP explainer
    
    Use the perturbation method similar to LimeExplainer to generate samples,
    then use XGBoost as a proxy model,
    finally apply TreeSHAP analysis to feature contribution
    """
    def __init__(self):
        self.base_model_endpoint = f"{HF_API_URL}/{MISTRAL_MODEL}"
        self.api_key = HF_API_KEY
        self.api_call_count = 0
        self.use_real_api = True
        self.api_failures = 0
        
    def explain_recommendation(self, financial_data: Dict[str, float], recommendation: str = None) -> Dict[str, Any]:
        """
        Use TreeSHAP to explain Mistral model recommendations
        
        Args:
            financial_data: Financial indicator data dictionary
            recommendation: Model-generated recommendation (optional)
            
        Returns:
            Dictionary containing SHAP values and related data
        """
        try:
            self.api_call_count = 0
            self.api_failures = 0
            logger.info("Starting TreeSHAP analysis...")
            
            # 1. Create perturbed samples using the LimeExplainer sample generation method
            samples_data = self._generate_perturbation_samples(financial_data)
            
            if not samples_data["success"]:
                return {
                    "success": False,
                    "error": samples_data["error"]
                }
                
            # 2. Use XGBoost as a proxy model
            xgb_result = self._build_xgboost_model(
                samples_data["samples"], 
                samples_data["feature_names"],
                financial_data
            )
            
            if not xgb_result["success"]:
                return {
                    "success": False, 
                    "error": xgb_result["error"]
                }
                
            # 3. Apply TreeSHAP analysis
            shap_result = self._apply_treeshap(
                xgb_result["model"],
                xgb_result["X_train"],
                xgb_result["feature_names"],
                financial_data
            )
            
            # 4. Add final prediction value and base value
            final_value = xgb_result["base_prediction"]
            
            # 5. Build the return result
            result = {
                "success": True,
                "shapValues": shap_result["shapValues"],
                "baseValue": shap_result["baseValue"],
                "finalValue": final_value,
                "baselineScore": samples_data["base_score"],
                "baseRecommendation": samples_data["base_prediction"],
                "rSquared": xgb_result["r_squared"],
                "sampleCount": len(samples_data["samples"]),
                "featureSampleCounts": samples_data["feature_sample_counts"],
                "apiCallInfo": {
                    "totalCalls": self.api_call_count,
                    "failures": self.api_failures
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in TreeSHAP analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
            
    def _call_mistral_api(self, values: np.ndarray, feature_names: List[str]) -> str:
        """Call Mistral API to get investment recommendation score, return API response text"""
        retries = 0
        while retries < MAX_API_RETRIES:
            try:
                # Build input data
                input_data = {
                    "financialData": {
                        feature: float(value) for feature, value in zip(feature_names, values)
                    }
                }
                
                # Record input data
                logger.info(f"API request financial data: {json.dumps(input_data)}")
                
                # Build a stricter prompt
                prompt = f"""Based on the following financial metrics, analyze each factor and provide an investment recommendation score between -1.0 and 1.0:

Financial Data:
- Revenue Growth: {input_data['financialData']['revenueGrowth']}%
- Net Income Growth: {input_data['financialData']['netIncomeGrowth']}%
- Current Ratio: {input_data['financialData']['currentRatio']}
- Debt to Equity: {input_data['financialData']['debtToEquity']}
- Return on Equity: {input_data['financialData']['returnOnEquity']}%
- P/E Ratio: {input_data['financialData']['peRatio']}

Step 1: Evaluate each metric individually on a scale from -1.0 to 1.0:
- Revenue Growth: Negative (<0%) is negative score, moderate (0-15%) is neutral, strong (>15%) is positive score
- Net Income Growth: Negative (<0%) is negative score, moderate (0-20%) is neutral, strong (>20%) is positive score
- Current Ratio: Below 1.0 is negative score, 1.0-2.0 is neutral, above 2.0 is positive score
- Debt to Equity: Above 2.0 is negative score, 1.0-2.0 is neutral, below 1.0 is positive score
- Return on Equity: Below 5% is negative score, 5-15% is neutral, above 15% is positive score
- P/E Ratio: Above 30 or negative is negative score, 15-30 is neutral, below 15 is positive score

Step 2: Weight the metrics (Net Income Growth and ROE are most important)

Step 3: Calculate final investment score between -1.0 and 1.0 by summing the weighted scores.

Scoring Guide:
- 1.0: Strong Buy - Excellent financial health with strong growth and profitability
- 0.5: Moderate Buy - Good financial performance with some positive indicators
- 0.0: Hold - Balanced financial metrics, neither strong positive nor negative indicators
- -0.5: Moderate Sell - Concerning financial metrics with some negative indicators
- -1.0: Strong Sell - Poor financial health with negative growth and weak profitability

Your response format should be:
Final Score: [NUMBER BETWEEN -1.0 AND 1.0]"""

                # Record prompt
                logger.info(f"API prompt: {prompt[:200]}...")
                
                # Call API, add timeout setting
                logger.info(f"Calling Mistral API, set timeout to {API_TIMEOUT} seconds")
                response = requests.post(
                    self.base_model_endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"inputs": prompt},
                    timeout=API_TIMEOUT  # Set request timeout
                )
                
                self.api_call_count += 1
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        # Extract generated text
                        generated_text = result[0].get('generated_text', '')
                        # Record complete API response and score
                        logger.info(f"API returns complete text: {generated_text}")
                        score = self._recommendation_to_score(generated_text)
                        logger.info(f"Converted score: {score}")
                        return generated_text
                    else:
                        logger.error(f"API returns incorrect format: {result}")
                        self.api_failures += 1
                else:
                    logger.error(f"API call failed: {response.status_code}, {response.text}")
                    self.api_failures += 1
                
                # Retry
                retries += 1
                if retries < MAX_API_RETRIES:
                    logger.info(f"API call failed, retry {retries}/{MAX_API_RETRIES}...")
                    time.sleep(API_RETRY_DELAY)
            
            # Special handling for timeout exceptions
            except requests.exceptions.Timeout:
                logger.error(f"API call timeout, exceeds {API_TIMEOUT} seconds")
                self.api_failures += 1
                retries += 1
                if retries < MAX_API_RETRIES:
                    logger.info(f"API call timeout, retry {retries}/{MAX_API_RETRIES}...")
                    time.sleep(API_RETRY_DELAY)
            # Handle network connection exceptions
            except requests.exceptions.ConnectionError as e:
                logger.error(f"API network connection error: {str(e)}")
                self.api_failures += 1
                retries += 1
                if retries < MAX_API_RETRIES:
                    logger.info(f"Network connection error, retry {retries}/{MAX_API_RETRIES}...")
                    time.sleep(API_RETRY_DELAY * 2)  # Network errors increase longer waiting times
            except Exception as e:
                logger.error(f"Error calling API: {str(e)}")
                self.api_failures += 1
                retries += 1
                if retries < MAX_API_RETRIES:
                    logger.info(f"API call exception, retry {retries}/{MAX_API_RETRIES}...")
                    time.sleep(API_RETRY_DELAY)
        
        # If the API call fails after multiple retries, use the fallback prediction
        logger.warning("API call failed multiple times, using fallback prediction")
        return self._fallback_prediction_text(values, feature_names)
            
    def _recommendation_to_score(self, text: str) -> float:
        """
        Extract and convert investment recommendation score from API response
        
        Args:
            text: API response text
            
        Returns:
            Float score between -1.0 and 1.0
        """
        try:
            # Record original length
            original_length = len(text)
            
            # 1. Try to separate the original prompt and generated reply
            # Find the last "Note:" as the separator, usually the last part of the original prompt
            note_pos = text.rfind("Note:")
            if note_pos > 0:
                # Find the first newline after Note:
                newline_pos = text.find("\n", note_pos)
                if newline_pos > 0:
                    # Extract the text after Note: line, which should be the generated reply part
                    response_text = text[newline_pos:].strip()
                    if response_text:
                        logger.info(f"Extracted reply part ({len(response_text)} characters) from original text ({original_length} characters): {response_text}")
                        text = response_text
                    else:
                        logger.warning("Unable to extract a valid reply part, using the complete text")
            
            # 2. Find specific scoring patterns
            # Find common patterns, such as "Score: X.X" or "the score is X.X" etc.
            score_patterns = [
                r'Score:\s*([-+]?\d*\.\d+|\d+)',  # Score: X.X
                r'score\s+is\s+([-+]?\d*\.\d+|\d+)',  # score is X.X
                r'score\s*:\s*([-+]?\d*\.\d+|\d+)',  # score: X.X
                r'score\s+of\s+([-+]?\d*\.\d+|\d+)',  # score of X.X
                r'recommendation\s+score\s+is\s+([-+]?\d*\.\d+|\d+)',  # recommendation score is X.X
                r'recommendation\s+score\s*:\s*([-+]?\d*\.\d+|\d+)',  # recommendation score: X.X
                r'rating\s+of\s+([-+]?\d*\.\d+|\d+)',  # rating of X.X
                r'rating\s*:\s*([-+]?\d*\.\d+|\d+)',  # rating: X.X
                r'[-\s]+(\-?[01]\.\d+)[\s\.$]',  # Match formats like "- -0.5" or "- 0.5"
                r'[\s\-](\-?[01]\.\d+)[\s\.$]',  # Match formats like "-0.5" or "0.5", but ensure it is a separate number
            ]
            
            for pattern in score_patterns:
                matches = re.search(pattern, text, re.IGNORECASE)
                if matches:
                    try:
                        score = float(matches.group(1))
                        # Ensure the score is within the valid range
                        constrained_score = max(min(score, 1.0), -1.0)
                        logger.info(f"Extracted score using pattern '{pattern}': {score} -> {constrained_score}")
                        return constrained_score
                    except ValueError:
                        logger.warning(f"Unable to convert matched result '{matches.group(1)}' to a float")
            
            # 3. If specific patterns are not found, try to extract all numbers from the text
            logger.info("Specific scoring patterns not found, trying to extract all numbers from the text")
            numbers = re.findall(r'([-+]?\d*\.\d+|\d+)', text)
            logger.info(f"Extracted all numbers: {numbers}")
            
            # 3. Filter out possible scoring numbers (-1.0 to 1.0 range)
            valid_scores = [float(n) for n in numbers if -1.0 <= float(n) <= 1.0]
            logger.info(f"Numbers within valid range: {valid_scores}")
            
            if valid_scores:
                # Return the first valid score
                logger.info(f"Using the first valid number as the score: {valid_scores[0]}")
                return valid_scores[0]
                
            # 4. Analyze based on keywords
            logger.info("No valid numeric score found, trying to analyze based on keywords")
            text_lower = text.lower()
            
            # 4. Define a more comprehensive keyword-score mapping
            keyword_scores = [
                (["strong buy", "strongly recommended", "excellent opportunity"], 1.0),
                (["buy", "recommend buy", "positive outlook", "good opportunity"], 0.5),
                (["hold", "neutral", "maintain position", "fair", "average"], 0.0),
                (["sell", "recommend sell", "negative outlook", "poor outlook"], -0.5),
                (["strong sell", "strongly recommended", "urgent sell", "highly risky"], -1.0)
            ]
            
            # 5. Check which keywords appear in the text
            found_keywords = []
            for keywords, score in keyword_scores:
                for keyword in keywords:
                    if keyword in text_lower:
                        found_keywords.append((keyword, score))
            
            logger.info(f"Found keywords and corresponding scores: {found_keywords}")
            
            if found_keywords:
                # Use the extreme score (absolute value maximum) of the found keywords
                extreme_keyword = max(found_keywords, key=lambda x: abs(x[1]))
                logger.info(f"Using the keyword'{extreme_keyword[0]}' corresponding score: {extreme_keyword[1]}")
                return extreme_keyword[1]
            
            # 5. If all methods fail, use the default value
            logger.warning("Failed to extract a score from the text, using default value 0.0")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error converting recommendation score: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.0
    
    def _fallback_prediction_text(self, values: np.ndarray, feature_names: List[str]) -> str:
        """Fallback prediction method, return text prediction result"""
        # Convert to dictionary format
        data_dict = {feature: value for feature, value in zip(feature_names, values)}
        
        # Use simple rules to calculate the score
        score = 0
        
        # 1. Revenue growth
        if 'revenueGrowth' in data_dict:
            revenue_growth = data_dict['revenueGrowth']
            if revenue_growth > 15:
                score += 1.5
            elif revenue_growth > 5:
                score += 1.0
            elif revenue_growth > 0:
                score += 0.5
            elif revenue_growth < -10:
                score -= 1.5
            elif revenue_growth < 0:
                score -= 1.0
        
        # 2. Net income growth
        if 'netIncomeGrowth' in data_dict:
            net_income_growth = data_dict['netIncomeGrowth']
            if net_income_growth > 20:
                score += 2.0
            elif net_income_growth > 10:
                score += 1.5
            elif net_income_growth > 0:
                score += 0.7
            elif net_income_growth < -15:
                score -= 2.0
            elif net_income_growth < 0:
                score -= 1.2
        
        # 3. Current ratio
        if 'currentRatio' in data_dict:
            current_ratio = data_dict['currentRatio']
            if current_ratio > 3:
                score += 0.8
            elif current_ratio > 2:
                score += 1.0
            elif current_ratio > 1:
                score += 0.5
            elif current_ratio < 0.8:
                score -= 1.5
            elif current_ratio < 1:
                score -= 1.0
        
        # 4. Debt to equity ratio
        if 'debtToEquity' in data_dict:
            debt_to_equity = data_dict['debtToEquity']
            if debt_to_equity > 2.5:
                score -= 1.5
            elif debt_to_equity > 1.5:
                score -= 1.0
            elif debt_to_equity > 1.0:
                score -= 0.5
            elif debt_to_equity < 0.3:
                score += 1.0
            elif debt_to_equity < 0.6:
                score += 0.5
        
        # 5. Return on equity
        if 'returnOnEquity' in data_dict:
            roe = data_dict['returnOnEquity']
            if roe > 100:
                score += 0.5
            elif roe > 25:
                score += 1.5
            elif roe > 15:
                score += 1.0
            elif roe > 10:
                score += 0.5
            elif roe < 5:
                score -= 0.5
            elif roe < 0:
                score -= 1.5
        
        # 6. PE ratio
        if 'peRatio' in data_dict:
            pe_ratio = data_dict['peRatio']
            if pe_ratio < 0:
                score -= 1.5
            elif pe_ratio < 10:
                score += 1.5
            elif pe_ratio < 15:
                score += 1.0
            elif pe_ratio < 20:
                score += 0.3
            elif pe_ratio < 25:
                score -= 0.3
            elif pe_ratio < 30:
                score -= 0.8
            else:
                score -= 1.2
        
        # Normalize to -1 to 1
        normalized_score = min(max(score / 5.0, -1.0), 1.0)
        
        return f"Final Score: {normalized_score}" 

    def _generate_perturbation_samples(self, financial_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate perturbation samples
        Args:
            financial_data: Financial data dictionary
            
        Returns:
            Dictionary of generated samples and related information
        """
        try:
            # 1. Prepare baseline data
            base_data = financial_data.copy()
            
            # Record very extreme original values, which may need special handling
            extreme_values = {}
            for feature, value in base_data.items():
                # Detect extreme ROE values
                if feature == "returnOnEquity" and abs(value) > 10:
                    extreme_values[feature] = value
                    logger.info(f"Detected extreme {feature} value: {value}, will apply special perturbation strategy")
                # Detect extreme PE values
                elif feature == "peRatio" and abs(value) > 30:
                    extreme_values[feature] = value
                    logger.info(f"Detected extreme {feature} value: {value}, will apply special perturbation strategy")
                # Detect extreme net income growth
                elif feature == "netIncomeGrowth" and abs(value) > 30:
                    extreme_values[feature] = value
                    logger.info(f"Detected extreme {feature} value: {value}, will apply special perturbation strategy")
            
            # Define the features to analyze and their default values
            feature_names = [
                "revenueGrowth", 
                "netIncomeGrowth",
                "currentRatio",
                "debtToEquity",
                "returnOnEquity",
                "peRatio"
            ]
            
            # Get baseline prediction results
            baseline_values = np.array([base_data[feature] for feature in feature_names])
            base_prediction = self._call_mistral_api(baseline_values, feature_names)
            base_score = float(self._recommendation_to_score(base_prediction))
            
            # Create sample record list
            sample_records = []
            
            # Track the number of samples for each feature
            feature_sample_counts = {feature: 0 for feature in feature_names}
            
            # 2. Generate random perturbation samples
            num_samples = 15  # Reduce the sample size from 100 to 15
            
            # Target at least 2 samples per feature
            min_samples_per_feature = 2  # Reduce from 12 to 2 samples/feature
            
            # Define specific perturbation ranges and strategies for each feature
            feature_perturbation_strategies = {
                # Revenue growth: Can be positive or negative, allow a large range
                "revenueGrowth": {
                    "absolute": [-30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 25.0, 35.0, 50.0],
                    "relative": [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
                    "use_absolute": 0.4,  # 40% probability to use absolute value
                    "allow_negative": True,
                    "extremes": [-50.0, 80.0],  # Extreme value range
                    "importance": 0.8  # Importance assessment
                },
                # Net income growth: Can be positive or negative, allow a large range, particularly important
                "netIncomeGrowth": {
                    "absolute": [-100.0, -70.0, -40.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 40.0, 70.0, 100.0, 150.0, 200.0],
                    "relative": [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
                    "use_absolute": 0.6,  # Increase the probability of using absolute values
                    "allow_negative": True,
                    "extremes": [-150.0, 300.0],  # Significantly expand the extreme value range
                    "importance": 0.9  # Very high importance assessment
                },
                # Current ratio: Usually positive, 1.0 is the healthy value boundary
                "currentRatio": {
                    "absolute": [0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0],
                    "relative": [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0, 4.0],
                    "use_absolute": 0.6,
                    "allow_negative": False,  # Not allowed to be negative
                    "min_value": 0.05,  # Minimum value limit
                    "extremes": [0.01, 15.0],  # Extreme value range
                    "importance": 0.8  # High importance
                },
                # Debt to equity ratio: Usually positive, but also has extreme cases
                "debtToEquity": {
                    "absolute": [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0],
                    "relative": [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 6.0],
                    "use_absolute": 0.5,
                    "allow_negative": False,
                    "min_value": 0.0,
                    "extremes": [0.0, 20.0],  # Extreme value range
                    "importance": 0.8  # High importance
                },
                # Return on equity: Can be positive or negative, key financial indicator, particularly important
                "returnOnEquity": {
                    "absolute": [-150.0, -100.0, -70.0, -50.0, -30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0, 120.0, 200.0, 300.0],
                    "relative": [-0.95, -0.8, -0.7, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0],
                    "use_absolute": 0.8,  # Significantly increase the probability of using absolute values, especially for high ROE
                    "allow_negative": True,
                    "extremes": [-200.0, 400.0],  # Extreme value range
                    "importance": 1.0  # Highest importance assessment
                },
                # PE ratio: Usually positive, but negative for loss-making companies
                "peRatio": {
                    "absolute": [-200.0, -150.0, -100.0, -50.0, -30.0, -15.0, 5.0, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0, 80.0, 120.0, 200.0],
                    "relative": [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0],
                    "use_absolute": 0.8,  # Significantly increase the probability of using absolute values, the relative change of PE is not significant
                    "allow_negative": True,
                    "extremes": [-300.0, 400.0],  # Extreme value range
                    "importance": 0.75  # High importance
                }
            }
            
            # Adjust the perturbation strategy for extreme original values
            for feature, value in extreme_values.items():
                if feature == "returnOnEquity":
                    # For extreme ROE, use more extreme perturbation values
                    feature_perturbation_strategies[feature]["absolute"] = [
                        -400.0, -300.0, -200.0, -100.0, -50.0, -20.0, -10.0, 0.0, 10.0, 20.0, 50.0, 100.0, 
                        value*0.25, value*0.5, value*0.8, value, value*1.2, value*1.5, value*2.0, value*3.0
                    ]
                    feature_perturbation_strategies[feature]["extremes"] = [-500.0, value*5.0]
                    logger.info(f"Adjusted the perturbation range for extreme ROE({value})")
                
                elif feature == "peRatio":
                    # For extreme PE, use more extreme perturbation values
                    feature_perturbation_strategies[feature]["absolute"] = [
                        -300.0, -200.0, -100.0, -50.0, -20.0, -10.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 
                        value*0.25, value*0.5, value*0.8, value, value*1.2, value*1.5, value*2.0
                    ]
                    feature_perturbation_strategies[feature]["extremes"] = [-400.0, value*3.0]
                    logger.info(f"Adjusted the perturbation range for extreme PE({value})")
                    
                elif feature == "netIncomeGrowth":
                    # For extreme net income growth, use more extreme perturbation values
                    feature_perturbation_strategies[feature]["absolute"] = [
                        -200.0, -150.0, -100.0, -50.0, -20.0, -10.0, 0.0, 10.0, 20.0, 50.0, 100.0, 150.0, 
                        value*0.25, value*0.5, value*0.8, value, value*1.2, value*1.5, value*2.0, value*3.0
                    ]
                    feature_perturbation_strategies[feature]["extremes"] = [-300.0, value*4.0]
                    logger.info(f"Adjusted the perturbation range for extreme net income growth({value})")
            
            # First generate single-indicator perturbation samples - ensure each indicator has sufficient sample points
            for feature in feature_names:
                strategy = feature_perturbation_strategies[feature]
                # Generate samples focused on a single indicator
                for _ in range(min_samples_per_feature):
                    sample_data = base_data.copy()
                    
                    # Decide to use absolute value, relative value, or extreme value
                    perturb_type = random.choices(
                        ["absolute", "relative", "extreme"], 
                        weights=[strategy["use_absolute"], 1-strategy["use_absolute"], 0.3], 
                        k=1
                    )[0]
                    
                    if perturb_type == "absolute":
                        # Use absolute value replacement
                        perturbed_value = random.choice(strategy["absolute"])
                    elif perturb_type == "relative":
                        # Use relative multiplier
                        scale = random.choice(strategy["relative"])
                        if base_data[feature] == 0:
                            perturbed_value = random.choice(strategy["absolute"])
                        else:
                            perturbed_value = base_data[feature] * scale
                    else:  # extreme
                        # Use extreme value
                        if "extremes" in strategy:
                            # 50% probability to use low extreme value, 50% probability to use high extreme value
                            if random.random() < 0.5:
                                perturbed_value = strategy["extremes"][0]  # Low extreme value
                            else:
                                perturbed_value = strategy["extremes"][1]  # High extreme value
                        else:
                            # If no extreme value is defined, use the extreme values of the absolute range
                            abs_values = strategy["absolute"]
                            perturbed_value = abs_values[0] if random.random() < 0.5 else abs_values[-1]
                    
                    # Apply restriction conditions
                    if not strategy.get("allow_negative", True) and perturbed_value < 0:
                        perturbed_value = strategy.get("min_value", 0)
                    
                    # Apply minimum value limit
                    if "min_value" in strategy and perturbed_value < strategy["min_value"]:
                        perturbed_value = strategy["min_value"]
                    
                    # Set the perturbed value
                    sample_data[feature] = perturbed_value
                    
                    # Record the change of feature values
                    feature_values = {
                        feature: {
                            "original": base_data[feature],
                            "perturbed": perturbed_value,
                            "perturb_type": perturb_type
                        }
                    }
                    
                    # Get the prediction result
                    try:
                        sample_values = np.array([sample_data[f] for f in feature_names])
                        rec_result = self._call_mistral_api(sample_values, feature_names)
                        score = float(self._recommendation_to_score(rec_result))
                        
                        # Record the sample
                        sample_records.append({
                            "sample_idx": len(sample_records),
                            "feature_values": feature_values,
                            "score": score,
                            "recommendation": rec_result,
                            "sample_type": "Single-indicator perturbation"
                        })
                        
                        # Update the sample count for the indicator
                        feature_sample_counts[feature] += 1
                        
                    except Exception as e:
                        logger.error(f"Single-indicator perturbation sample prediction error: {str(e)}")
                        continue
            
            logger.info(f"Single-indicator perturbation stage completed, each indicator sample count: {feature_sample_counts}")
            
            # Generate multi-feature random perturbation samples
            remaining_samples = num_samples - len(sample_records)
            
            for sample_idx in range(remaining_samples):
                sample_data = base_data.copy()
                
                # To ensure sample diversity, the first 30 samples include extreme combination cases
                if sample_idx < 30:
                    # Select 2-3 indicators for extreme perturbation
                    num_features = random.randint(2, 3)
                    
                    # Select indicators based on importance
                    features_importance = [(f, feature_perturbation_strategies[f].get("importance", 0.5)) 
                                         for f in feature_names]
                    sampled_features = random.choices(
                        [f for f, _ in features_importance],
                        weights=[w for _, w in features_importance],
                        k=num_features
                    )
                    
                    # Record the change of feature values
                    feature_values = {}
                    
                    # Apply extreme perturbation to the selected multiple features
                    for extreme_feature in sampled_features:
                        strategy = feature_perturbation_strategies[extreme_feature]
                        
                        # 50% probability to use low extreme value, 50% probability to use high extreme value
                        if random.random() < 0.5 and "extremes" in strategy:
                            # Use predefined extreme values
                            extreme_value = strategy["extremes"][0]  # Low extreme value
                        else:
                            if "extremes" in strategy:
                                extreme_value = strategy["extremes"][1]  # High extreme value
                            else:
                                # Use the extreme values of the predefined perturbation range
                                abs_values = strategy["absolute"]
                                extreme_value = abs_values[0] if random.random() < 0.5 else abs_values[-1]
                        
                        # Special handling for specific features
                        if extreme_feature in extreme_values:
                            # For features that are already extreme values, ensure enough diversity
                            if random.random() < 0.4:
                                # 40% probability to use the opposite sign value
                                extreme_value = -abs(extreme_value) if extreme_value > 0 else abs(extreme_value)
                            elif random.random() < 0.5:
                                # 50% of the time (30% overall) a fraction of the original value is used
                                extreme_value = extreme_values[extreme_feature] * random.choice([0.05, 0.1, 0.2, 0.3, 0.5])
                        
                        # Apply restriction conditions
                        if not strategy.get("allow_negative", True) and extreme_value < 0:
                            extreme_value = strategy.get("min_value", 0)
                        
                        # Set the perturbed value
                        sample_data[extreme_feature] = extreme_value
                        
                        # Record the change of feature values
                        feature_values[extreme_feature] = {
                            "original": base_data[extreme_feature],
                            "perturbed": extreme_value,
                            "perturb_type": "Extreme combination"
                        }
                        
                        # Update the sample count for the indicator
                        feature_sample_counts[extreme_feature] += 1
                    
                else:
                    # Randomly determine the number of features to perturb (1 to 5) - select based on importance
                    max_features = min(5, len(feature_names))
                    num_features_to_perturb = random.randint(2, max_features)
                    
                    # Ensure some samples focus on certain important indicators
                    if sample_idx % 10 == 0:  # Every 10 samples
                        # Ensure ROE and net income growth are in the perturbed features
                        features_to_perturb = random.sample(
                            [f for f in feature_names if f not in ["returnOnEquity", "netIncomeGrowth"]], 
                            num_features_to_perturb-2
                        )
                        features_to_perturb.extend(["returnOnEquity", "netIncomeGrowth"])
                    elif sample_idx % 10 == 5:  # Middle of every 10 samples
                        # Ensure PE and debt to equity ratio are in the perturbed features
                        features_to_perturb = random.sample(
                            [f for f in feature_names if f not in ["peRatio", "debtToEquity"]], 
                            num_features_to_perturb-2
                        )
                        features_to_perturb.extend(["peRatio", "debtToEquity"])
                    else:
                        # Select indicators based on importance
                        features_importance = [(f, feature_perturbation_strategies[f].get("importance", 0.5)) 
                                            for f in feature_names]
                        features_to_perturb = random.choices(
                            [f for f, _ in features_importance],
                            weights=[w for _, w in features_importance],
                            k=num_features_to_perturb
                        )
                    
                    # Record the change of feature values
                    feature_values = {}
                    
                    # Apply perturbation to the selected features
                    for feature in features_to_perturb:
                        strategy = feature_perturbation_strategies[feature]
                        original_value = base_data[feature]
                        
                        # For extreme value features, increase the probability of using absolute value perturbation
                        use_absolute_prob = strategy["use_absolute"]
                        if feature in extreme_values:
                            use_absolute_prob = 0.9  # 90% probability to use absolute value
                        
                        # Increase the probability of using extreme value based on importance
                        extreme_prob = strategy.get("importance", 0.5) * 0.3
                        
                        # Decide the perturbation type
                        perturb_type = random.choices(
                            ["absolute", "relative", "extreme"], 
                            weights=[use_absolute_prob, 1-use_absolute_prob, extreme_prob], 
                            k=1
                        )[0]
                        
                        if perturb_type == "absolute":
                            # Use absolute value replacement
                            perturbed_value = random.choice(strategy["absolute"])
                        elif perturb_type == "relative":
                            # Use relative multiplier
                            scale = random.choice(strategy["relative"])
                            if original_value == 0:
                                # The original value is zero, use absolute value directly
                                perturbed_value = random.choice(strategy["absolute"])
                            else:
                                perturbed_value = original_value * scale
                        else:  # extreme
                            # Use extreme value
                            if "extremes" in strategy:
                                if random.random() < 0.5:
                                    perturbed_value = strategy["extremes"][0]  # Low extreme value
                                else:
                                    perturbed_value = strategy["extremes"][1]  # 高极端值
                            else:
                                # If no extreme value is defined, use the extreme values of the absolute range
                                abs_values = strategy["absolute"]
                                perturbed_value = abs_values[0] if random.random() < 0.5 else abs_values[-1]
                        
                        # Apply restriction conditions
                        if not strategy.get("allow_negative", True) and perturbed_value < 0:
                            perturbed_value = strategy.get("min_value", 0)
                        
                        # Apply minimum value limit
                        if "min_value" in strategy and perturbed_value < strategy["min_value"]:
                            perturbed_value = strategy["min_value"]
                        
                        # Set the perturbed value
                        sample_data[feature] = perturbed_value
                        
                        # Record the change of feature values
                        feature_values[feature] = {
                            "original": original_value,
                            "perturbed": perturbed_value,
                            "perturb_type": perturb_type
                        }
                        
                        # Update the sample count for the indicator
                        feature_sample_counts[feature] += 1
                
                # Get the prediction result
                try:
                    sample_values = np.array([sample_data[f] for f in feature_names])
                    rec_result = self._call_mistral_api(sample_values, feature_names)
                    score = float(self._recommendation_to_score(rec_result))
                    
                    # Record the sample
                    sample_records.append({
                        "sample_idx": len(sample_records),
                        "feature_values": feature_values,
                        "score": score,
                        "recommendation": rec_result,
                        "sample_type": "Multi-indicator perturbation"
                    })
                except Exception as e:
                    logger.error(f"Multi-indicator perturbation sample {len(sample_records)} prediction error: {str(e)}")
                    continue
            
            # Output the final indicator sample count statistics
            logger.info(f"Total sample count: {len(sample_records)}")
            logger.info(f"Indicator sample counts: {feature_sample_counts}")
            
            # Ensure enough samples are generated
            if len(sample_records) < 8:
                logger.warning(f"The number of valid samples generated is insufficient ({len(sample_records)}), the analysis result may be inaccurate")
                return {
                    "success": False,
                    "error": "The number of valid samples is insufficient, the analysis result may be inaccurate"
                }
            else:
                logger.info(f"Successfully generated {len(sample_records)} valid samples")
                
            return {
                "success": True,
                "samples": sample_records,
                "feature_names": feature_names,
                "base_score": base_score,
                "base_prediction": base_prediction,
                "feature_sample_counts": feature_sample_counts
            }
            
        except Exception as e:
            logger.error(f"Error generating perturbation samples: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            } 

    def _build_xgboost_model(self, samples: List[Dict], feature_names: List[str], base_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Build an XGBoost proxy model using perturbation samples
        
        Args:
            samples: List of generated perturbation samples
            feature_names: List of feature names
            base_data: Original financial data
            
        Returns:
            A dictionary containing the XGBoost model and related data
        """
        try:
            # 1. Prepare the feature matrix and target variable
            X = []  # Feature matrix
            y = []  # Score
            
            for sample in samples:
                # Create a feature vector
                feature_vector = []
                for feature in feature_names:
                    if feature in sample["feature_values"]:
                        # Use the perturbed value
                        feature_vector.append(sample["feature_values"][feature]["perturbed"])
                    else:
                        # Use the original value
                        feature_vector.append(base_data[feature])
                
                X.append(feature_vector)
                y.append(sample["score"])
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Log the shape of the feature matrix and target variable
            logger.info(f"Feature matrix shape: {X.shape}, target variable shape: {y.shape}")
            
            # Feature normalization processing - normalize each feature
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Log the normalized feature information
            logger.info(f"Normalized feature values:")
            for i, feature in enumerate(feature_names):
                logger.info(f"  {feature}: min={np.min(X_scaled[:, i]):.4f}, max={np.max(X_scaled[:, i]):.4f}")
            
            # Build the XGBoost model   
            # Use conservative parameters to avoid overfitting
            model = xgb.XGBRegressor(
                n_estimators=100,        # Number of trees
                max_depth=4,             # Maximum depth of the tree
                learning_rate=0.1,       # Learning rate
                subsample=0.8,           # Sample rate
                colsample_bytree=0.8,    # Feature sampling rate
                objective='reg:squarederror',  # Regression target
                random_state=42           # Random seed
            )
            
            # Train the model
            model.fit(X_scaled, y)
            
            # Evaluate the model performance
            r_squared = model.score(X_scaled, y)
            logger.info(f"XGBoost model R²: {r_squared:.4f}")
            
            # Verify the model prediction
            y_pred = model.predict(X_scaled)
            mse = np.mean((y - y_pred) ** 2)
            logger.info(f"XGBoost model MSE: {mse:.6f}")
            
            # Prepare the baseline data point
            base_x = np.array([[base_data[feature] for feature in feature_names]])
            base_x_scaled = scaler.transform(base_x)
            
            # Predict the baseline point
            base_prediction = model.predict(base_x_scaled)[0]
            logger.info(f"XGBoost prediction of the baseline point: {base_prediction:.4f}")
            
            # Return the model and related data
            return {
                "success": True,
                "model": model,
                "scaler": scaler,
                "X_train": X_scaled,
                "y_train": y,
                "feature_names": feature_names,
                "r_squared": r_squared,
                "mse": mse,
                "base_prediction": base_prediction
            }
            
        except Exception as e:
            logger.error(f"Error building XGBoost model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def _apply_treeshap(self, model, X_train, feature_names, base_data):
        """
        Apply the TreeSHAP algorithm to analyze the XGBoost model, explain the feature contributions
        
        Args:
            model: Trained XGBoost model
            X_train: Training data feature matrix
            feature_names: Feature name list
            base_data: Original financial data
            
        Returns:
            TreeSHAP analysis result dictionary
        """
        try:
            # Create the TreeSHAP explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate the SHAP values
            logger.info("Calculating the SHAP values of the sample set...")
            shap_values = explainer.shap_values(X_train)
            
            # Convert to numpy array, ensure the format is correct
            if not isinstance(shap_values, np.ndarray):
                shap_values = np.array(shap_values)
            
            # Calculate the base value (expected_value)
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray) and len(base_value) == 1:
                base_value = base_value[0]
            
            logger.info(f"XGBoost model base value: {base_value:.4f}")
            
            # Calculate the global feature importance (average absolute SHAP value of all samples)
            global_importance = np.abs(shap_values).mean(axis=0)
            
            # Normalize the global importance, ensure the sum is 1
            normalized_importance = global_importance / global_importance.sum()
            
            # Calculate the SHAP values of the original data point
            # First find the sample in the training set that is closest to the original data point
            original_point = np.array([[base_data[feature] for feature in feature_names]])
            
            # Calculate the SHAP values of the original data point
            logger.info("Calculating the SHAP values of the original data point...")
            original_shap_values = explainer.shap_values(original_point)
            
            if isinstance(original_shap_values, np.ndarray):
                if len(original_shap_values.shape) > 1:
                    original_shap_values = original_shap_values[0]  # 获取第一行（唯一一行）
            
            # Generate Chinese feature display names
            feature_display_names = FEATURE_DISPLAY_NAMES
            
            # Build the result data structure
            shap_results = []
            for i, feature in enumerate(feature_names):
                display_name = feature_display_names.get(feature, feature)
                shap_value = float(original_shap_values[i])
                
                result = {
                    "feature": display_name,
                    "originalName": feature,
                    "value": float(base_data[feature]),
                    "shapValue": shap_value,
                    "impact": shap_value,  # Add impact alias for frontend rendering convenience
                    "absShapValue": float(abs(original_shap_values[i])),
                    "globalImportance": float(normalized_importance[i])
                }
                
                shap_results.append(result)
            
            # Sort by global importance
            shap_results.sort(key=lambda x: x["globalImportance"], reverse=True)
            
            return {
                "shapValues": shap_results,
                "baseValue": float(base_value)
            }
            
        except Exception as e:
            logger.error(f"Error applying TreeSHAP: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "shapValues": [],
                "baseValue": 0.0
            } 

def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to Python native types
    
    Args:
        obj: Any Python object, possibly containing NumPy types
        
    Returns:
        The converted object, all NumPy types are replaced with Python native types
    """
    # Process NumPy scalar types
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()  # Convert NumPy arrays to lists
    # Process dictionaries
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    # Process lists or tuples
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    # Other types remain unchanged
    return obj 