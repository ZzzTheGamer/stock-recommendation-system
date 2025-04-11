import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import os
import torch
import logging
import time
import pathlib
import threading
import traceback
import uuid
from werkzeug.serving import run_simple
from collections import defaultdict
import pandas as pd
from dotenv import load_dotenv

# Import the Mistral model interpreter
from mistral_treeshap_explainer import MistralTreeSHAPExplainer, convert_numpy_types

# Import the FinBERT model interpreter
import finbert_explainer
from finbert_explainer import (
    load_model, 
    get_attention_weights, 
    compute_token_importance
)

# Import the new semantic text analyzer
from semantic_text_analyzer import SemanticTextAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables have been loaded from the.env file")

# Set request timeout
REQUEST_TIMEOUT = 300  # 5 minutes timeout
logger.info(f"Set request timeout: {REQUEST_TIMEOUT} seconds")

# Add task queue system
class TaskManager:
    def __init__(self):
        self.tasks = {}
        # Status: pending, running, completed, failed
        self.task_status = {}
        self.task_results = {}
        self.task_errors = {}
        self.task_progress = defaultdict(lambda: {"progress": 0, "message": "等待处理..."})
        self.task_lock = threading.Lock()
        
    def create_task(self):
        """Create a new task and return the task ID"""
        task_id = str(uuid.uuid4())
        with self.task_lock:
            self.task_status[task_id] = "pending"
            self.task_progress[task_id] = {"progress": 0, "message": "已创建任务，等待处理..."}
        return task_id
    
    def start_task(self, task_id, func, *args, **kwargs):
        """Start the asynchronous processing of a task"""
        def task_wrapper():
            with self.task_lock:
                self.task_status[task_id] = "running"
                self.task_progress[task_id] = {"progress": 10, "message": "开始处理任务..."}
            
            try:
                # Set the task progress update callback
                def update_progress(progress, message):
                    with self.task_lock:
                        self.task_progress[task_id] = {"progress": progress, "message": message}
                
                # Add the progress callback to the keyword arguments
                kwargs['progress_callback'] = update_progress
                
                # Execute the actual task function
                result = func(*args, **kwargs)
                
                # Save the result
                with self.task_lock:
                    self.task_results[task_id] = result
                    self.task_status[task_id] = "completed"
                    self.task_progress[task_id] = {"progress": 100, "message": "任务完成"}
                
                logger.info(f"Task {task_id} completed")
            except Exception as e:
                logger.error(f"Task {task_id} failed: {str(e)}")
                logger.error(traceback.format_exc())
                
                with self.task_lock:
                    self.task_errors[task_id] = str(e)
                    self.task_status[task_id] = "failed"
                    self.task_progress[task_id] = {"progress": 100, "message": f"任务失败: {str(e)}"}
        
        # Create a new thread to execute the task
        task_thread = threading.Thread(target=task_wrapper)
        task_thread.daemon = True  # Set to a daemon thread, it will automatically terminate when the main thread ends
        task_thread.start()
        
        with self.task_lock:
            self.tasks[task_id] = task_thread
        
        return task_id
    
    def get_task_status(self, task_id):
        """Get the task status"""
        with self.task_lock:
            if task_id not in self.task_status:
                return None
            
            status = self.task_status[task_id]
            progress = self.task_progress[task_id]
            
            if status == "completed":
                return {
                    "status": status,
                    "progress": progress,
                    "result": self.task_results.get(task_id)
                }
            elif status == "failed":
                return {
                    "status": status,
                    "progress": progress,
                    "error": self.task_errors.get(task_id)
                }
            else:
                return {
                    "status": status,
                    "progress": progress
                }
    
    def clean_old_tasks(self, max_age=3600):
        """Clean old tasks that have exceeded a certain time"""
        # Implement the cleanup logic
        pass

# Create a task manager instance
task_manager = TaskManager()

app = Flask(__name__)
# Configure CORS to allow cross-domain requests - use a more permissive configuration
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# Add a response hook to check the CORS headers, ensure this hook is the first to be called
@app.after_request
def after_request_func(response):
    # Ensure all responses set the correct CORS headers
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    
    # Record the status code and CORS headers for each response
    logger.info(f"Response status code: {response.status_code}, CORS headers set")
    return response

# Pre-flight request processing - immediately respond to OPTIONS requests
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    """Process all OPTIONS preflight requests"""
    logger.info(f"Processing OPTIONS preflight request: /{path}")
    return jsonify({})

# Add an error handler to avoid uncaught exceptions causing 500 errors
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Uncaught exception: {str(e)}")
    logger.error(f"Exception stack trace: {traceback.format_exc()}")
    
    # Check if it is a CORS-related error
    error_message = str(e).lower()
    if 'cors' in error_message or 'origin' in error_message or 'access-control' in error_message:
        return jsonify({
            "error": True,
            "message": f"CORS error: {str(e)}",
            "type": "CORSError"
        }), 400
    
    # Return a formatted error response
    return jsonify({
        "error": True,
        "message": f"Server internal error: {str(e)}",
        "type": "ServerError"
    }), 500

# Add a mutex lock to prevent concurrent loading
model_lock = threading.Lock()

# Initialize the Mistral interpreter components
mistral_treeshap_explainer = MistralTreeSHAPExplainer()

# Initialize the new semantic text analyzer
semantic_text_analyzer = SemanticTextAnalyzer()
logger.info("Initialization of semantic text analyzer completed")

# Load the model when the application starts, ensuring it is loaded only once
with model_lock:
    if finbert_explainer.model is None or finbert_explainer.tokenizer is None:
        logger.info("Loading FinBERT model when the application starts...")
        load_model()

@app.route('/api/explain-single-text', methods=['POST'])
def explain_single_text():
    try:
        # Record the request
        start_time = time.time()
        logger.info("Starting to process the explainability analysis request for a single text...")

        # Check if the request contains JSON data
        if not request.is_json:
            logger.error("Request content type error: needs JSON")
            return jsonify({
                "error": True,
                "message": "Request content type error: needs JSON"
            }), 400

        # Parse the text from the request body
        data = request.get_json()
        
        if not data or 'text' not in data:
            logger.error("Request data is incomplete: missing text field")
            return jsonify({
                "error": True,
                "message": "Request data is incomplete: missing text field"
            }), 400

        text = data['text']
        
        # Validate text parameter
        if not text or not isinstance(text, str):
            logger.error(f"Invalid text parameter: {text}")
            return jsonify({
                "error": True,
                "message": "Invalid text parameter"
            }), 400
        
        # Limit text length
        max_length = 1000
        if len(text) > max_length:
            logger.warning(f"Text too long ({len(text)} characters), truncated to {max_length} characters")
            text = text[:max_length]
        
        # CORS problem check
        logger.info(f"Request header: Origin={request.headers.get('Origin', 'None')}")
        
        # Use the model for sentiment analysis
        sentiment_start = time.time()
        logger.info(f"Starting sentiment analysis: {text[:50]}...")
        
        # Ensure the model is loaded
        if finbert_explainer.model is None or finbert_explainer.tokenizer is None:
            with model_lock:
                if finbert_explainer.model is None or finbert_explainer.tokenizer is None:
                    logger.warning("Model not loaded, attempting to load")
                    load_model()
        
        # Encode the text
        encoded_inputs = finbert_explainer.tokenizer([text], padding=True, truncation=True, 
                                  return_tensors="pt", max_length=256)
        
        # Transfer to the correct device
        encoded_inputs = {k: v.to(finbert_explainer.device) for k, v in encoded_inputs.items()}
        
        # Calculate the sentiment score
        with torch.no_grad():
            outputs = finbert_explainer.model(**encoded_inputs)
            
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
        
        # Determine the sentiment
        max_idx = np.argmax(probs[0])
        sentiment = finbert_explainer.FINBERT_LABELS[max_idx].capitalize()
        
        # Calculate the sentiment score (positive score minus negative score)
        sentiment_score = float(probs[0][finbert_explainer.POSITIVE_IDX] - probs[0][finbert_explainer.NEGATIVE_IDX])
        
        # Record the detailed score calculation process
        prob_details = ", ".join([f"{finbert_explainer.FINBERT_LABELS[j]}: {probs[0][j]:.4f}" for j in range(len(finbert_explainer.FINBERT_LABELS))])
        logger.info(f"Sentiment probability distribution - {prob_details}")
        logger.info(f"Score calculation: {sentiment_score:.4f}, Sentiment judgment: {sentiment}")
        
        sentiment_time = time.time() - sentiment_start
        logger.info(f"Sentiment analysis completed, time taken: {sentiment_time:.2f} seconds")
        
        # Initialize variables
        attention_weights = None
        importance_values = None
        tokens = None
        importance_error = None
        attention_time = 0
        importance_time = 0
        
        try:
            # Calculate attention weights
            attention_start = time.time()
            
            # Ensure the encoded input is passed in instead of the original text
            encoded_inputs = finbert_explainer.tokenizer([text], padding=True, truncation=True, 
                                  return_tensors="pt", max_length=256)
            encoded_inputs = {k: v.to(finbert_explainer.device) for k, v in encoded_inputs.items()}
            
            # Use the encoded input to get attention weights
            attention_weights = finbert_explainer.get_attention_weights(encoded_inputs)
            
            attention_time = time.time() - attention_start
            logger.info(f"Attention weights calculation completed, time taken: {attention_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Attention weights calculation error: {str(e)}")
            logger.error(traceback.format_exc())
            attention_weights = None
        
        try:
            # Calculate token importance
            importance_start = time.time()
            logger.info("Starting to calculate the importance of tokens in a single text using integrated gradients...")
            
            # Change the original unpacking to handle the return value of the dictionary structure
            importance_result = finbert_explainer.compute_token_importance(text)
            
            # Extract data from the result dictionary
            if importance_result and isinstance(importance_result, dict) and "tokens" in importance_result and "values" in importance_result:
                # Get tokens and importance_values (if it's the first element in the list)
                tokens = importance_result["tokens"][0] if importance_result["tokens"] and len(importance_result["tokens"]) > 0 else None
                importance_values = importance_result["values"][0] if importance_result["values"] and len(importance_result["values"]) > 0 else None
            else:
                logger.error("Invalid data structure returned by token importance calculation")
                tokens = None
                importance_values = None
                
            importance_time = time.time() - importance_start
            logger.info(f"Token importance calculation completed, time taken: {importance_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Token importance calculation error: {str(e)}")
            logger.error(traceback.format_exc())
            importance_values = None
            tokens = None
            importance_error = str(e)
        
        # Calculate the total processing time
        total_time = time.time() - start_time
        
        # Build the response
        response = {
            "sentiment": sentiment,
            "sentimentScore": sentiment_score,
            "explainability": {
                "tokens": tokens,
                "importanceValues": importance_values.tolist() if isinstance(importance_values, np.ndarray) else importance_values,
                "attentionMatrix": attention_weights.tolist() if isinstance(attention_weights, np.ndarray) else attention_weights,
                "importanceError": importance_error
            },
            "methodInfo": {
                "method": "IntegratedGradients",
                "description": "Using integrated gradients to calculate feature importance"
            },
            "processingTimes": {
                "total": total_time,
                "sentiment": sentiment_time,
                "attention": attention_time,
                "importance": importance_time
            }
        }
        
        logger.info(f"📊 Single text analysis completed, sentiment: {sentiment}, used explanation method: IntegratedGradients, total time taken: {total_time:.2f} seconds")
        
        # Return the result
        return jsonify(response)
        
    except Exception as e:
        # Capture any unhandled exceptions
        logger.error(f"Error processing single text explanation request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": True,
            "message": f"Server internal error: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "model_loaded": finbert_explainer.model is not None,
        "tokenizer_loaded": finbert_explainer.tokenizer is not None,
        "explainer_available": finbert_explainer.ig_explainer is not None,
        "device": str(finbert_explainer.device),
        "gpu_available": torch.cuda.is_available(),
        "finbert_labels": finbert_explainer.FINBERT_LABELS if finbert_explainer.model is not None else None
    }
    return jsonify(status)


@app.route('/api/explain-mistral/text', methods=['POST'])
def explain_mistral_text():
    """
    Provide semantic importance analysis of Mistral model text output
    
    Request body:
    {
        "text": "The text to analyze...",
        "section": "summary" or "recommendation"
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({
                "success": False,
                "error": "Request data is empty"
            }), 400
            
        # Extract data
        text = data.get('text', '')
        section = data.get('section', 'summary')
        
        if not text:
            return jsonify({
                "success": False,
                "error": "Text is empty"
            }), 400
        
        # Use the new semantic analyzer
        logger.info(f"Using the semantic analyzer to analyze the {section} text...")
        result = semantic_text_analyzer.analyze_text(text, section)
        
        # Add analysis method information to the result
        if result.get("success", False):
            result["method"] = "Semantic similarity and financial domain knowledge analysis"
            result["description"] = "Mixed analysis based on sentence similarity and weighted financial indicators"
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Mistral text analysis error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# Add TreeSHAP method API endpoint
@app.route('/api/explain-mistral/treeshap', methods=['POST'])
def treeshap_mistral_explain():
    start_time = time.time()
    try:
        # Add progress logs
        logger.info("Starting TreeSHAP analysis process...")
        
        # Check request data
        data = request.json
        if not data:
            logger.error("TreeSHAP analysis request data is empty")
            return jsonify({
                'success': False,
                'error': 'Missing required financial data',
                'error_type': 'EMPTY_REQUEST'
            }), 400
            
        if 'financialData' not in data:
            logger.error("TreeSHAP analysis request missing financialData field")
            return jsonify({
                'success': False,
                'error': 'Missing financial data (financialData) field',
                'error_type': 'MISSING_FINANCIAL_DATA'
            }), 400
            
        financial_data = data['financialData']
        recommendation = data.get('recommendation', None)
        
        # Check the validity of the financial data
        required_fields = ["revenueGrowth", "netIncomeGrowth", "currentRatio", 
                          "debtToEquity", "returnOnEquity", "peRatio"]
        missing_fields = [field for field in required_fields if field not in financial_data]
        
        if missing_fields:
            logger.error(f"Financial data missing required fields: {missing_fields}")
            return jsonify({
                'success': False,
                'error': f'Financial data missing required fields: {", ".join(missing_fields)}',
                'error_type': 'INCOMPLETE_FINANCIAL_DATA'
            }), 400
        
        # The calculation may take a long time
        logger.info(f"TreeSHAP analysis may take a long time, timeout set to {REQUEST_TIMEOUT} seconds...")
        
        # Use TreeSHAP explainer for analysis
        logger.info("Calling TreeSHAP explainer...")
        progress_start = time.time()
        result = mistral_treeshap_explainer.explain_recommendation(financial_data, recommendation)
        analysis_time = time.time() - progress_start
        logger.info(f"TreeSHAP explainer analysis completed, time taken: {analysis_time:.2f} seconds")
        
        if not result.get("success", False):
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"TreeSHAP analysis failed: {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg,
                "error_type": "TREESHAP_ANALYSIS_FAILED",
                "processing_time": analysis_time
            }), 500
        
        # Convert NumPy types to Python native types
        logger.info("Converting result data types...")
        result = convert_numpy_types(result)
        
        # Add processing time information
        total_time = time.time() - start_time
        result["processingTime"] = total_time
        logger.info(f"TreeSHAP analysis completed, total time taken: {total_time:.2f} seconds")
        
        # Add result statistics information
        if "shapValues" in result:
            pos_features = sum(1 for item in result["shapValues"] if item.get("shapValue", 0) > 0)
            neg_features = sum(1 for item in result["shapValues"] if item.get("shapValue", 0) < 0)
            logger.info(f"Analysis results: {len(result['shapValues'])} features, {pos_features} positive impacts, {neg_features} negative impacts")
        
        # Return the result data directly
        return jsonify(result)
    
    except Exception as e:
        # Detailed record of exceptions
        logger.error(f"TreeSHAP analysis error: {str(e)}")
        logger.error(f"Exception details: {traceback.format_exc()}")
        
        # Calculate the elapsed processing time
        elapsed_time = time.time() - start_time
        
        # Return formatted error information
        error_type = type(e).__name__
        error_message = str(e)
        
        return jsonify({
            "success": False,
            "error": f"TreeSHAP analysis failed: {error_message}",
            "error_type": error_type,
            "processing_time": elapsed_time,
            "stack_trace": traceback.format_exc() if app.debug else None
        }), 500

# Add asynchronous TreeSHAP analysis endpoint
@app.route('/api/explain-mistral/treeshap-async', methods=['POST'])
def treeshap_mistral_explain_async():
    """Create a TreeSHAP analysis asynchronous task, return the task ID immediately"""
    try:
        # Check request data
        data = request.json
        if not data:
            logger.error("TreeSHAP analysis request data is empty")
            return jsonify({
                'success': False,
                'error': 'Missing required financial data',
                'error_type': 'EMPTY_REQUEST'
            }), 400
            
        if 'financialData' not in data:
            logger.error("TreeSHAP analysis request missing financialData field")
            return jsonify({
                'success': False,
                'error': 'Missing financial data (financialData) field',
                'error_type': 'MISSING_FINANCIAL_DATA'
            }), 400
            
        financial_data = data['financialData']
        recommendation = data.get('recommendation', None)
        
        # Check the validity of the financial data
        required_fields = ["revenueGrowth", "netIncomeGrowth", "currentRatio", 
                          "debtToEquity", "returnOnEquity", "peRatio"]
        missing_fields = [field for field in required_fields if field not in financial_data]
        
        if missing_fields:
            logger.error(f"Financial data missing required fields: {missing_fields}")
            return jsonify({
                'success': False,
                'error': f'Financial data missing required fields: {", ".join(missing_fields)}',
                'error_type': 'INCOMPLETE_FINANCIAL_DATA'
            }), 400
        
        # Create an asynchronous task
        task_id = task_manager.create_task()
        logger.info(f"Creating TreeSHAP analysis asynchronous task: {task_id}")
        
        # Define the function to execute the task, including the progress callback function
        def run_treeshap_analysis(financial_data, recommendation, progress_callback=None):
            # Default progress callback function
            if progress_callback is None:
                progress_callback = lambda p, m: None
                
            progress_callback(15, "Starting TreeSHAP analysis...")
            
            # Wrap the original explain_recommendation method, add progress reporting
            original_method = mistral_treeshap_explainer.explain_recommendation
            
            # Redefine the method to add progress reporting
            def wrapped_method(financial_data, recommendation):
                progress_callback(25, "Generating perturbation samples...")
                
                # Save the original _generate_perturbation_samples method
                original_generate_samples = mistral_treeshap_explainer._generate_perturbation_samples
                
                # Redefine the method to add progress reporting
                def wrapped_generate_samples(*args, **kwargs):
                    progress_callback(30, "Starting to generate samples...")
                    result = original_generate_samples(*args, **kwargs)
                    progress_callback(45, "Sample generation completed, building XGBoost model...")
                    return result
                
                # Replace the method
                mistral_treeshap_explainer._generate_perturbation_samples = wrapped_generate_samples
                
                # Save the original _build_xgboost_model method
                original_build_model = mistral_treeshap_explainer._build_xgboost_model
                
                # Redefine the method to add progress reporting
                def wrapped_build_model(*args, **kwargs):
                    progress_callback(50, "Building proxy model...")
                    result = original_build_model(*args, **kwargs)
                    progress_callback(70, "Model building completed, applying TreeSHAP...")
                    return result
                
                # Replace the method
                mistral_treeshap_explainer._build_xgboost_model = wrapped_build_model
                
                # Save the original _apply_treeshap method
                original_apply_treeshap = mistral_treeshap_explainer._apply_treeshap
                
                # Redefine the method to add progress reporting
                def wrapped_apply_treeshap(*args, **kwargs):
                    progress_callback(75, "Calculating SHAP values...")
                    result = original_apply_treeshap(*args, **kwargs)
                    progress_callback(90, "SHAP calculation completed, preparing results...")
                    return result
                
                # Replace the method
                mistral_treeshap_explainer._apply_treeshap = wrapped_apply_treeshap
                
                # Call the original method
                try:
                    result = original_method(financial_data, recommendation)
                    progress_callback(95, "Processing completed, formatting results...")
                    return result
                finally:
                    # Restore the original method
                    mistral_treeshap_explainer._generate_perturbation_samples = original_generate_samples
                    mistral_treeshap_explainer._build_xgboost_model = original_build_model
                    mistral_treeshap_explainer._apply_treeshap = original_apply_treeshap
            
            # Use the wrapped method to perform the analysis
            result = wrapped_method(financial_data, recommendation)
            
            if not result.get("success", False):
                progress_callback(100, f"Analysis failed: {result.get('error', 'Unknown error')}")
                return result
            
            # Convert NumPy types to Python native types
            progress_callback(98, "Converting data types...")
            result = convert_numpy_types(result)
            
            progress_callback(100, "Analysis completed")
            return result
        
        # Start the asynchronous task
        task_manager.start_task(task_id, run_treeshap_analysis, financial_data, recommendation)
        
        # Immediately return the task ID
        return jsonify({
            'success': True,
            'message': 'TreeSHAP analysis asynchronous task created',
            'taskId': task_id
        })
        
    except Exception as e:
        # Detailed record of exceptions
        logger.error(f"Error creating TreeSHAP asynchronous task: {str(e)}")
        logger.error(f"Exception details: {traceback.format_exc()}")
        
        # Return formatted error information
        error_type = type(e).__name__
        error_message = str(e)
        
        return jsonify({
            "success": False,
            "error": f"Error creating TreeSHAP asynchronous task: {error_message}",
            "error_type": error_type,
            "stack_trace": traceback.format_exc() if app.debug else None
        }), 500

# Add endpoint to check task status
@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get the current status and result of the task (if completed)"""
    try:
        status = task_manager.get_task_status(task_id)
        
        if status is None:
            return jsonify({
                'success': False,
                'error': f'Task not found: {task_id}',
                'error_type': 'TASK_NOT_FOUND'
            }), 404
        
        # If the task is completed, ensure the result is converted to a JSON safe format
        if status.get('status') == 'completed' and 'result' in status:
            result = status['result']
            if isinstance(result, dict) and 'shapValues' in result:
                for i, shap_value in enumerate(result['shapValues']):
                    if isinstance(shap_value.get('shapValue'), (np.float32, np.float64)):
                        result['shapValues'][i]['shapValue'] = float(shap_value['shapValue'])
            
            status['result'] = result
        
        return jsonify({
            'success': True,
            'taskStatus': status
        })
        
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': f'Error getting task status: {str(e)}',
            'error_type': 'TASK_STATUS_ERROR'
        }), 500

if __name__ == '__main__':
    # Start the Flask application, set a longer timeout
    logger.info(f"Starting Flask application, request timeout set to {REQUEST_TIMEOUT} seconds")
    
    # Set the socket timeout in the global scope
    import socket
    socket.setdefaulttimeout(REQUEST_TIMEOUT)
    
    # For werkzeug, run_simple does not support the request_timeout parameter
    run_simple('0.0.0.0', 5000, app, use_reloader=True, use_debugger=True, threaded=True) 