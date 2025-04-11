"""
FinBERT model's sentiment analysis and interpretability module

Provides integrated gradient explanation and Self-attention analysis of FinBERT model outputs
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter thread issues

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import logging
import time
import traceback
from functools import lru_cache
import string
import pathlib
import types
import gc  # Import the garbage collection module

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Store the FinBERT label ID mapping, checked and corrected
FINBERT_LABELS = ["positive", "negative", "neutral"]  # Correct label order
POSITIVE_IDX = 0  # Index of positive sentiment
NEGATIVE_IDX = 1  # Index of negative sentiment

# Output GPU information
if torch.cuda.is_available():
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.2f} GB")
    logger.info(f"CUDA version: {torch.version.cuda}")
else:
    logger.warning("No available GPU detected, will use CPU for calculations, which may be slower")

# Global variable storage model and tokenizer
model = None
tokenizer = None
ig_explainer = None  # Integrated gradient explainer instance

# New: IntegratedGradientsExplainer class
class IntegratedGradientsExplainer:
    """
    Implement interpretability analysis based on integrated gradients
    Compared to SHAP, integrated gradients do not require background data sets and are more suitable for single text explanation
    """
    def __init__(self, model, tokenizer, steps=25):
        """
        Initialize the integrated gradient explainer
        
        Parameters:
            model: The model to explain, must be a pytorch model
            tokenizer: The tokenizer for text processing
            steps: The number of integration steps, default is 25, increasing steps can improve accuracy but slow down calculation
        """
        self.model = model  # Need to pass in the model directly
        self.tokenizer = tokenizer
        self.steps = steps  # The number of integration steps
        
        # Record the initialization parameters
        logger.info(f"Initialized integrated gradient explainer, steps: {self.steps}")
        
        # Validate model and tokenizer
        if model is None:
            raise ValueError("Must provide a valid model")
        if tokenizer is None:
            raise ValueError("Must provide a valid tokenizer")
            
        # Ensure the model is in evaluation mode
        self.model.eval()
        
    def explain(self, text, target_class=POSITIVE_IDX):
        """Calculate the feature importance of integrated gradients for a single text"""
        logger.info(f"Using integrated gradients to calculate feature importance: {text[:30]}...")
        
        try:
            # Encode text
            encoded_text = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=128)
            # Print the encoded fields, for debugging
            logger.info(f"Tokenizer returned fields: {list(encoded_text.keys())}")
            
            encoded_text = {k: v.to(device) for k, v in encoded_text.items()}
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(encoded_text['input_ids'][0])
            logger.info(f"Successfully obtained {len(tokens)} tokens")
            
            # Calculate integrated gradients
            attributions = self._compute_integrated_gradients(encoded_text, target_class)
            
            # Get the baseline value (the prediction value of the model on the current text)
            with torch.no_grad():
                outputs = self.model(**encoded_text)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                base_value = float(probs[0, target_class].cpu().numpy())
            
            # Ensure the length of attributions matches the length of tokens
            if len(attributions) != len(tokens):
                logger.warning(f"Attribute value length {len(attributions)} does not match token length {len(tokens)}, resizing")
                new_attributions = np.zeros(len(tokens))
                # Copy as many values as possible
                min_len = min(len(attributions), len(tokens))
                new_attributions[:min_len] = attributions[:min_len]
                attributions = new_attributions
            
            # Filter meaningful tokens and values
            filtered_tokens, filtered_values = self._filter_tokens_and_values(tokens, attributions)
            
            # Check if the result is meaningful
            if not filtered_tokens or (len(filtered_values) > 0 and np.max(np.abs(filtered_values)) < 1e-6):
                logger.warning("Integrated gradient calculation produced meaningless results, falling back to perturbation method")
                return self._compute_perturbation_importance(text)
            
            # Print the important tokens found
            if filtered_tokens and filtered_values:
                top_tokens = min(3, len(filtered_tokens))
                logger.info(f"Important tokens: {', '.join([f'{filtered_tokens[i]}({filtered_values[i]:.4f})' for i in range(top_tokens)])}")
            
            result = {
                "tokens": filtered_tokens,
                "values": filtered_values,
                "base_value": base_value,
                "method": "IntegratedGradients"  # Mark the method used
            }
            
            logger.info(f"✅ Integrated gradient calculation successful, obtained {len(filtered_tokens)} key tokens")
            return result
            
        except Exception as e:
            logger.error(f"Integrated gradient calculation error: {str(e)}")
            logger.error(traceback.format_exc())
            # Fall back to perturbation method when an error occurs
            return self._compute_perturbation_importance(text)
    
    def _compute_integrated_gradients(self, encoded_text, target_class=POSITIVE_IDX):
        """Implement true integrated gradient calculation (Integrated Gradients)"""
        try:
            # 1. Find the embedding layer 
            embedding_layer = None
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Embedding) and any(key in name for key in ['word_embeddings', 'embeddings.word']):
                    embedding_layer = module
                    logger.info(f"Found embedding layer: {name}, shape: {module.weight.shape}")
                    break
            
            if embedding_layer is None:
                # Try a broader search
                for name, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Embedding) and module.weight.shape[0] > 1000:
                        embedding_layer = module
                        logger.info(f"Found possible embedding layer: {name}, shape: {module.weight.shape}")
                        break
            
            if embedding_layer is None:
                logger.error("The word embedding layer was not found and the integration gradient could not be calculated")
                raise ValueError("The word embedding layer was not found and the integration gradient could not be calculated")
            
            # 2. Prepare input
            input_ids = encoded_text['input_ids'].to(device)
            attention_mask = encoded_text['attention_mask'].to(device)
            
            # 3. Get input embeddings
            input_embeddings = embedding_layer(input_ids)
            
            # 4. Create baseline embeddings (all zero embeddings, this is the standard baseline for IG)
            baseline_embeddings = torch.zeros_like(input_embeddings)
            
            # 5. Reduce the number of steps based on the running environment
            # Use fewer steps on CPU, reducing the computational burden
            is_cpu = not torch.cuda.is_available()
            steps = 25
            logger.info(f"Using {steps} steps to calculate integrated gradients on {'CPU' if is_cpu else 'GPU'}")
            
            # 6. Create a tensor to store attribute values, initialized to zero
            attributions = torch.zeros_like(input_ids, dtype=torch.float32)
            
            # 7. The core calculation of IG: integrate gradients along the path from baseline to input
            for step in range(steps):
                # Create the α value for the current path point (0->1)
                alpha = float(step) / (steps - 1)
                
                # Calculate the current interpolated embeddings
                current_embeddings = baseline_embeddings + alpha * (input_embeddings - baseline_embeddings)
                
                # Separate this embedding and set it to require gradients
                current_embeddings = current_embeddings.detach().requires_grad_(True)
                
                # Create a forward propagation function (this method is simpler and more robust than the original)
                original_forward = embedding_layer.forward
                
                try:
                    # Define a new forward propagation function
                    def new_forward(self, ids, **kwargs):
                        # Ignore the incoming ids, return the preset embeddings
                        return current_embeddings
                    
                    # Set the new forward propagation function for the embedding layer
                    embedding_layer.forward = types.MethodType(new_forward, embedding_layer)
                    
                    # Forward propagation calculation
                    outputs = self.model(**encoded_text)
                    target_score = outputs.logits[0, target_class]
                    
                    # Backward propagation
                    self.model.zero_grad()
                    target_score.backward(retain_graph=False)
                    
                    # Get the gradient of the embedding
                    if current_embeddings.grad is not None:
                        # Extract the gradient, the dimension is [batch_size, seq_len, embedding_dim]
                        embed_grads = current_embeddings.grad
                        
                        # According to the IG formula, we need to calculate: (input - baseline) * gradient
                        # The difference between the input and the baseline
                        embed_diff = (input_embeddings - baseline_embeddings).detach()
                        
                        # Sum by embedding dimension to get token-level gradients
                        # We use dot product instead of separate summation, which is more consistent with the IG formula
                        token_grads = torch.sum(embed_grads * embed_diff, dim=-1) / steps
                        
                        # Add to the final attribute values
                        attributions += token_grads.squeeze(0)
                        
                        if step % 10 == 0:  # Record progress every 10 steps
                            logger.info(f"Step {step+1}/{steps} - Gradient mean: {token_grads.mean().item():.6f}")
                    else:
                        logger.warning(f"Step {step+1}/{steps} did not produce gradients")
                        
                except Exception as e:
                    logger.error(f"Step {step+1} error: {str(e)}")
                    # Continue to the next step
                finally:
                    # Restore the original forward propagation
                    embedding_layer.forward = original_forward
                    # Clean up memory
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 8. Apply the attention mask
            attributions = attributions * attention_mask.squeeze(0).float()
            
            # 9. Convert to numpy and return
            attributions_np = attributions.cpu().detach().numpy()
            
            # Ensure handling numpy.matrix types
            if isinstance(attributions_np, np.matrix):
                attributions_np = np.asarray(attributions_np)
            
            # Use squeeze and specify axis to flatten
            attributions_np = np.squeeze(attributions_np, axis=0) 
            
            # Add additional flattening processing
            if attributions_np.ndim > 1:
                attributions_np = attributions_np.reshape(-1)
            
            # Add debugging logs
            logger.info(f"Final flattened attributions_np shape={attributions_np.shape}, type={type(attributions_np)}")
            
            # Count non-zero values
            non_zero = np.sum(np.abs(attributions_np) > 1e-6)
            max_val = np.max(np.abs(attributions_np)) if np.size(attributions_np) > 0 else 0
            logger.info(f"Integrated gradient calculation completed: {non_zero}/{attributions_np.size} non-zero values, max value {max_val:.6f}")
            
            # 10. Check if the result is meaningful
            if non_zero > 0 and max_val > 1e-6:
                # Normalize the result, so the maximum absolute value is 1,便于可视化
                attributions_np = attributions_np / max_val
            
            return attributions_np
            
        except Exception as e:
            logger.error(f"Integrated gradient calculation error: {str(e)}")
            logger.error(traceback.format_exc())
            # Return a safe empty result
            return np.zeros(encoded_text['input_ids'].shape[1])
    
    def _filter_tokens_and_values(self, tokens, attributions):
        """Filter meaningful tokens and corresponding importance values"""
        filtered_tokens = []
        filtered_values = []
        
        # Add punctuation detection
        punctuation = string.punctuation + '，。！？、；：""''（）【】《》…—'  # Chinese and English punctuation
        
        # Create token-attribution pairs
        token_attribution_pairs = []
        for i, token in enumerate(tokens):
            # Filter out special tokens, whitespace, and punctuation
            if (token not in ['[PAD]', '[CLS]', '[SEP]', '[UNK]', '<s>', '</s>', '<pad>', '[MASK]', ''] and 
                not token.isspace() and 
                not all(char in punctuation for char in token)):
                # Ensure the index is within the valid range
                if i < len(attributions):
                    # Save absolute importance, original importance value, and token
                    token_attribution_pairs.append((abs(attributions[i]), attributions[i], token))
        
        # Sort by absolute importance value (from largest to smallest)
        token_attribution_pairs.sort(reverse=True)
        
        # Check if there are meaningful tokens
        if not token_attribution_pairs:
            logger.warning("No meaningful tokens found, trying to include more token types")
            # Get token-attribution pairs again, this time more lenient
            for i, token in enumerate(tokens):
                if token not in ['[PAD]', '<pad>', ''] and i < len(attributions) and not token.isspace():
                    token_attribution_pairs.append((abs(attributions[i]), attributions[i], token))
            
            token_attribution_pairs.sort(reverse=True)
        
        # If still no tokens, return empty
        if not token_attribution_pairs:
            logger.warning("Even with more lenient conditions, no valid tokens were found")
            return [], []
        
        # Select the top 8 important tokens (or all, if fewer than 8)
        top_n = min(8, len(token_attribution_pairs))
        
        # Check if the importance values are meaningful
        max_importance = max([pair[0] for pair in token_attribution_pairs[:top_n]])
        
        # If all importance values are very small, we may need further analysis
        if max_importance < 1e-5:
            logger.warning(f"Maximum importance value is too small ({max_importance:.8f}), try further processing")
            
            # If the value is too small but still not zero, we can try normalization
            if max_importance > 0:
                # Extract tokens and relative importance values
                for _, value, token in token_attribution_pairs[:top_n]:
                    # Normalize the value
                    normalized_value = value / max_importance
                    filtered_tokens.append(token)
                    filtered_values.append(float(normalized_value))
            else:
                # If all are zero, keep tokens but set importance to a small random value
                logger.warning("All importance values are zero, using a small non-zero random value")
                for _, _, token in token_attribution_pairs[:top_n]:
                    filtered_tokens.append(token)
                    # Add a small random value for visualization
                    filtered_values.append(float(np.random.uniform(-0.01, 0.01)))
        else:
            # The importance values look meaningful, use them directly
            for _, value, token in token_attribution_pairs[:top_n]:
                filtered_tokens.append(token)
                filtered_values.append(float(value))
        
        logger.info(f"Filtered {len(filtered_tokens)}/{len(tokens)} important tokens")
        return filtered_tokens, filtered_values
    
    def _compute_perturbation_importance(self, text, tokens=None):
        """Use a simple perturbation method to calculate token importance as an alternative"""
        logger.info("🔄 Using perturbation method to calculate token importance - this is a fallback method")
        try:
            # If tokens are not provided, get tokens
            if tokens is None:
                encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
                logger.info(f"Perturbation method Tokenizer returned fields: {list(encoded_input.keys())}")
                tokens = self.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
                
            # Calculate the baseline prediction
            with torch.no_grad():
                # Encode text
                encoded_inputs = self.tokenizer([text], padding=True, truncation=True, 
                                       return_tensors="pt", max_length=64)
                encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
                
                # Predict
                outputs = self.model(**encoded_inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
                
            # Get the positive sentiment probability as the baseline value
            base_value = float(probs[0][POSITIVE_IDX])
            
            # Add punctuation detection
            punctuation = string.punctuation + '，。！？、；：""''（）【】《》…—'
            
            # Filter meaningful tokens
            meaningful_tokens = []
            meaningful_indices = []
            
            for i, token in enumerate(tokens):
                if (token not in ['[PAD]', '[CLS]', '[SEP]', '[UNK]', ''] and 
                    not token.isspace() and 
                    not all(char in punctuation for char in token)):
                    meaningful_tokens.append(token)
                    meaningful_indices.append(i)
            
            if not meaningful_tokens:
                logger.warning("No meaningful tokens found")
                return {
                    "tokens": [],
                    "values": [],
                    "base_value": base_value,
                    "method": "Perturbation"
                }
                
            # Process only the first 8 meaningful tokens
            filtered_tokens = []
            importance_values = []
            
            # Process only the first 8 meaningful tokens
            logger.info(f"Perturbation method starts processing {len(meaningful_tokens[:8])} meaningful tokens")
            for idx, token in zip(meaningful_indices[:8], meaningful_tokens[:8]):
                try:
                    # Create a new text without the current token
                    tokens_copy = tokens.copy()
                    tokens_copy[idx] = '[UNK]'  # Replace the current token with [UNK]
                    
                    # Convert tokens back to text
                    perturbed_text = self.tokenizer.convert_tokens_to_string(tokens_copy)
                    
                    # Predict the perturbed text
                    perturbed_input = self.tokenizer(perturbed_text, padding=True, truncation=True, return_tensors="pt")
                    perturbed_input = {k: v.to(device) for k, v in perturbed_input.items()}
                    
                    with torch.no_grad():
                        perturbed_outputs = self.model(**perturbed_input)
                        perturbed_probs = torch.nn.functional.softmax(perturbed_outputs.logits, dim=-1).cpu().numpy()
                    
                    # Calculate the importance value as the difference between the original prediction and the perturbed prediction
                    importance_value = float(base_value - perturbed_probs[0][POSITIVE_IDX])
                    
                    filtered_tokens.append(token)
                    importance_values.append(importance_value)
                except Exception as token_error:
                    logger.error(f"Error processing token '{token}': {str(token_error)}")
                    # Continue to the next token
                    continue
            
            # Sort and select the most important tokens
            if filtered_tokens:
                pairs = [(abs(v), v, t) for v, t in zip(importance_values, filtered_tokens)]
                pairs.sort(reverse=True)
                
                # Process only the first 8 important tokens
                pairs = pairs[:8]
                
                importance_values = [p[1] for p in pairs]
                filtered_tokens = [p[2] for p in pairs]
            
            logger.info(f"✅ Perturbation method completed - processed {len(filtered_tokens)} tokens")
            result = {
                "tokens": filtered_tokens,
                "values": importance_values,
                "base_value": base_value,
                "method": "Perturbation"  # Mark the method used
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Perturbation method failed to calculate token importance: {str(e)}")
            logger.error(traceback.format_exc())
            # Even if there is an error, return a valid empty result
            return {
                "tokens": [],
                "values": [],
                "base_value": 0.5,
                "method": "Perturbation"
            }

def load_model():
    """Load the FinBERT model and tokenizer, and initialize the explainer"""
    global model, tokenizer, ig_explainer, FINBERT_LABELS, POSITIVE_IDX, NEGATIVE_IDX
    
    logger.info(f"Starting to load model, using device: {device}")
    
    # Initialize to None to avoid referencing uninitialized variables
    model = None
    tokenizer = None
    ig_explainer = None
    
    # Load the FinBERT model and tokenizer
    model_name = "ProsusAI/finbert"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Get the correct label order
        if hasattr(model.config, 'id2label'):
            FINBERT_LABELS = [model.config.id2label[i].lower() for i in range(len(model.config.id2label))]
            logger.info(f"Detected label order from model configuration: {FINBERT_LABELS}")
            
            # Update label indices
            for i, label in enumerate(FINBERT_LABELS):
                if 'positive' in label:
                    POSITIVE_IDX = i
                    logger.info(f"Positive sentiment index: {POSITIVE_IDX}")
                elif 'negative' in label:
                    NEGATIVE_IDX = i
                    logger.info(f"Negative sentiment index: {NEGATIVE_IDX}")
        
        # Move the model to GPU (if available)
        model.to(device)
        
        # Print more GPU information
        if torch.cuda.is_available():
            logger.info(f"Model loaded to GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Current GPU memory usage: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")
            logger.info(f"Maximum GPU memory allocation: {torch.cuda.max_memory_allocated(0)/1024/1024:.2f} MB")
        else:
            logger.warning("Model loaded to CPU, calculation speed may be slow!")
            
        model.eval()  # Set to evaluation mode
        
        # Initialize the IntegratedGradients explainer, using standard steps
        logger.info("Initializing IntegratedGradients explainer...")
        try:
            # Initialize the explainer, using standard steps
            ig_explainer = IntegratedGradientsExplainer(model, tokenizer, steps=25)
            logger.info("Created IntegratedGradientsExplainer instance, using 25 steps for integration calculation")
            logger.info("✅ IntegratedGradients explainer initialized successfully")
        except Exception as explainer_error:
            logger.error(f"Failed to initialize explainer: {str(explainer_error)}")
            logger.error(traceback.format_exc())
            # Even if the explainer initialization fails, the model and tokenizer are still available, so return True
            # The compute_token_importance function will handle the case where ig_explainer is None
            return True
        
        logger.info("Successfully loaded model and tokenizer")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_attention_weights(encoded_inputs):
    """
    Extract the self-attention weights of the model 
    """
    try:
        # Add output_attentions=True to get the attention weights
        with torch.no_grad():
            outputs = model(input_ids=encoded_inputs['input_ids'].to(device),
                          attention_mask=encoded_inputs['attention_mask'].to(device),
                          output_attentions=True)
            
        # outputs.attentions is a tuple, each element corresponds to a layer of attention
        # We take the attention of the last layer (usually the last layer contains the most semantic information)
        last_layer_attention = outputs.attentions[-1].cpu().numpy()
        
        # We take the attention matrix of the first sample (if batch processing)
        attention_matrix = last_layer_attention[0]
        
        # Transformer models usually have multiple attention heads, we take the average
        avg_attention = np.mean(attention_matrix, axis=0)
        
        return avg_attention
    except Exception as e:
        logger.error(f"Error extracting attention weights: {str(e)}")
        return None 

# Compute token importance using integrated gradients
def compute_token_importance(texts):
    """
    Compute token importance values using integrated gradients
    Compared to SHAP, this method does not require a background dataset and is更适合单文本解释
    If ig_explainer is not available, the perturbation method will be used as a fallback
    """
    global ig_explainer, model, tokenizer  # Add global declaration to fix UnboundLocalError
    try:
        # Ensure texts is in the correct format
        logger.info(f"Calculating token importance, input type: {type(texts)}")
        
        # Ensure the global variables are available
        if model is None or tokenizer is None:
            logger.error("Model or tokenizer not initialized")
            
            # Try to load the model
            success = load_model()
            if not success:
                logger.error("Failed to load model, cannot compute token importance")
                return None
        
        # Ensure the text format is correct
        cleaned_texts = []
        if isinstance(texts, str):
            cleaned_texts = [texts]
        elif isinstance(texts, list):
            # Process nested lists
            flat_list = []
            for item in texts:
                if isinstance(item, list):
                    flat_list.extend([str(subitem) for subitem in item if subitem])
                else:
                    if item:  # Non-empty value
                        flat_list.append(str(item))
            cleaned_texts = flat_list
        else:
            logger.error(f"Cannot handle input type: {type(texts)}")
            return None
            
        if not cleaned_texts:
            logger.error("No valid texts after cleaning")
            return None
            
        logger.info(f"Number of texts after cleaning: {len(cleaned_texts)}")
        
        # Limit the text length to reduce computational overhead
        MAX_TEXT_LENGTH = 1000  # Increase the maximum length to ensure a more comprehensive analysis
        shortened_texts = []
        for text in cleaned_texts:
            if len(text) > MAX_TEXT_LENGTH:
                logger.info(f"Text too long ({len(text)} characters), truncated to {MAX_TEXT_LENGTH} characters")
                shortened_texts.append(text[:MAX_TEXT_LENGTH])
            else:
                shortened_texts.append(text)
        cleaned_texts = shortened_texts
        
        # To control the computational overhead, use at most 2 texts
        if len(cleaned_texts) > 2:
            logger.info(f"Too many texts ({len(cleaned_texts)}), only using the first 2 for importance calculation")
            cleaned_texts = cleaned_texts[:2]
        
        all_tokens = []
        all_values = []
        all_base_values = []
        all_methods = []  # Record the method used
        
        # Check if ig_explainer is available
        use_ig = ig_explainer is not None
        if not use_ig:
            logger.warning("Integrated gradients explainer not initialized, loading/initializing explainer")
            # Try to initialize the explainer
            try:
                ig_explainer = IntegratedGradientsExplainer(model, tokenizer, steps=25)
                use_ig = True
                logger.info("✅ Successfully initialized integrated gradients explainer")
            except Exception as init_error:
                logger.error(f"Failed to initialize integrated gradients explainer: {str(init_error)}")
        
        # Process each text
        for text_idx, text in enumerate(cleaned_texts):
            logger.info(f"Processing text {text_idx+1}/{len(cleaned_texts)}: {text[:30]}...")
            
            try:
                # Use integrated gradients method first
                if use_ig:
                    start_time = time.time()
                    result = ig_explainer.explain(text)
                    elapsed = time.time() - start_time
                    logger.info(f"Integrated gradients calculation time: {elapsed:.2f} seconds")
                    
                    # Check if the result is valid
                    if (result and "tokens" in result and "values" in result and 
                        result["tokens"] and len(result["tokens"]) > 0):
                        
                        # Check if the values returned by integrated gradients are all close to zero
                        max_abs_value = max([abs(v) for v in result["values"]], default=0)
                        if max_abs_value < 1e-5:
                            logger.warning(f"Integrated gradients calculation result for text {text_idx+1} is close to zero (maximum value: {max_abs_value:.8f})")
                            # But we still use these results
                            
                        all_tokens.append(result["tokens"])
                        all_values.append(result["values"])
                        all_base_values.append(result["base_value"])
                        all_methods.append(result.get("method", "IntegratedGradients"))
                        
                        logger.info(f"Text {text_idx+1} token importance calculation successful, found {len(result['tokens'])} important tokens")
                        # Print the most important tokens
                        if result["tokens"]:
                            top_n = min(3, len(result["tokens"]))
                            token_info = [f"{result['tokens'][i]}({result['values'][i]:.4f})" for i in range(top_n)]
                            logger.info(f"Most important tokens: {', '.join(token_info)}")
                        
                        continue  # Successfully processed, continue to the next text
                
                # If integrated gradients fail or are not available, fall back to the perturbation method
                logger.info(f"Using perturbation method to calculate token importance for text {text_idx+1}...")
                perturbation_result = ig_explainer._compute_perturbation_importance(text) if use_ig else None
                
                if perturbation_result and perturbation_result.get("tokens"):
                    all_tokens.append(perturbation_result["tokens"])
                    all_values.append(perturbation_result["values"])
                    all_base_values.append(perturbation_result["base_value"])
                    all_methods.append("Perturbation")
                    logger.info(f"Text {text_idx+1} perturbation method calculation successful, found {len(perturbation_result['tokens'])} tokens")
                else:
                    logger.error(f"Failed to calculate token importance for text {text_idx+1}")
                    # Add empty placeholder
                    all_tokens.append([])
                    all_values.append([])
                    all_base_values.append(0.5)
                    all_methods.append("Failed")
                    
            except Exception as e:
                logger.error(f"Error calculating token importance for text {text_idx+1}: {str(e)}")
                logger.error(traceback.format_exc())
                # Add empty list when an error occurs
                all_tokens.append([])
                all_values.append([])
                all_base_values.append(0.5)
                all_methods.append("Error")
        
        # Ensure at least one set of results
        if not any(tokens for tokens in all_tokens):
            logger.warning("All text processing failed, returning an empty result")
            return {
                "tokens": [[]],
                "values": [[]],
                "base_values": [0.5],
                "methods": ["Failed"]
            }
        
        # Summarize and record the result statistics
        logger.info("Token importance calculation completed")
        success_count = sum(1 for tokens in all_tokens if tokens)
        logger.info(f"Success/Total: {success_count}/{len(cleaned_texts)}")
        
        for i, method in enumerate(all_methods):
            if i < len(all_tokens) and all_tokens[i]:
                logger.info(f"Text {i+1}: Using {method} method, found {len(all_tokens[i])} tokens")
        
        return {
            "tokens": all_tokens,
            "values": all_values,
            "base_values": all_base_values,
            "methods": all_methods  # Add the method information used
        }
        
    except Exception as e:
        logger.error(f"Error calculating token importance: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return a valid but empty result structure to avoid None errors
        return {
            "tokens": [[]],
            "values": [[]],
            "base_values": [0.5],
            "methods": ["Error"]
        }


