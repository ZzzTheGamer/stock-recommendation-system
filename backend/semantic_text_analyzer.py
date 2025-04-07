"""
Semantic Text Analyzer
Use sentence similarity and financial domain professional vocabulary to analyze important sentences and keywords in text
"""

import re
import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter
import traceback
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import financial domain data
from financial_domain_data import financial_indicators, prompt_words, special_phrases

# Set environment variables to avoid CUDA related warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SemanticTextAnalyzer:
    """
    Text analyzer based on semantic similarity and financial domain knowledge
    Analyze important sentences and keywords in text
    """
    def __init__(self):
        # Load the model lazily to avoid starting time too long
        self._model = None
        self.financial_indicators = financial_indicators
        self.prompt_words = prompt_words
        self.special_phrases = special_phrases
        logger.info("Semantic text analyzer initialized")
    
    @property
    def model(self):
        """Lazy load sentence embedding model"""
        if self._model is None:
            logger.info("Loading sentence vectorization model...")
            try:
                from sentence_transformers import SentenceTransformer
                # Use lightweight model, balance efficiency and quality
                self._model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                logger.info("Sentence vectorization model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load sentence vectorization model: {str(e)}")
                logger.error(traceback.format_exc())
                # Use backup plan
                self._model = None
        return self._model
    
    def analyze_text(self, text: str, section_name: str = "summary") -> Dict[str, Any]:
        """
        Analyze important sentences and keywords in text
        
        Args:
            text: Text to analyze
            section_name: Name of the section the text belongs to

        Returns:
            A dictionary containing the results of important sentences and keywords
        """
        if not text:
            return {
                "success": False,
                "error": "Input text is empty"
            }
            
        try:
            logger.info(f"Starting to analyze {section_name} text, length: {len(text)} characters")
            
            # Split text into sentences, use an improved method to avoid mistaking decimal points in numbers as sentence delimiters
            sentences = self._split_text_to_sentences(text)
            
            if not sentences:
                return {
                    "success": False,
                    "error": "Unable to extract valid sentences"
                }
            
            # Calculate the semantic centrality and keyword weight of the sentences
            sentences_with_scores = self._analyze_sentences(sentences)
            
            # Extract keywords
            important_words = self._extract_keywords(sentences)
            
            # Keep the original text to ensure completeness
            logger.info(f"{section_name} text analysis completed, extracted {len(sentences_with_scores)} sentences and {len(important_words)} keywords")
            
            return {
                "success": True,
                "section": section_name,
                "sentences": sentences_with_scores,
                "importantWords": important_words,
                "originalText": text  # Add the original complete text field
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def _split_text_to_sentences(self, text: str) -> List[str]:
        """
        Improved text segmentation method, avoiding mistaking decimal points in numbers as sentence delimiters
        
        Args:
            text: Text to split
            
        Returns:
            A list of sentences after splitting
        """
        # 1. More precise protection pattern - match all decimal points in numbers
        # This regular expression matches any decimal point surrounded by numbers
        # Pattern: any number (possibly with commas) + decimal point + any number (possibly with a percentage)
        decimal_pattern = r'(\d[\d,]*\.\d+%?)'
        
        # Create a unique identifier to replace decimal points
        DECIMAL_PLACEHOLDER = "__DECIMAL_DOT__"
        
        # 2. First find all decimals in the text
        protected_text = text
        
        # Find all matches of decimals
        decimal_matches = re.finditer(decimal_pattern, text)
        for match in decimal_matches:
            decimal = match.group(0)
            # Create a protected decimal point version
            protected_decimal = decimal.replace('.', DECIMAL_PLACEHOLDER)
            # Replace the decimal in the text
            protected_text = protected_text.replace(decimal, protected_decimal, 1)
        
        # 3. Use a more precise sentence splitting regular expression
        # Match any end punctuation (period, exclamation mark, question mark), but not match the period in URLs and common abbreviations
        # These symbols may be followed by spaces and words starting with uppercase letters
        sentences_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<!\w\.\w)(?<=\.|\?|\!)\s'
        
        # 4. Split sentences
        raw_sentences = re.split(r'[。.!?！？]', protected_text)
        sentences = []
        
        for s in raw_sentences:
            s = s.strip()
            if not s:
                continue
                
            # 5. Restore decimal points
            restored_sentence = s.replace(DECIMAL_PLACEHOLDER, '.')
            sentences.append(restored_sentence)
        
        # 6. Ensure sentences are in order, no sorting
        return sentences
    
    def _analyze_sentences(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Analyze the importance of sentences"""
        try:
            # Calculate sentence similarity and centrality
            centrality_scores = self._calculate_centrality(sentences)
            
            # Calculate the weight of financial indicators and prompt words
            keyword_weights = self._calculate_keyword_weights(sentences)
            
            # Combine scores
            final_scores = []
            for i, (centrality, keyword_weight) in enumerate(zip(centrality_scores, keyword_weights)):
                # Semantic centrality accounted for 60%, and keyword weight accounted for 40%
                combined_score = 0.6 * centrality + 0.4 * keyword_weight
                final_scores.append(combined_score)
            
            # Normalize to a reasonable range (preserve relative differences)
            max_score = max(final_scores) if final_scores else 1.0
            normalized_scores = [(score / max_score) * 3.0 for score in final_scores]  # Maximum value may exceed 100%
            
            # Combine sentences and scores, keep the original order index
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                sentence_scores.append({
                    "text": sentence,
                    "score": float(normalized_scores[i]),
                    "centrality": float(centrality_scores[i]),
                    "keyword_weight": float(keyword_weights[i]),
                    "original_index": i  # Save the original order index
                })
            
            # Sort by score
            sentence_scores.sort(key=lambda x: x["score"], reverse=True)
            
            return sentence_scores
            
        except Exception as e:
            logger.error(f"Error analyzing sentence importance: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fallback plan: use simple keyword matching
            return self._fallback_sentence_analysis(sentences)
    
    def _calculate_centrality(self, sentences: List[str]) -> List[float]:
        """Calculate the centrality score of sentences"""
        if self.model is None:
            # If the model fails to load, use the fallback method
            return [1.0] * len(sentences)
        
        try:
            # Encode sentences
            sentence_embeddings = self.model.encode(sentences)
            
            # Calculate the cosine similarity between sentences
            similarity_matrix = np.zeros((len(sentences), len(sentences)))
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    # Calculate the cosine similarity
                    vec1 = sentence_embeddings[i]
                    vec2 = sentence_embeddings[j]
                    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    similarity_matrix[i][j] = max(0, sim)  # 确保相似度非负
            
            # Calculate the centrality score of each sentence (average similarity with other sentences)
            centrality_scores = np.sum(similarity_matrix, axis=1) / len(sentences)
            
            # Normalize
            max_score = max(centrality_scores) if len(centrality_scores) > 0 else 1.0
            normalized_scores = centrality_scores / max_score if max_score > 0 else centrality_scores
            
            return normalized_scores.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating sentence centrality: {str(e)}")
            logger.error(traceback.format_exc())
            # Return uniform weights when an error occurs
            return [1.0] * len(sentences)
    
    def _calculate_keyword_weights(self, sentences: List[str]) -> List[float]:
        """Calculate the weight of financial indicators and prompt words in each sentence"""
        weights = []
        
        for sentence in sentences:
            # Initial weight
            weight = 1.0
            lower_sentence = sentence.lower()
            
            # Check special phrases (highest priority)
            for phrase, importance in self.special_phrases.items():
                if phrase in lower_sentence:
                    weight += importance * 1.5  # Special phrases have higher weight
            
            # Check financial indicator words
            for indicator, importance in self.financial_indicators.items():
                if indicator in lower_sentence:
                    weight += importance
            
            # Check prompt words
            for prompt, importance in self.prompt_words.items():
                if prompt in lower_sentence:
                    weight += importance
            
            # Check numbers (numerical information)
            num_numbers = len(re.findall(r'\d+(?:\.\d+)?%?', sentence))
            weight += num_numbers * 0.2  # Each number adds 0.2 weight
            
            weights.append(weight)
        
        # Normalize weights
        max_weight = max(weights) if weights else 1.0
        normalized_weights = [w / max_weight for w in weights]
        
        return normalized_weights
    
    def _extract_keywords(self, sentences: List[str]) -> List[Dict[str, float]]:
        """Extract keywords from text"""
        # Merge all sentences
        full_text = " ".join(sentences)
        
        # Simple tokenization (you can use a more complex tokenization method as needed)
        words = []
        for sentence in sentences:
            # Use a regular expression to tokenize, keep Chinese words and English words
            tokens = re.findall(r'[\w]+|[^\s\w]+', sentence)
            words.extend(tokens)
        
        # Calculate word frequency
        word_counter = Counter(words)
        
        # Calculate word importance
        word_scores = {}
        for word, count in word_counter.items():
            if len(word) <= 1:  # Ignore single characters
                continue
                
            score = count
            
            # If it's a financial indicator word, increase the weight
            if word in self.financial_indicators:
                score *= (1 + self.financial_indicators[word] * 2)
            
            # If it's a prompt word, increase the weight
            elif word in self.prompt_words:
                score *= (1 + self.prompt_words[word])
                
            word_scores[word] = score
        
        # Sort by score
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take the top 15 keywords
        top_words = sorted_words[:15]
        
        # Normalize the importance score
        max_score = max(word_scores.values()) if word_scores else 1.0
        normalized_words = [
            {"word": word, "importance": score / max_score}
            for word, score in top_words
        ]
        
        return normalized_words
    
    def _fallback_sentence_analysis(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Fallback sentence analysis method, used when the main method fails"""
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            # Initial score
            score = 1.0
            lower_sentence = sentence.lower()
            
            # Simple keyword matching
            for indicator in self.financial_indicators:
                if indicator in lower_sentence:
                    score += 0.5
            
            for prompt in self.prompt_words:
                if prompt in lower_sentence:
                    score += 0.3
            
            # Check numbers
            num_numbers = len(re.findall(r'\d+(?:\.\d+)?%?', sentence))
            score += num_numbers * 0.2
            
            # Position weight (sentences at the beginning and end may be more important)
            if i == 0 or i == len(sentences) - 1:
                score *= 1.2
            
            sentence_scores.append({
                "text": sentence,
                "score": float(score),
                "centrality": 1.0,  # The fallback method cannot calculate centrality
                "keyword_weight": float(score)
            })
        
        # Normalize
        max_score = max(item["score"] for item in sentence_scores) if sentence_scores else 1.0
        for item in sentence_scores:
            item["score"] = item["score"] / max_score
        
        # Sort by score
        sentence_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return sentence_scores 