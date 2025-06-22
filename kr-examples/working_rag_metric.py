# working_rag_metric.py - Save this file in your project directory
import re
import json
from typing import Any, Dict, Union
from difflib import SequenceMatcher

class WorkingRAGMetric:
    """Simple and effective RAG metric for text-based answers."""
    
    def __init__(self, output_field: str = "answer", **kwargs):
        self.output_field = output_field
        
    def __call__(self, gold: Any, pred: Any, trace: bool = False, **kwargs) -> Union[Dict[str, float], float]:
        """
        Evaluate prediction against gold standard.
        
        Args:
            gold: Ground truth (can be dict, string, or object with answer field)
            pred: Prediction (can be dict, string, or object with answer field)  
            trace: If True, return detailed scores; if False, return single score
            
        Returns:
            Float score (0.0 to 1.0) or dict of scores if trace=True
        """
        try:
            # Extract text answers
            gold_text = self._extract_text(gold)
            pred_text = self._extract_text(pred)
            
            if not gold_text or not pred_text:
                return {"exact_match": 0.0, "similarity": 0.0, "total": 0.0} if trace else 0.0
            
            # Normalize texts
            gold_clean = self._normalize_text(gold_text)
            pred_clean = self._normalize_text(pred_text)
            
            # Calculate metrics
            exact_match = 1.0 if gold_clean.lower() == pred_clean.lower() else 0.0
            similarity = SequenceMatcher(None, gold_clean.lower(), pred_clean.lower()).ratio()
            
            # Bonus for partial matches
            if gold_clean.lower() in pred_clean.lower() or pred_clean.lower() in gold_clean.lower():
                similarity = max(similarity, 0.7)
            
            # Check for key information presence (for factual answers)
            contains_key_info = self._check_key_info(gold_clean, pred_clean)
            if contains_key_info:
                similarity = max(similarity, 0.8)
            
            total_score = max(exact_match, similarity)
            
            if trace:
                return {
                    "exact_match": exact_match,
                    "similarity": similarity,
                    "contains_key_info": float(contains_key_info),
                    "total": total_score
                }
            else:
                return total_score
                
        except Exception as e:
            print(f"Error in RAG metric: {e}")
            return {"error": 1.0, "total": 0.0} if trace else 0.0
    
    def _extract_text(self, data: Any) -> str:
        """Extract text from various data formats."""
        if isinstance(data, str):
            return data.strip()
        
        if isinstance(data, dict):
            # Try multiple possible field names
            for field in [self.output_field, "answer", "response", "text"]:
                if field in data:
                    value = data[field]
                    if isinstance(value, str):
                        return value.strip()
                    elif isinstance(value, (dict, list)):
                        return str(value)
            
            # Try nested paths
            if "outputs" in data and isinstance(data["outputs"], dict):
                if "answer" in data["outputs"]:
                    return str(data["outputs"]["answer"]).strip()
        
        # Fall back to string conversion
        return str(data).strip()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common punctuation for better matching
        text = re.sub(r'[.,!?;:"\']', '', text)
        
        # Remove articles for better content matching
        text = re.sub(r'\b(the|a|an)\b', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _check_key_info(self, gold: str, pred: str) -> bool:
        """Check if prediction contains key information from gold standard."""
        # Split into words and check overlap
        gold_words = set(gold.lower().split())
        pred_words = set(pred.lower().split())
        
        # Remove common stop words
        stop_words = {'is', 'was', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'}
        gold_words = gold_words - stop_words
        pred_words = pred_words - stop_words
        
        if not gold_words:
            return False
            
        # Calculate word overlap
        overlap = len(gold_words.intersection(pred_words))
        overlap_ratio = overlap / len(gold_words)
        
        # Consider it a match if significant overlap
        return overlap_ratio >= 0.5