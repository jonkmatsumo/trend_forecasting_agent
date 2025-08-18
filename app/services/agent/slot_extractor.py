"""
Slot Extractor
Advanced parameter extraction from natural language queries.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from app.models.agent_models import AgentIntent
from app.utils.text_normalizer import normalize_views


@dataclass
class ExtractedSlots:
    """Container for extracted slots/parameters."""
    keywords: Optional[List[str]] = None
    horizon: Optional[int] = None
    quantiles: Optional[List[float]] = None
    date_range: Optional[Dict[str, str]] = None
    model_id: Optional[str] = None
    geo: Optional[str] = None
    category: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to slots."""
        return getattr(self, key)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in slots."""
        return hasattr(self, key)


class SlotExtractor:
    """Advanced slot extraction from natural language queries."""
    
    def __init__(self):
        """Initialize the slot extractor."""
        self.time_patterns = self._build_time_patterns()
        self.quantile_patterns = self._build_quantile_patterns()
        self.date_patterns = self._build_date_patterns()
        
    def extract_slots(self, query: str, intent: AgentIntent) -> ExtractedSlots:
        """Extract slots from natural language query.
        
        Args:
            query: The natural language query
            intent: The recognized intent
            
        Returns:
            ExtractedSlots with all found parameters
        """
        # Use text normalizer for dual-view normalization
        norm_loose, norm_strict, _ = normalize_views(query)
        
        slots = ExtractedSlots()
        
        # Route keywords to use norm_loose (preserves case and edge punctuation for better keyword extraction)
        slots.keywords = self._extract_keywords(query, norm_loose)
        
        # Route regex fields to use norm_strict (casefolded and trimmed for consistent regex matching)
        if intent in [AgentIntent.FORECAST, AgentIntent.TRAIN]:
            slots.horizon = self._extract_horizon(norm_strict)
            slots.quantiles = self._extract_quantiles(norm_strict)
            
        elif intent == AgentIntent.EVALUATE:
            slots.model_id = self._extract_model_id(norm_strict)
            
        # Extract common slots using strict normalization (for regex matching)
        slots.geo = self._extract_geo(norm_strict)
        slots.category = self._extract_category(norm_strict)
        
        # Extract date range using strict normalization (for regex matching)
        if intent in [AgentIntent.FORECAST, AgentIntent.SUMMARY, AgentIntent.COMPARE]:
            slots.date_range = self._extract_date_range(norm_strict)
        
        return slots
    
    def _extract_keywords(self, query: str, norm_loose: str) -> Optional[List[str]]:
        """Extract keywords from the query.
        
        Args:
            query: The original natural language query (for quoted extraction)
            norm_loose: Loosely normalized query text (preserves case and edge punctuation)
            
        Returns:
            List of extracted keywords
        """
        keywords = []
        
        # Extract quoted keywords (highest priority) - use original query for quotes
        quoted_keywords = re.findall(r'"([^"]*)"', query)
        keywords.extend(quoted_keywords)
        
        # Extract single quoted keywords - use original query for quotes
        single_quoted_keywords = re.findall(r"'([^']*)'", query)
        keywords.extend(single_quoted_keywords)
        
        # Extract keywords from comparison patterns (vs, versus, etc.) - use norm_loose
        # Note: norm_loose preserves case and edge punctuation for better keyword extraction
        comparison_patterns = [
            r'\b([a-zA-Z0-9\s]+?)\s+vs\.?\s+([a-zA-Z0-9\s]+?)\b',
            r'\b([a-zA-Z0-9\s]+?)\s+versus\s+([a-zA-Z0-9\s]+?)\b',
            r'\b([a-zA-Z0-9\s]+?)\s+compared\s+to\s+([a-zA-Z0-9\s]+?)\b',
            r'\b([a-zA-Z0-9\s]+?)\s+and\s+([a-zA-Z0-9\s]+?)\b'
        ]
        
        for pattern in comparison_patterns:
            matches = re.findall(pattern, norm_loose, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    for keyword in match:
                        keyword = keyword.strip()
                        if len(keyword) > 2 and keyword not in keywords:
                            keywords.append(keyword)
                else:
                    keyword = match.strip()
                    if len(keyword) > 2 and keyword not in keywords:
                        keywords.append(keyword)
        
        # Extract keywords after "for" or "about" - use norm_loose
        # Note: norm_loose preserves case and edge punctuation for better keyword extraction
        for_patterns = [
            r'\bfor\s+([a-zA-Z0-9\s]+?)(?:\s+(?:next|in|over|during|the|a|an|this|that))',
            r'\babout\s+([a-zA-Z0-9\s]+?)(?:\s+(?:next|in|over|during|the|a|an|this|that))',
            r'\bof\s+([a-zA-Z0-9\s]+?)(?:\s+(?:next|in|over|during|the|a|an|this|that))',
            r'\babout\s+([a-zA-Z0-9\s]+?)(?:\s+(?:trends?|data|information|summary))',
            r'\b(?:forecast|predict|analyze|compare)\s+([a-zA-Z0-9\s]+?)(?:\s+(?:for|next|in|over|during|the|a|an|this|that))',
            r'\b(?:trends?|data|information)\s+(?:for|about|of)\s+([a-zA-Z0-9\s]+?)(?:\s+(?:next|in|over|during|the|a|an|this|that))'
        ]
        
        for pattern in for_patterns:
            matches = re.findall(pattern, norm_loose, re.IGNORECASE)
            for match in matches:
                keyword = match.strip()
                if len(keyword) > 2 and keyword not in keywords:
                    keywords.append(keyword)
        
        # Extract keywords before common stop words - use norm_loose
        # Note: norm_loose preserves case and edge punctuation for better keyword extraction
        stop_words = ['next', 'in', 'over', 'during', 'the', 'a', 'an', 'this', 'that', 'will', 'is', 'are', 'what', 'how', 'when', 'where', 'why', 'who', 'for', 'with', 'and', 'or', 'but', 'to', 'from', 'by', 'at', 'on', 'up', 'down', 'out', 'off', 'through', 'between', 'among', 'within', 'without', 'against', 'toward', 'towards', 'into', 'onto', 'upon', 'about', 'above', 'below', 'beneath', 'under', 'over', 'across', 'along', 'around', 'behind', 'before', 'after', 'since', 'until', 'while', 'during', 'throughout', 'despite', 'except', 'besides', 'like', 'unlike', 'as', 'than', 'per', 'via', 'versus', 'vs', 'week', 'month', 'year', 'day', 'days', 'weeks', 'months', 'years', 'last', 'first', 'current', 'recent', 'latest', 'previous', 'trends', 'trend', 'data', 'information', 'summary', 'overview', 'insights', 'performance', 'accuracy', 'metrics', 'scores', 'results', 'health', 'working', 'alive', 'okay', 'cache', 'stats', 'statistics', 'list', 'show', 'display', 'models', 'model', 'train', 'build', 'create', 'develop', 'evaluate', 'assess', 'test']
        words = norm_loose.split()
        
        for i, word in enumerate(words):
            if i < len(words) - 1 and words[i + 1].lower() in stop_words:
                # Check if this looks like a keyword (not a common word)
                if len(word) > 2 and word.lower() not in stop_words:
                    if word not in keywords:
                        keywords.append(word)
        
        # Filter out common question words and stop words
        filtered_keywords = []
        for keyword in keywords:
            # Clean up the keyword
            keyword = keyword.strip()
            if not keyword or len(keyword) < 2:
                continue
                
            # Skip if it's a stop word
            if keyword.lower() in stop_words:
                continue
                
            # Skip if it's just a single letter or number
            if len(keyword) == 1 and not keyword.isalpha():
                continue
                
            # Skip if it's a common question word
            if keyword.lower() in ['what', 'how', 'when', 'where', 'why', 'who', 'which']:
                continue
                
            # Check if this keyword is a substring of any existing keyword
            is_substring = False
            for existing in filtered_keywords:
                if keyword.lower() in existing.lower() or existing.lower() in keyword.lower():
                    is_substring = True
                    break
            
            if not is_substring:
                filtered_keywords.append(keyword)
        
        # If we have multi-word keywords, also extract individual meaningful words
        individual_keywords = []
        for keyword in filtered_keywords:
            if ' ' in keyword:  # Multi-word keyword
                words = keyword.split()
                for word in words:
                    if len(word) > 2 and word.lower() not in stop_words:
                        individual_keywords.append(word)
        
        # Combine and deduplicate
        all_keywords = filtered_keywords + individual_keywords
        unique_keywords = []
        for keyword in all_keywords:
            if keyword not in unique_keywords:
                unique_keywords.append(keyword)
        
        return unique_keywords if unique_keywords else None
    
    def _extract_horizon(self, query: str) -> Optional[int]:
        """Extract forecast horizon from query.
        
        Args:
            query: Strictly normalized query text (casefolded and trimmed)
                   Uses norm_strict for consistent regex matching
            
        Returns:
            Horizon in days, or None if not found
        """
        # Check for explicit time expressions
        for pattern, days in self.time_patterns.items():
            if re.search(pattern, query):
                return days
        
        # Check for numeric expressions
        numeric_patterns = [
            (r'(\d+)\s+days?', lambda x: int(x)),
            (r'(\d+)\s+weeks?', lambda x: int(x) * 7),
            (r'(\d+)\s+months?', lambda x: int(x) * 30),
            (r'(\d+)\s+years?', lambda x: int(x) * 365)
        ]
        
        for pattern, converter in numeric_patterns:
            match = re.search(pattern, query)
            if match:
                days = converter(match.group(1))
                # Clamp to reasonable range
                return min(max(days, 1), 365)
        
        return None
    
    def _extract_quantiles(self, query: str) -> Optional[List[float]]:
        """Extract quantiles from query.
        
        Args:
            query: Strictly normalized query text (casefolded and trimmed)
                   Uses norm_strict for consistent regex matching
            
        Returns:
            List of quantiles, or None if not found
        """
        quantiles = []
        
        # Check for explicit quantile patterns (from quantile_patterns)
        for pattern, quantile in self.quantile_patterns.items():
            if re.search(pattern, query):
                if quantile not in quantiles:  # Avoid duplicates
                    quantiles.append(quantile)
        
        # Enhanced quantile patterns with case variations and whitespace
        # B3.2: Support for P10, p 10, p10 formats
        # B3.4: Handle whitespace variations
        enhanced_p_patterns = [
            r'\bp\s*(\d+)\b',  # p 10, p10
            r'\bP\s*(\d+)\b',  # P 10, P10
        ]
        
        for pattern in enhanced_p_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                percentile = int(match) / 100.0
                # B3.5: Ensure proper validation (0 < q < 1)
                if 0 < percentile < 1 and percentile not in quantiles:
                    quantiles.append(percentile)
        
        # B3.3: Add support for percentage notation (90%, 90 %)
        percentage_patterns = [
            r'(\d+)\s*%',  # 90%, 90 %
            r'\b(\d+)\s+percent\b',  # 90 percent
        ]
        
        for pattern in percentage_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                percentile = int(match) / 100.0
                # B3.5: Ensure proper validation (0 < q < 1)
                if 0 < percentile < 1 and percentile not in quantiles:
                    quantiles.append(percentile)
        
        # Enhanced percentile expressions with whitespace variations
        enhanced_percentile_patterns = [
            r'\b(\d+)th\s+percentile\b',  # 10th percentile
            r'\b(\d+)\s*th\s+percentile\b',  # 10 th percentile
        ]
        
        for pattern in enhanced_percentile_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                percentile = int(match) / 100.0
                # B3.5: Ensure proper validation (0 < q < 1)
                if 0 < percentile < 1 and percentile not in quantiles:
                    quantiles.append(percentile)
        
        # Check for "confidence interval" expressions
        if 'confidence' in query and 'interval' in query:
            # Default confidence intervals
            if '90%' in query or '90 percent' in query:
                quantiles = [0.05, 0.95]
            elif '95%' in query or '95 percent' in query:
                quantiles = [0.025, 0.975]
            elif '99%' in query or '99 percent' in query:
                quantiles = [0.005, 0.995]
        elif 'confidence' in query:
            # Handle confidence without "interval"
            if '90%' in query or '90 percent' in query:
                quantiles = [0.05, 0.95]
            elif '95%' in query or '95 percent' in query:
                quantiles = [0.025, 0.975]
            elif '99%' in query or '99 percent' in query:
                quantiles = [0.005, 0.995]
        
        # Sort quantiles for consistent output
        quantiles.sort()
        
        return quantiles if quantiles else None
    
    def _extract_date_range(self, query: str) -> Optional[Dict[str, str]]:
        """Extract date range from query.
        
        Args:
            query: Strictly normalized query text (casefolded and trimmed)
                   Uses norm_strict for consistent regex matching
            
        Returns:
            Dictionary with start_date and end_date, or None
        """
        # Check for explicit date patterns (YYYY-MM-DD to YYYY-MM-DD)
        explicit_date_pattern = r'(\d{4}-\d{2}-\d{2})\s+(?:to|until|through)\s+(\d{4}-\d{2}-\d{2})'
        match = re.search(explicit_date_pattern, query)
        if match:
            return {
                'start_date': match.group(1),
                'end_date': match.group(2)
            }
        
        # Check for "from X to Y" pattern
        from_to_pattern = r'from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})'
        match = re.search(from_to_pattern, query)
        if match:
            return {
                'start_date': match.group(1),
                'end_date': match.group(2)
            }
        
        # Check for relative time expressions with "last X days"
        last_days_pattern = r'last\s+(\d+)\s+days?'
        match = re.search(last_days_pattern, query)
        if match:
            days = int(match.group(1))
            now = datetime.now()
            start_date = (now - timedelta(days=days)).strftime('%Y-%m-%d')
            end_date = now.strftime('%Y-%m-%d')
            return {'start_date': start_date, 'end_date': end_date}
        
        # Check for explicit date patterns
        for pattern, date_range in self.date_patterns.items():
            if re.search(pattern, query):
                return date_range
        
        # Check for relative time expressions
        now = datetime.now()
        
        if 'last week' in query:
            start_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
            end_date = now.strftime('%Y-%m-%d')
            return {'start_date': start_date, 'end_date': end_date}
        
        elif 'last month' in query:
            start_date = (now - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = now.strftime('%Y-%m-%d')
            return {'start_date': start_date, 'end_date': end_date}
        
        elif 'last year' in query:
            start_date = (now - timedelta(days=365)).strftime('%Y-%m-%d')
            end_date = now.strftime('%Y-%m-%d')
            return {'start_date': start_date, 'end_date': end_date}
        
        elif 'last 90 days' in query:
            start_date = (now - timedelta(days=90)).strftime('%Y-%m-%d')
            end_date = now.strftime('%Y-%m-%d')
            return {'start_date': start_date, 'end_date': end_date}
        
        return None
    
    def _extract_model_id(self, query: str) -> Optional[str]:
        """Extract model ID from query.
        
        Args:
            query: Strictly normalized query text (casefolded and trimmed)
                   Uses norm_strict for consistent regex matching
            
        Returns:
            Model ID, or None if not found
        """
        # Look for UUID patterns
        uuid_pattern = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
        match = re.search(uuid_pattern, query)
        if match:
            return match.group(0)
        
        # Look for "model" followed by identifier
        model_pattern = r'model\s+([a-zA-Z0-9_-]+)'
        match = re.search(model_pattern, query)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_geo(self, query: str) -> Optional[str]:
        """Extract geographic location from query.
        
        Args:
            query: Strictly normalized query text (casefolded and trimmed)
                   Uses norm_strict for consistent regex matching
            
        Returns:
            Geographic location, or None if not found
        """
        geo_patterns = [
            r'\b(united states|us|usa)\b',
            r'\b(united kingdom|uk|britain)\b',
            r'\b(canada)\b',
            r'\b(australia)\b',
            r'\b(germany)\b',
            r'\b(france)\b',
            r'\b(spain)\b',
            r'\b(italy)\b',
            r'\b(japan)\b',
            r'\b(china)\b',
            r'\b(india)\b',
            r'\b(brazil)\b'
        ]
        
        # Map abbreviations to full names (case insensitive)
        geo_mapping = {
            'us': 'united states',
            'usa': 'united states',
            'uk': 'united kingdom',
            'britain': 'united kingdom'
        }
        
        for pattern in geo_patterns:
            match = re.search(pattern, query)
            if match:
                geo = match.group(1)
                # Map to full name if it's an abbreviation
                return geo_mapping.get(geo, geo)
        
        return None
    
    def _extract_category(self, query: str) -> Optional[str]:
        """Extract category from query.
        
        Args:
            query: Strictly normalized query text (casefolded and trimmed)
                   Uses norm_strict for consistent regex matching
            
        Returns:
            Category, or None if not found
        """
        category_patterns = [
            r'\b(technology|tech)\b',
            r'\b(business|finance)\b',
            r'\b(entertainment|movies|music)\b',
            r'\b(sports|fitness)\b',
            r'\b(health|medical)\b',
            r'\b(education|learning)\b',
            r'\b(politics|news)\b',
            r'\b(shopping|retail)\b',
            r'\b(travel|tourism)\b',
            r'\b(food|cooking)\b'
        ]
        
        for pattern in category_patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)
        
        return None
    
    def _build_time_patterns(self) -> Dict[str, int]:
        """Build time expression patterns."""
        return {
            r'\bnext week\b': 7,
            r'\bnext month\b': 30,
            r'\bnext year\b': 365,
            r'\bin a week\b': 7,
            r'\bin a month\b': 30,
            r'\bin a year\b': 365,
            r'\bover the next week\b': 7,
            r'\bover the next month\b': 30,
            r'\bover the next year\b': 365,
            r'\bfor the next week\b': 7,
            r'\bfor the next month\b': 30,
            r'\bfor the next year\b': 365
        }
    
    def _build_quantile_patterns(self) -> Dict[str, float]:
        """Build quantile expression patterns."""
        return {
            # B3.1: Enhanced quantile regex patterns
            # B3.2: Support for P10, p 10, p10 formats
            r'\bp10\b': 0.1,
            r'\bP10\b': 0.1,
            r'\bp\s*10\b': 0.1,
            r'\bP\s*10\b': 0.1,
            r'\bp25\b': 0.25,
            r'\bP25\b': 0.25,
            r'\bp\s*25\b': 0.25,
            r'\bP\s*25\b': 0.25,
            r'\bp50\b': 0.5,
            r'\bP50\b': 0.5,
            r'\bp\s*50\b': 0.5,
            r'\bP\s*50\b': 0.5,
            r'\bp75\b': 0.75,
            r'\bP75\b': 0.75,
            r'\bp\s*75\b': 0.75,
            r'\bP\s*75\b': 0.75,
            r'\bp90\b': 0.9,
            r'\bP90\b': 0.9,
            r'\bp\s*90\b': 0.9,
            r'\bP\s*90\b': 0.9,
            # B3.4: Handle whitespace variations in percentile expressions
            r'\b10th\s+percentile\b': 0.1,
            r'\b10\s*th\s+percentile\b': 0.1,
            r'\b25th\s+percentile\b': 0.25,
            r'\b25\s*th\s+percentile\b': 0.25,
            r'\b50th\s+percentile\b': 0.5,
            r'\b50\s*th\s+percentile\b': 0.5,
            r'\b75th\s+percentile\b': 0.75,
            r'\b75\s*th\s+percentile\b': 0.75,
            r'\b90th\s+percentile\b': 0.9,
            r'\b90\s*th\s+percentile\b': 0.9,
            r'\bmedian\b': 0.5
        }
    
    def _build_date_patterns(self) -> Dict[str, Dict[str, str]]:
        """Build date range patterns."""
        now = datetime.now()
        
        return {
            r'\btoday\b': {
                'start_date': now.strftime('%Y-%m-%d'),
                'end_date': now.strftime('%Y-%m-%d')
            },
            r'\byesterday\b': {
                'start_date': (now - timedelta(days=1)).strftime('%Y-%m-%d'),
                'end_date': (now - timedelta(days=1)).strftime('%Y-%m-%d')
            },
            r'\bthis week\b': {
                'start_date': (now - timedelta(days=7)).strftime('%Y-%m-%d'),
                'end_date': now.strftime('%Y-%m-%d')
            },
            r'\bthis month\b': {
                'start_date': (now - timedelta(days=30)).strftime('%Y-%m-%d'),
                'end_date': now.strftime('%Y-%m-%d')
            }
        } 