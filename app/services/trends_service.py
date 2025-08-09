"""
Google Trends Service
Handles fetching and processing Google Trends data
"""

import pandas as pd
import time
import logging
from typing import List, Dict, Any, Optional
from pytrends.request import TrendReq
from datetime import datetime

from app.models.trend_model import TrendData, TrendsRequest, TrendsResponse
from app.utils.validators import InputValidator
from app.utils.error_handlers import TrendsAPIError, ValidationError, RateLimitError


class TrendsService:
    """Service for fetching and processing Google Trends data"""
    
    def __init__(self):
        """Initialize the trends service"""
        self.trend = TrendReq(hl='en-US', tz=360)
        self.validator = InputValidator()
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.rate_limit_counter = 0
        self.last_request_time = 0
        self.max_requests_per_minute = 60  # Conservative limit
        
        # Cache for recent requests (simple in-memory cache)
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
    
    def fetch_trends_data(self, request: TrendsRequest) -> TrendsResponse:
        """
        Fetch Google Trends data for the given request
        
        Args:
            request: Validated TrendsRequest object
            
        Returns:
            TrendsResponse with trend data
            
        Raises:
            TrendsAPIError: If trends API fails
            RateLimitError: If rate limit is exceeded
        """
        try:
            self.logger.info(f"Fetching trends data for keywords: {request.keywords}")
            
            # Check rate limiting
            self._check_rate_limit()
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                self.logger.info("Returning cached trends data")
                return cached_data
            
            # Fetch fresh data from Google Trends
            trend_data_list = self._fetch_from_google_trends(request)
            
            # Create response
            response = TrendsResponse(
                status="success",
                data=trend_data_list,
                request_info=request.to_dict(),
                timestamp=datetime.utcnow()
            )
            
            # Cache the response
            self._cache_data(cache_key, response)
            
            self.logger.info(f"Successfully fetched trends data for {len(trend_data_list)} keywords")
            return response
            
        except Exception as e:
            self.logger.error(f"Error fetching trends data: {str(e)}")
            if isinstance(e, (TrendsAPIError, RateLimitError)):
                raise
            raise TrendsAPIError(f"Failed to fetch trends data: {str(e)}")
    
    def _fetch_from_google_trends(self, request: TrendsRequest) -> List[TrendData]:
        """
        Fetch data from Google Trends API
        
        Args:
            request: Validated TrendsRequest object
            
        Returns:
            List of TrendData objects
            
        Raises:
            TrendsAPIError: If API request fails
        """
        try:
            # Build payload for Google Trends
            self.trend.build_payload(
                kw_list=request.keywords,
                timeframe=request.timeframe,
                geo=request.geo if request.geo else ''
            )
            
            # Get interest over time data
            interest_over_time = self.trend.interest_over_time()
            
            if interest_over_time.empty:
                raise TrendsAPIError("No data returned from Google Trends API")
            
            # Process results into TrendData objects
            trend_data_list = []
            for keyword in request.keywords:
                if keyword in interest_over_time.columns:
                    # Extract dates and values
                    dates = interest_over_time.index.strftime('%Y-%m-%d').tolist()
                    values = interest_over_time[keyword].fillna(0).tolist()
                    
                    # Create TrendData object
                    trend_data = TrendData(
                        keyword=keyword,
                        dates=dates,
                        interest_values=values,
                        timeframe=request.timeframe,
                        geo=request.geo,
                        last_updated=datetime.utcnow()
                    )
                    
                    trend_data_list.append(trend_data)
                else:
                    self.logger.warning(f"Keyword '{keyword}' not found in trends data")
            
            return trend_data_list
            
        except Exception as e:
            self.logger.error(f"Google Trends API error: {str(e)}")
            raise TrendsAPIError(f"Google Trends API request failed: {str(e)}")
    
    def _check_rate_limit(self):
        """
        Check and enforce rate limiting
        
        Raises:
            RateLimitError: If rate limit is exceeded
        """
        current_time = time.time()
        
        # Simple rate limiting: ensure at least 1 second between requests
        if current_time - self.last_request_time < 1:
            sleep_time = 1 - (current_time - self.last_request_time)
            time.sleep(sleep_time)
        
        # Check requests per minute
        if self.rate_limit_counter >= self.max_requests_per_minute:
            raise RateLimitError(
                f"Rate limit exceeded. Maximum {self.max_requests_per_minute} requests per minute allowed.",
                retry_after=60
            )
        
        self.last_request_time = time.time()
        self.rate_limit_counter += 1
    
    def _generate_cache_key(self, request: TrendsRequest) -> str:
        """Generate cache key for the request"""
        return f"{','.join(request.keywords)}_{request.timeframe}_{request.geo}"
    
    def _get_cached_data(self, cache_key: str) -> Optional[TrendsResponse]:
        """Get cached data if available and not expired"""
        if cache_key in self._cache:
            cached_item = self._cache[cache_key]
            if time.time() - cached_item['timestamp'] < self._cache_ttl:
                return cached_item['data']
            else:
                # Remove expired cache entry
                del self._cache[cache_key]
        return None
    
    def _cache_data(self, cache_key: str, response: TrendsResponse):
        """Cache the response data"""
        self._cache[cache_key] = {
            'data': response,
            'timestamp': time.time()
        }
        
        # Simple cache size management (keep only last 100 entries)
        if len(self._cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(self._cache.keys(), 
                               key=lambda k: self._cache[k]['timestamp'])[:20]
            for key in oldest_keys:
                del self._cache[key]
    
    def get_trends_for_keyword(self, keyword: str, timeframe: str = "today 12-m", 
                              geo: str = "") -> TrendData:
        """
        Convenience method to get trends for a single keyword
        
        Args:
            keyword: The keyword to search for
            timeframe: Time frame for the search
            geo: Geographic location (optional)
            
        Returns:
            TrendData object
            
        Raises:
            ValidationError: If input validation fails
            TrendsAPIError: If API request fails
        """
        # Create request object (this will validate the input)
        request = TrendsRequest(
            keywords=[keyword],
            timeframe=timeframe,
            geo=geo
        )
        
        # Fetch data
        response = self.fetch_trends_data(request)
        
        # Return the first (and only) trend data
        if response.data:
            return response.data[0]
        else:
            raise TrendsAPIError(f"No data returned for keyword: {keyword}")
    
    def get_trends_for_keywords(self, keywords: List[str], timeframe: str = "today 12-m", 
                               geo: str = "") -> List[TrendData]:
        """
        Convenience method to get trends for multiple keywords
        
        Args:
            keywords: List of keywords to search for
            timeframe: Time frame for the search
            geo: Geographic location (optional)
            
        Returns:
            List of TrendData objects
            
        Raises:
            ValidationError: If input validation fails
            TrendsAPIError: If API request fails
        """
        # Create request object (this will validate the input)
        request = TrendsRequest(
            keywords=keywords,
            timeframe=timeframe,
            geo=geo
        )
        
        # Fetch data
        response = self.fetch_trends_data(request)
        
        return response.data
    
    def clear_cache(self):
        """Clear the internal cache"""
        self._cache.clear()
        self.logger.info("Trends service cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self._cache),
            'cache_ttl': self._cache_ttl,
            'rate_limit_counter': self.rate_limit_counter,
            'max_requests_per_minute': self.max_requests_per_minute
        }
    
    def get_trends_summary(self, keywords: List[str], timeframe: str = "today 12-m", 
                          geo: str = "") -> Dict[str, Any]:
        """
        Get a summary of trends data for multiple keywords
        
        Args:
            keywords: List of keywords to analyze
            timeframe: Time frame for the search
            geo: Geographic location (optional)
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            # Get trends data
            trend_data_list = self.get_trends_for_keywords(keywords, timeframe, geo)
            
            # Calculate summary statistics
            summary = {
                'keywords': keywords,
                'timeframe': timeframe,
                'geo': geo,
                'total_keywords': len(trend_data_list),
                'keyword_stats': {}
            }
            
            for trend_data in trend_data_list:
                summary['keyword_stats'][trend_data.keyword] = {
                    'average_interest': trend_data.average_interest,
                    'max_interest': trend_data.max_interest,
                    'min_interest': trend_data.min_interest,
                    'data_points': trend_data.data_points,
                    'trend_direction': self._calculate_trend_direction(trend_data)
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating trends summary: {str(e)}")
            raise TrendsAPIError(f"Failed to generate trends summary: {str(e)}")
    
    def _calculate_trend_direction(self, trend_data: TrendData) -> str:
        """
        Calculate the overall trend direction
        
        Args:
            trend_data: TrendData object
            
        Returns:
            Trend direction: 'increasing', 'decreasing', or 'stable'
        """
        if len(trend_data.interest_values) < 2:
            return 'stable'
        
        # Calculate linear trend using simple linear regression
        n = len(trend_data.interest_values)
        x_sum = sum(range(n))
        y_sum = sum(trend_data.interest_values)
        xy_sum = sum(i * val for i, val in enumerate(trend_data.interest_values))
        x2_sum = sum(i * i for i in range(n))
        
        # Calculate slope
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        # Determine direction based on slope
        if slope > 0.5:  # Threshold for significant increase
            return 'increasing'
        elif slope < -0.5:  # Threshold for significant decrease
            return 'decreasing'
        else:
            return 'stable'
    
    def compare_keywords(self, keywords: List[str], timeframe: str = "today 12-m", 
                        geo: str = "") -> Dict[str, Any]:
        """
        Compare trends between multiple keywords
        
        Args:
            keywords: List of keywords to compare
            timeframe: Time frame for the search
            geo: Geographic location (optional)
            
        Returns:
            Dictionary with comparison data
        """
        try:
            # Get trends data
            trend_data_list = self.get_trends_for_keywords(keywords, timeframe, geo)
            
            if len(trend_data_list) < 2:
                raise ValidationError("At least 2 keywords required for comparison")
            
            # Find the keyword with highest average interest
            max_avg_keyword = max(trend_data_list, key=lambda x: x.average_interest)
            
            # Find the keyword with most volatile interest (highest standard deviation)
            most_volatile = max(trend_data_list, key=lambda x: self._calculate_volatility(x))
            
            comparison = {
                'keywords': keywords,
                'timeframe': timeframe,
                'geo': geo,
                'highest_average_interest': {
                    'keyword': max_avg_keyword.keyword,
                    'value': max_avg_keyword.average_interest
                },
                'most_volatile': {
                    'keyword': most_volatile.keyword,
                    'volatility': self._calculate_volatility(most_volatile)
                },
                'keyword_comparison': {}
            }
            
            # Compare each keyword against others
            for i, trend_data in enumerate(trend_data_list):
                comparison['keyword_comparison'][trend_data.keyword] = {
                    'average_interest': trend_data.average_interest,
                    'volatility': self._calculate_volatility(trend_data),
                    'trend_direction': self._calculate_trend_direction(trend_data),
                    'rank': self._calculate_rank(trend_data, trend_data_list)
                }
            
            return comparison
            
        except ValidationError:
            # Re-raise ValidationError as-is
            raise
        except Exception as e:
            self.logger.error(f"Error comparing keywords: {str(e)}")
            raise TrendsAPIError(f"Failed to compare keywords: {str(e)}")
    
    def _calculate_volatility(self, trend_data: TrendData) -> float:
        """Calculate volatility (standard deviation) of interest values"""
        if len(trend_data.interest_values) < 2:
            return 0.0
        
        mean = trend_data.average_interest
        variance = sum((x - mean) ** 2 for x in trend_data.interest_values) / len(trend_data.interest_values)
        return variance ** 0.5
    
    def _calculate_rank(self, trend_data: TrendData, all_trends: List[TrendData]) -> int:
        """Calculate rank of a keyword based on average interest"""
        sorted_trends = sorted(all_trends, key=lambda x: x.average_interest, reverse=True)
        for i, trend in enumerate(sorted_trends):
            if trend.keyword == trend_data.keyword:
                return i + 1
        return len(sorted_trends) 