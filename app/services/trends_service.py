"""
Trends Service for Google Trends Quantile Forecaster
Handles Google Trends data fetching and processing
"""

import pandas as pd
from pytrends.request import TrendReq
from flask import current_app
import logging

logger = logging.getLogger(__name__)


class TrendsService:
    """Service for handling Google Trends data operations"""
    
    def __init__(self):
        """Initialize the trends service"""
        self.pytrends = TrendReq(
            delay=current_app.config.get('PYTRENDS_DELAY', 1),
            retries=current_app.config.get('PYTRENDS_RETRIES', 3),
            timeout=current_app.config.get('PYTRENDS_TIMEOUT', 30)
        )
    
    def get_trends_data(self, keyword, timeframe='today 12-m', geo=''):
        """
        Get Google Trends data for a keyword
        
        Args:
            keyword (str): Search keyword
            timeframe (str): Time range for trends data
            geo (str): Geographic location code
            
        Returns:
            dict: Trends data with dates and interest values
        """
        try:
            logger.info(f"Fetching trends data for keyword: {keyword}")
            
            # Build payload
            self.pytrends.build_payload(kw_list=[keyword], timeframe=timeframe, geo=geo)
            
            # Get interest over time
            interest_over_time = self.pytrends.interest_over_time()
            
            if interest_over_time.empty:
                logger.warning(f"No trends data found for keyword: {keyword}")
                return {
                    'dates': [],
                    'interest_values': [],
                    'keyword': keyword,
                    'timeframe': timeframe,
                    'geo': geo
                }
            
            # Remove the 'isPartial' column if it exists
            if 'isPartial' in interest_over_time.columns:
                interest_over_time = interest_over_time.drop('isPartial', axis=1)
            
            # Convert to list format
            dates = interest_over_time.index.strftime('%Y-%m-%d').tolist()
            interest_values = interest_over_time[keyword].tolist()
            
            logger.info(f"Successfully fetched {len(dates)} data points for keyword: {keyword}")
            
            return {
                'dates': dates,
                'interest_values': interest_values,
                'keyword': keyword,
                'timeframe': timeframe,
                'geo': geo,
                'data_points': len(dates)
            }
            
        except Exception as e:
            logger.error(f"Error fetching trends data for {keyword}: {str(e)}")
            raise Exception(f"Failed to fetch trends data: {str(e)}")
    
    def get_trends_for_keywords(self, keywords, timeframe='today 12-m', geo=''):
        """
        Get trends data for multiple keywords
        
        Args:
            keywords (list): List of search keywords
            timeframe (str): Time range for trends data
            geo (str): Geographic location code
            
        Returns:
            dict: Trends data for all keywords
        """
        try:
            logger.info(f"Fetching trends data for {len(keywords)} keywords")
            
            # Build payload for multiple keywords
            self.pytrends.build_payload(kw_list=keywords, timeframe=timeframe, geo=geo)
            
            # Get interest over time
            interest_over_time = self.pytrends.interest_over_time()
            
            if interest_over_time.empty:
                logger.warning("No trends data found for any keywords")
                return {}
            
            # Remove the 'isPartial' column if it exists
            if 'isPartial' in interest_over_time.columns:
                interest_over_time = interest_over_time.drop('isPartial', axis=1)
            
            # Convert to dictionary format
            trends_data = {}
            dates = interest_over_time.index.strftime('%Y-%m-%d').tolist()
            
            for keyword in keywords:
                if keyword in interest_over_time.columns:
                    trends_data[keyword] = {
                        'dates': dates,
                        'interest_values': interest_over_time[keyword].tolist(),
                        'data_points': len(dates)
                    }
            
            logger.info(f"Successfully fetched trends data for {len(trends_data)} keywords")
            return trends_data
            
        except Exception as e:
            logger.error(f"Error fetching trends data for multiple keywords: {str(e)}")
            raise Exception(f"Failed to fetch trends data: {str(e)}") 