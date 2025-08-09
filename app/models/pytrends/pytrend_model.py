"""
Data models for Google Trends data
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class TrendData:
    """Data model for Google Trends data"""
    
    keyword: str
    dates: List[str]
    interest_values: List[float]
    category: Optional[str] = None
    timeframe: str = "today 12-m"
    geo: str = ""
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate data after initialization"""
        if not self.keyword or not self.keyword.strip():
            raise ValueError("Keyword cannot be empty")
        
        if len(self.dates) != len(self.interest_values):
            raise ValueError("Dates and interest values must have same length")
        
        if len(self.dates) == 0:
            raise ValueError("At least one data point is required")
        
        # Validate interest values are within expected range (0-100)
        for i, value in enumerate(self.interest_values):
            if not isinstance(value, (int, float)):
                raise ValueError(f"Interest value at index {i} must be a number")
            if value < 0 or value > 100:
                raise ValueError(f"Interest value at index {i} must be between 0 and 100")
    
    @property
    def data_points(self) -> int:
        """Get the number of data points"""
        return len(self.dates)
    
    @property
    def average_interest(self) -> float:
        """Calculate average interest value"""
        if not self.interest_values:
            return 0.0
        return sum(self.interest_values) / len(self.interest_values)
    
    @property
    def max_interest(self) -> float:
        """Get maximum interest value"""
        return max(self.interest_values) if self.interest_values else 0.0
    
    @property
    def min_interest(self) -> float:
        """Get minimum interest value"""
        return min(self.interest_values) if self.interest_values else 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            'keyword': self.keyword,
            'dates': self.dates,
            'interest_values': self.interest_values,
            'category': self.category,
            'timeframe': self.timeframe,
            'geo': self.geo,
            'data_points': self.data_points,
            'average_interest': self.average_interest,
            'max_interest': self.max_interest,
            'min_interest': self.min_interest,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TrendData':
        """Create TrendData from dictionary"""
        return cls(
            keyword=data['keyword'],
            dates=data['dates'],
            interest_values=data['interest_values'],
            category=data.get('category'),
            timeframe=data.get('timeframe', 'today 12-m'),
            geo=data.get('geo', ''),
            last_updated=datetime.fromisoformat(data['last_updated']) if data.get('last_updated') else None
        )


@dataclass
class TrendsRequest:
    """Data model for trends API request"""
    
    keywords: List[str]
    timeframe: str = "today 12-m"
    geo: str = ""
    
    def __post_init__(self):
        """Validate request data"""
        if not self.keywords:
            raise ValueError("At least one keyword is required")
        
        if len(self.keywords) > 5:
            raise ValueError("Maximum 5 keywords allowed per request")
        
        # Validate each keyword
        for i, keyword in enumerate(self.keywords):
            if not keyword or not keyword.strip():
                raise ValueError(f"Keyword at index {i} cannot be empty")
            if len(keyword.strip()) > 100:
                raise ValueError(f"Keyword at index {i} is too long (max 100 characters)")
        
        # Clean keywords
        self.keywords = [kw.strip() for kw in self.keywords]
    
    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            'keywords': self.keywords,
            'timeframe': self.timeframe,
            'geo': self.geo
        }


@dataclass
class TrendsResponse:
    """Data model for trends API response"""
    
    status: str
    data: List[TrendData]
    request_info: dict
    timestamp: datetime
    
    def __post_init__(self):
        """Set timestamp if not provided"""
        if not self.timestamp:
            self.timestamp = datetime.utcnow()
    
    @property
    def total_keywords(self) -> int:
        """Get the total number of keywords in the response"""
        return len(self.data)
    
    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            'status': self.status,
            'data': [trend.to_dict() for trend in self.data],
            'request_info': self.request_info,
            'timestamp': self.timestamp.isoformat(),
            'total_keywords': self.total_keywords
        } 