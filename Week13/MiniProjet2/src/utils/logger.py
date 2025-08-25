"""
Logging utilities for Smart Data Scout
Provides structured logging with rotation and filtering
"""

import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/app.log', mode='a')
    ]
)

# Create logs directory
Path('logs').mkdir(exist_ok=True)

try:
    import streamlit as st
except ImportError:
    st = None

class StreamlitLogHandler(logging.Handler):
    """Custom log handler for Streamlit integration"""
    
    def __init__(self):
        super().__init__()
        self.logs = []
        self.max_logs = 1000
    
    def emit(self, record):
        """Write log message to Streamlit session state"""
        try:
            msg = self.format(record)
            if st and hasattr(st, 'session_state') and 'logs' in st.session_state:
                st.session_state.logs.append(msg)
                # Keep only recent logs
                if len(st.session_state.logs) > self.max_logs:
                    st.session_state.logs = st.session_state.logs[-self.max_logs:]
        except Exception:
            pass

def setup_logging(config=None):
    """Setup logging configuration"""
    
    # Add streamlit handler if available
    if st and 'logs' not in st.session_state:
        st.session_state.logs = []
        
        streamlit_handler = StreamlitLogHandler()
        streamlit_handler.setFormatter(
            logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        )
        logging.getLogger().addHandler(streamlit_handler)

def get_logger(name: str):
    """Get a logger instance"""
    return logging.getLogger(name)

def log_tool_call(tool_name: str, inputs: Dict[str, Any], outputs: Any = None, error: str = None):
    """Log tool call with inputs/outputs (sanitized)"""
    
    # Sanitize inputs (remove sensitive data)
    sanitized_inputs = sanitize_log_data(inputs)
    
    log_entry = {
        "tool": tool_name,
        "inputs": sanitized_inputs,
        "timestamp": datetime.now().isoformat(),
        "success": error is None
    }
    
    if outputs is not None:
        log_entry["outputs"] = sanitize_log_data(outputs)
    
    if error:
        log_entry["error"] = str(error)
    
    logger = get_logger(__name__)
    logger.info(f"Tool call: {json.dumps(log_entry, indent=2)}")
    
    return log_entry

def sanitize_log_data(data: Any) -> Any:
    """Remove sensitive information from log data"""
    
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            # Skip sensitive keys
            if any(sensitive in key.lower() for sensitive in ['key', 'token', 'password', 'secret']):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = sanitize_log_data(value)
        return sanitized
    
    elif isinstance(data, list):
        return [sanitize_log_data(item) for item in data]
    
    elif isinstance(data, str) and len(data) > 1000:
        # Truncate long strings
        return data[:1000] + "... (truncated)"
    
    else:
        return data

# Initialize logging on import
try:
    setup_logging()
except Exception as e:
    print(f"Failed to setup logging: {e}")

# Export commonly used functions
__all__ = ['get_logger', 'log_tool_call', 'setup_logging']
