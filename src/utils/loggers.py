import os
import logging
import sys
from typing import Optional

class UTF8StreamHandler(logging.StreamHandler):
    """Custom StreamHandler that enforces UTF-8 encoding"""
    def emit(self, record):
        try:
            msg = self.format(record)
            if hasattr(self.stream, 'buffer'):
                # Binary writing for better encoding support
                self.stream.buffer.write(msg.encode('utf-8') + self.terminator.encode('utf-8'))
            else:
                self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logger(log_dir: Optional[str] = None, 
                log_file: str = 'pipeline.log', 
                level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging with both console and file handlers.
    
    Args:
        log_dir: Directory for log files. If None, file logging is disabled.
        log_file: Name of the log file.
        level: Logging level.
    
    Returns:
        Configured logger instance.
    """
    # Clear any existing handlers
    logging.root.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    # Create console handler with UTF-8 support
    console_handler = UTF8StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    handlers = [console_handler]
    
    # Add file handler if log_dir is specified
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=handlers,
        force=True  # Override any existing handlers
    )
    
    return logging.getLogger(__name__)

# Initialize basic console logging when module is imported
setup_logger(log_dir=None)