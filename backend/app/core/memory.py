import os
import psutil
import logging

logger = logging.getLogger("hireagent.memory")

def get_current_rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) in Megabytes."""
    try:
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return 0.0

def log_memory_checkpoint(stage: str, extra: str = "") -> float:
    """Logs the current memory usage for a specific pipeline stage.
    
    Provides lightweight instrumentation without introducing profiling overhead.
    """
    rss = get_current_rss_mb()
    extra_str = f" - {extra}" if extra else ""
    logger.info(f"[MEMORY] [{stage}] Process RSS: {rss:.2f} MB{extra_str}")
    return rss
