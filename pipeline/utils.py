import time

def call_with_retries(func, max_retries=3, initial_delay=2, *args, **kwargs):
    """
    Retry a function call with exponential backoff.
    Args:
        func: Callable to execute.
        max_retries: Maximum number of attempts.
        initial_delay: Initial delay in seconds.
        *args, **kwargs: Arguments to pass to func.
    Returns:
        The result of func(*args, **kwargs) if successful.
    Raises:
        The last exception if all retries fail.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[Retry] Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2  # Exponential backoff 