#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import multiprocessing


def _wrapper(func, queue, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        queue.put(result)  # Pass the result back via the queue
    except Exception as e:
        queue.put(e)  # Pass any exception back via the queue


def call_with_timeout(func, timeout, *args, **kwargs):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_wrapper, args=(func, queue, *args), kwargs=kwargs
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()  # Forcefully terminate the process
        process.join()
        raise TimeoutError(
            f"Function {func.__name__} did not complete within {timeout} seconds."
        )

    if not queue.empty():
        result = queue.get()
        if isinstance(result, Exception):
            raise result  # Re-raise the exception if it occurred in the function
        return result
    else:
        raise TimeoutError(
            f"Function {func.__name__} completed but did not return a result."
        )
