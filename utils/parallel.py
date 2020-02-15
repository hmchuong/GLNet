import threading
import logging
import sys

def map_parallel(f, iter, max_parallel = 4):
    """Just like map(f, iter) but each is done in a separate thread."""
    # Put all of the items in the queue, keep track of order.
    from queue import Queue, Empty
    total_items = 0
    queue = Queue()
    for i, arg in enumerate(iter):
        queue.put((i, arg))
        total_items += 1
    # No point in creating more thread objects than necessary.
    if max_parallel > total_items:
        max_parallel = total_items

    # The worker thread.
    res = {}
    errors = {}
    class Worker(threading.Thread):
        def run(self):
            while not errors:
                try:
                    num, arg = queue.get(block = False)
                    try:
                        res[num] = f(arg)
                    except Exception as e:
                        errors[num] = sys.exc_info()
                except Empty:
                    break

    # Create the threads.
    threads = [Worker() for _ in range(max_parallel)]
    # Start the threads.
    [t.start() for t in threads]
    # Wait for the threads to finish.
    [t.join() for t in threads]

    if errors:
        if len(errors) > 1:
            logging.warning("map_parallel multiple errors: %d:\n%s"%(
                len(errors), errors))
        # Just raise the first one.
        item_i = min(errors.keys())
        type, value, tb = errors[item_i]
        # Print the original traceback
        logging.info("map_parallel exception on item %s/%s:\n%s"%(
            item_i, total_items, "\n".join(traceback.format_tb(tb))))
        raise value
    return [res[i] for i in range(len(res))]