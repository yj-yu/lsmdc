import Queue
import threading

from videocap.util import log


class BatchQueue(object):
    def __init__(self, batch_iter, max_size=10, name='train'):
        self.queue = Queue.Queue(maxsize=max_size)
        self.batch_iter = batch_iter
        self.name = name

    def thread_close(self):
        self.thread.exit()

    def get_inputs(self):
        batch_chunk = self.queue.get(block=True)
        return batch_chunk

    def thread_main(self):
        for batch_chunk in self.batch_iter:
            self.queue.put(batch_chunk, block=True)

    def start_threads(self):
        log.info("Start {} dataset Queue".format(self.name))
        self.thread = threading.Thread(target=self.thread_main, args=())
        self.thread.daemon = True
        self.thread.start()
