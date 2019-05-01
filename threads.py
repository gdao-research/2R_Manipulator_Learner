import threading
from contextlib import contextmanager
from six.moves import queue
import logger


class ShareSessionThread(threading.Thread):
    def __init__(self, th=None):
        '''
        Share TensorFlow session between threads
        Args:
            th: threading.Thread or None
        '''
        super(ShareSessionThread, self).__init__()
        if th is not None:
            assert isinstance(th, threading.Thread), th
            self._th = th
            self.name = th.name
            self.daemon = th.daemon

    @contextmanager
    def default_sess(self):
        if self._sess:
            with self._sess.as_default():
                yield self._sess
        else:
            logger.warn(f"ShareSessionThread {self.name} wasn't under a default session!")
            yield None

    def start(self):
        import tensorflow as tf
        self._sess = tf.get_default_session()
        super(ShareSessionThread, self).start()

    def run(self):
        if not self._th:
            raise NotImplementedError()
        with self._sess.as_default():
            self._th.run()


class StopableThread(threading.Thread):
    def __init__(self, event=None):
        '''
        Create a stopable thread
        Args:
            event: threading.Event or None
        '''
        super(StopableThread, self).__init__()
        if event is None:
            self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def stopped(self):
        return self._stop_evt.isSet()

    def queue_put_stopable(self, q, obj):
        ''' Try to put obj to q (queue.Queue), but give up when thread is stopped'''
        while not self.stopped():
            try:
                q.put(obj, timeout=5)
                break
            except queue.Full:
                pass

    def queue_get_stopable(self, q):
        ''' Try to get obj from q, but give up when thread is stopped'''
        while not self.stopped():
            try:
                return q.get(timeout=5)
            except queue.Empty:
                pass


class LoopThread(StopableThread):
    def __init__(self, pausable=True):
        super(LoopThread, self).__init__()
        self.paused = False
        if pausable:
            self._lock = threading.Lock()
        self.daemon = True

    def run(self):
        while not self.stopped():
            if not self.paused:
                raise NotImplementedError  # This is a sample to overide

    def pause(self):
        self.paused = True
        self._lock.acquire()

    def resume(self):
        self.paused = False
        self._lock.release()
