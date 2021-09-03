from queue import Queue, Empty
from threading import Thread


class SafeFileWriter:
    def __init__(self, *args):
        self.writer_args = args
        self.queue = Queue()
        self.finished = False
        self.file_writer = None

    def start(self):
        Thread(name="SafeWriter", target=self.internal_writer).start()

    def write(self, data):
        self.queue.put(data)

    def internal_writer(self):
        self.file_writer = open(*self.writer_args)
        while not self.finished:
            try:
                data = self.queue.get(True, 1)
            except Empty:
                continue
            for data_part in data:
                if self.file_writer.closed:
                    break
                self.file_writer.writelines(str(data_part) + '\n')
            self.queue.task_done()

    def close(self):
        self.queue.join()
        self.finished = True
        self.file_writer.close()