from queue import Queue, Empty
from threading import Thread


class SafeOrderedMultiFileWriter:
    def __init__(self, files_urls):
        self.files_urls = files_urls
        self.queue_list = []
        self.file_desc_list = []
        for i in range(len(files_urls)):
            self.queue_list.append(Queue())
        self.finished = False
        self.block_on_get = True
        self.internal_thread = None

    def start(self):
        self.internal_thread = Thread(name="SafeWriter", target=self.internal_writer).start()

    def write(self, data, queue_index=0):
        self.queue_list[queue_index].put(data)

    def internal_writer(self):
        index_list = [i for i in range(len(self.files_urls))]
        for i in range(len(self.files_urls)):
            self.file_desc_list.append(open(self.files_urls[i], "w", 1))
        while len(index_list) > 0:
            for i in index_list:
                try:
                    data = self.queue_list[i].get(timeout=1, block=self.block_on_get)
                except Empty:
                    if self.finished:
                        index_list.remove(i)
                    continue
                for data_part in data:
                    self.file_desc_list[i].writelines(str(data_part) + '\n')
                self.queue_list[i].task_done()

    def close(self):
        self.block_on_get = False
        self.finished = True
        self.close_data_operators()
        if self.internal_thread is not None:
            self.internal_thread.join()

    def close_data_operators(self):
        for i in range(len(self.files_urls)):
            self.queue_list[i].join()
        for i in range(len(self.files_urls)):
            self.file_desc_list[i].close()
