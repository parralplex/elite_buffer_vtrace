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

    def start(self):
        Thread(name="SafeWriter", target=self.internal_writer).start()

    def write(self, data, queue_index=0):
        self.queue_list[queue_index].put(data)

    def internal_writer(self):
        for i in range(len(self.files_urls)):
            self.file_desc_list.append(open(self.files_urls[i], "w", 1))
        while not self.finished:
            for i in range(len(self.files_urls)):
                try:
                    data = self.queue_list[i].get(False)
                except Empty:
                    continue
                for data_part in data:
                    if self.file_desc_list[i].closed:
                        break
                    self.file_desc_list[i].writelines(str(data_part) + '\n')
                self.queue_list[i].task_done()

    def close(self):
        for i in range(len(self.files_urls)):
            self.queue_list[i].join()
        self.finished = True
        for i in range(len(self.files_urls)):
            self.file_desc_list[i].close()
