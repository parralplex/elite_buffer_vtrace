import logging

logger = logging.getLogger('Debug_logger')


def create_logger(save_url):
    global logger
    new_logger = logging.getLogger('Debug_logger')
    new_logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(processName)s - %(threadName)s - %(levelname)s - %(message)s')

    stream_handler.setFormatter(formatter)

    new_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(filename=save_url + '/debug.log', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(threadName)s - %(levelname)s - %(message)s', '%H:%M:%S')

    file_handler.setFormatter(formatter)
    new_logger.addHandler(file_handler)
    logger = new_logger


def change_logger_file_handler(save_url):
    global logger
    logger.handlers[1].close()
    logger.removeHandler(logger.handlers[1])
    file_handler = logging.FileHandler(filename=save_url + '/debug.log', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(threadName)s - %(levelname)s - %(message)s',
                                  '%H:%M:%S')

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)