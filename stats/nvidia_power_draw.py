from pynvml import *
import subprocess
from threading import Thread, Event
from utils import logger


class PowerDrawAgent:
    def __init__(self, *args):
        self.writer_args = args
        self.stop_event = Event()
        self.file_desc = None
        self.finished = False

    def start(self):
        Thread(name="PowerDrawWriter", target=self.internal_writer).start()

    def internal_writer(self):
        self.file_desc = open(*self.writer_args)
        nvmlInit()
        gpu_handle = nvmlDeviceGetHandleByIndex(0)
        while not self.stop_event.wait(1):               # wait 1 sec
            if self.finished:
                break
            gpu_wattage = nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0

            cores_power = 0
            package_power = 0

            try:
                output = subprocess.check_output(['rapl'], stderr=subprocess.STDOUT, universal_newlines=True)
                for line in output:
                    parts = str(line, 'ascii').strip().split()
                    if len(parts) == 7 and parts[0] == 'Core':
                        package_power = float(parts[6][:-1])
                    elif len(parts) == 3 and parts[1] == 'sum:':
                        cores_power = float(parts[2][:-1])
                if self.file_desc.closed:
                    break
                self.file_desc.writelines(str(cores_power) + ',' + str(package_power) + ',' + str(gpu_wattage) + '\n')

            except subprocess.CalledProcessError as error:
                logger.exception("Error while executing: ralp. Returned " + str(error.output))

            except FileNotFoundError as error:
                logger.exception("Error " + error.strerror)

            # with subprocess.Popen(['rapl'], stdout=subprocess.PIPE) as rapl:
            #     for line in rapl.stdout:
            #         parts = str(line, 'ascii').strip().split()
            #         if len(parts) == 7 and parts[0] == 'Core':
            #             package_power = float(parts[6][:-1])
            #         elif len(parts) == 3 and parts[1] == 'sum:':
            #             cores_power = float(parts[2][:-1])
            #     if self.file_desc.closed:
            #         break
            #     self.file_desc.writelines(str(cores_power) + ',' + str(package_power) + ',' + str(gpu_wattage) + '\n')

    def close(self):
        self.stop_event.set()
        self.finished = True
        nvmlShutdown()
        self.file_desc.close()







