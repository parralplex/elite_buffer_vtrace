import unittest

from compression_test import TestCompression
from reproducibility_test import ReproducibilityTest
from v_trace_test import VTraceTest
from worker_test import WorkerTest
from stats_test import TestStats

if __name__ == '__main__':
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestCompression))
    test_suite.addTest(unittest.makeSuite(WorkerTest))
    test_suite.addTest(unittest.makeSuite(TestStats))
    test_suite.addTest(unittest.makeSuite(VTraceTest))
    test_suite.addTest(unittest.makeSuite(ReproducibilityTest))
    runner = unittest.TextTestRunner()
    runner.run(test_suite)
