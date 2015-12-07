import binoculars.util
import os

import unittest

class TestCase(unittest.TestCase):
    def setUp(self):
        fn = 'examples/configs/example_config_id03'
        self.cfg = binoculars.util.ConfigFile.fromtxtfile(fn)

    def test_IO(self):
        self.cfg.totxtfile('test.txt')
        self.cfg.tofile('test.hdf5')
        print binoculars.util.ConfigFile.fromfile('test.hdf5')
        self.assertRaises(IOError, binoculars.util.ConfigFile.fromtxtfile, '')
        self.assertRaises(IOError, binoculars.util.ConfigFile.fromfile, '')

    def tearDown(self):
        os.remove('test.txt')
        os.remove('test.hdf5')
   
if __name__ == '__main__':
    unittest.main()


