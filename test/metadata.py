import BINoculars.util
import BINoculars.space
import os
import numpy

import unittest

class TestCase(unittest.TestCase):
    def setUp(self):
        fn = 'examples/configs/example_config_id03'
        self.cfg = BINoculars.util.ConfigFile.fromtxtfile(fn)

    def test_IO(self):
        test = {'string' : 'string', 'numpy.array' : numpy.arange(10),  'list' : range(10), 'tuple' : tuple(range(10))}
        metasection = BINoculars.util.MetaBase()
        metasection.add_section('first', test)
        print metasection

        metadata = BINoculars.util.MetaData()
        metadata.add_dataset(metasection)
        metadata.add_dataset(self.cfg)

        metadata.tofile('test.hdf5')

        metadata +=  BINoculars.util.MetaData.fromfile('test.hdf5')

        axis = tuple(BINoculars.space.Axis(0,10,1,label) for label in ['h', 'k', 'l'])
        axes =  BINoculars.space.Axes(axis)
        space = BINoculars.space.Space(axes)
        spacedict = dict(z for z in zip('abcde', range(5)))
        dataset = BINoculars.util.MetaBase('fromspace', spacedict)
        space.metadata.add_dataset(dataset)

        space.tofile('test2.hdf5')
        testspace = BINoculars.space.Space.fromfile('test2.hdf5')

        print (space + testspace).metadata



    def tearDown(self):
        os.remove('test.hdf5')
        os.remove('test2.hdf5')
   
if __name__ == '__main__':
    unittest.main()


