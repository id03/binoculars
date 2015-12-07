import binoculars.util
import binoculars.space
import os
import numpy

import unittest

class TestCase(unittest.TestCase):
    def setUp(self):
        fn = 'examples/configs/example_config_id03'
        self.cfg = binoculars.util.ConfigFile.fromtxtfile(fn)

    def test_IO(self):
        test = {'string' : 'string', 'numpy.array' : numpy.arange(10),  'list' : range(10), 'tuple' : tuple(range(10))}
        metasection = binoculars.util.MetaBase()
        metasection.add_section('first', test)
        print metasection

        metadata = binoculars.util.MetaData()
        metadata.add_dataset(metasection)
        metadata.add_dataset(self.cfg)

        metadata.tofile('test.hdf5')

        metadata +=  binoculars.util.MetaData.fromfile('test.hdf5')

        axis = tuple(binoculars.space.Axis(0,10,1,label) for label in ['h', 'k', 'l'])
        axes =  binoculars.space.Axes(axis)
        space = binoculars.space.Space(axes)
        spacedict = dict(z for z in zip('abcde', range(5)))
        dataset = binoculars.util.MetaBase('fromspace', spacedict)
        space.metadata.add_dataset(dataset)

        space.tofile('test2.hdf5')
        testspace = binoculars.space.Space.fromfile('test2.hdf5')

        print (space + testspace).metadata

        print '--------------------------------------------------------'
        print metadata
        print metadata.serialize()
        print binoculars.util.MetaData.fromserial(metadata.serialize())

    def tearDown(self):
        os.remove('test.hdf5')
        os.remove('test2.hdf5')
   
if __name__ == '__main__':
    unittest.main()


