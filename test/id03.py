from binoculars.backends import id03
import binoculars.util
import binoculars.space
import os
import numpy

import unittest

class TestCase(unittest.TestCase):
    def setUp(self):
        cfg_unparsed = {}
        specfile = os.path.join(os.path.split(os.getcwd())[0], 'binoculars-binaries/examples/dataset/sixc_tutorial.spec' )
        cfg_unparsed['specfile'] = specfile
        cfg_unparsed['sdd'] = '1000'
        cfg_unparsed['pixelsize'] = '0.055, 0.055'
        cfg_unparsed['imagefolder'] = specfile.replace('sixc_tutorial.spec', 'images')
        cfg_unparsed['centralpixel'] = '50 ,50'
        numpy.save('mask.npy', numpy.identity(516))
        cfg_unparsed['maskmatrix'] = 'mask.npy'
        self.id03input = id03.EH2(cfg_unparsed)
        self.projection = id03.HKLProjection({'resolution' : '0.01', 'limits' : '[0:, :-1, 0:0.2]'})

    def test_IO(self):
        jobs = list(self.id03input.generate_jobs(['820']))
        destination_opts = self.id03input.get_destination_options(['820'])
        imagedata = self.id03input.process_job(jobs[0])
        intensity, weights, coords = imagedata.next()
        projected = self.projection.project(*coords)
        limits = self.projection.config.limits

        space1 = binoculars.space.Space.from_image(self.projection.config.resolution, self.projection.get_axis_labels(), projected, intensity, weights, limits = limits[0])

        print space1
        intensity, weights, coords = imagedata.next()
        projected = self.projection.project(*coords)

        space2 = binoculars.space.Space.from_image(self.projection.config.resolution, self.projection.get_axis_labels(), projected, intensity, weights)
        print space1 + space2

    def tearDown(self):
        os.remove('mask.npy')
  
if __name__ == '__main__':
    unittest.main()


