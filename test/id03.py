from BINoculars.backends import id03
import BINoculars.util
import os

import unittest

class TestCase(unittest.TestCase):
    def setUp(self):
        cfg_unparsed = {}
        specfile = os.path.join(os.path.split(os.getcwd())[0], 'BINoculars-binaries/examples/dataset/sixc_tutorial.spec' )
        cfg_unparsed['specfile'] = specfile
        cfg_unparsed['sdd'] = '1000'
        cfg_unparsed['pixelsize'] = '0.055, 0.055'
        cfg_unparsed['imagefolder'] = specfile.replace('sixc_tutorial.spec', 'images')
        cfg_unparsed['centralpixel'] = '50 ,50'
        self.id03input = id03.EH2(cfg_unparsed)
        self.projection = id03.pixels({'resolution' : '1'})

    def test_IO(self):
        jobs = list(self.id03input.generate_jobs(['820']))
        destination_opts = self.id03input.get_destination_options(['820'])
        imagedata = self.id03input.process_job(jobs[0])
        self.projection.project(*imagedata.next()[1])

    def tearDown(self):
        pass
  
if __name__ == '__main__':
    unittest.main()


