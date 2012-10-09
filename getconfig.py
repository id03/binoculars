import ConfigParser
import numpy

class ConfigError(Exception):
    def __init__(self,key):
        self.key = key
    def __str__(self):
        return repr('{0} missing in configfile'.format(self.key))

class Struct(object):
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

def parserange(r):
	if '-' in r:
		a, b = r.split('-')
		return range(int(a), int(b)+1)
	else:
		return [int(r)]

def parsemultirange(s):
	out = []
	ranges = s.split(',')
	for r in ranges:
		out.extend(parserange(r))
	return numpy.asarray(out)

def cfg(configfile):
    configdict = {}
    config = ConfigParser.RawConfigParser()
    config.optionxform = lambda option: option
    config.read(configfile)
    for n in config.sections():
        for m in config.options(n):
            configdict[m] = config.get(n,m)
    test(configdict)
    
    configdict['xmask'] = parsemultirange(configdict['xmask'])
    configdict['ymask'] = parsemultirange(configdict['ymask'])
    
    for n in ['centralpixel','app']:
        configdict[n] = numpy.array(configdict[n].split(','),dtype = numpy.float)
    for n in configdict.keys():
        try: configdict[n] = float(configdict[n])
        except: continue
    
    return Struct(**configdict)

def test(configdict):
    must = ['specfile','imagefolder', 'outputfolder', 'Hres','Kres','Lres','centralpixel','app','ymask','xmask']
    for n in must:
        if n not in configdict.keys(): raise ConfigError(n)

if __name__ == "__main__":
    cfg = cfg('config')

    
