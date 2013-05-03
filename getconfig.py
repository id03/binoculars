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

def parsetuple(s, length=None):
    t = tuple(float(i) for i in s.split(','))
    if length is not None and len(t) != length:
        raise ValueError('invalid configuration value')
    return t

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
    try:
        configdict['resolution'] = parsetuple(configdict['resolution'])
    except:
        pass

    if 'UB' in configdict.keys():
        configdict['UB'] = parsetuple(configdict['UB'],9)

    configdict['centralpixel'] = parsetuple(configdict['centralpixel'], 2)
    configdict['app'] = parsetuple(configdict['app'], 2)

    for n in configdict.keys():
        try: configdict[n] = float(configdict[n])
        except: continue
    
    return Struct(**configdict)

def test(configdict):
    must = ['specfile','imagefolder', 'outputfolder', 'resolution','centralpixel','app','ymask','xmask']
    for n in must:
        if n not in configdict.keys(): raise ConfigError(n)

if __name__ == "__main__":
    cfg = cfg('config')

    
