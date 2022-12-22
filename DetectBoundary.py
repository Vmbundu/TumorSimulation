import numpy as np
def DetectBoundary(nod3xyz,leng):
    geom = {
        "a": np.array(np.where(nod3xyz[0:,0] == 0)),
        "b": np.array(np.where(nod3xyz[0:,0] == leng)),
        "c": np.array(np.where(nod3xyz[0:,1] == 0)),
        "d": np.array(np.where(nod3xyz[0:,1] == leng)),
        "e": np.array(np.where(nod3xyz[0:,2] == 0)),
        "f": np.array(np.where(nod3xyz[0:,2] == leng))
            }

    geom['c'] = np.setdiff1d(geom['c'], geom['a'])
    geom['c'] = np.setdiff1d(geom['c'], geom['b'])
    #geom['c'] = geom['c'] - geom['b']
    
    geom['d'] = np.setdiff1d(geom['d'], geom['a'])
    geom['d'] = np.setdiff1d(geom['d'], geom['b'])
    #geom.d = geom.d - geom.b
    
    geom['e'] = np.setdiff1d(geom['e'], geom['a'])
    geom['e'] = np.setdiff1d(geom['e'], geom['b'])
    geom['e'] = np.setdiff1d(geom['e'], geom['c'])
    geom['e'] = np.setdiff1d(geom['e'], geom['d'])
    #geom.e = geom.e - geom.d
    
    geom['f'] = np.setdiff1d(geom['f'], geom['a'])
    geom['f'] = np.setdiff1d(geom['f'], geom['b'])
    geom['f'] = np.setdiff1d(geom['f'], geom['c'])
    geom['f'] = np.setdiff1d(geom['f'], geom['d'])
    #geom['f'] = geom['f'] - geom['d']

    return np.concatenate((geom['a'], geom['b'], geom['c'], geom['d'], geom['e'], geom['f']), axis=None)
