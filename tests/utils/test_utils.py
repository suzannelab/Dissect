import os

import numpy as np
import pandas as pd

from Dissects.utils.utils import (pixel_to_um,
                            	  um_to_pixel)


def test_pixel_to_um():
    df = pd.DataFrame(data={'x': np.random.randint(0, 20, 10),
                            'y': np.random.randint(0, 20, 10)})
    pixel_size={'x':0.5,
    			'y':0.25}
    pixel_to_um(df, pixel_size, list('xy'), ['xum', 'yum'])

    assert (df['xum'] == df['x']*pixel_size['x']).all()
    assert (df['yum'] == df['y']*pixel_size['y']).all()


def test_um_to_pixel():
    df = pd.DataFrame(data={'xum': np.random.rand(10)*10,
                            'yum': np.random.rand(10)*10})
    pixel_size={'xum':0.5,
    			'yum':0.25}
    um_to_pixel(df, pixel_size, ['xum', 'yum'], list('xy'))

    assert (df['x'] == (df['xum']/pixel_size['xum']).astype(int)).all()
    assert (df['y'] == (df['yum']/pixel_size['yum']).astype(int)).all()
