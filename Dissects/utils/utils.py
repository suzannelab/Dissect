import pandas as pd
import numpy as np


def pixel_to_um(df, pixel_size, coords, n_coords):
    for i in range(len(coords)):
    	df[n_coords[i]] = df[coords[i]]*pixel_size[n_coords[i].upper()+'_SIZE']
    

def um_to_pixel(df, pixel_size, coords, n_coords):
    for i in range(len(coords)):
    	df[n_coords[i]] = (df[coords[i]]/pixel_size[n_coords[i].upper()+'_SIZE']).astype(int)
