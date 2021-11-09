import random

import pandas as pd
import numpy as np

import matplotlib as mpl
import pylab as pl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import plotly.graph_objects as go



def initiate_figure(segmentation):
    fig = go.Figure()

    x_size = 1
    y_size = 1
    z_size = 1

    x_size = segmentation.specs["x_size"]
    y_size = segmentation.specs["y_size"]
    try:
        z_size = segmentation.specs["z_size"]
    except:
        pass
    fig.update_layout(title='',
                  autosize=False,
                  width=1000,
                  height=1000,
                  margin=dict(l=65, r=50, b=65, t=90),
                  showlegend=False,
                  scene=dict(aspectmode="manual",
                             aspectratio=dict(x=x_size*segmentation.specs['x_shape'],
                                              y=y_size*segmentation.specs['y_shape'],
                                              z=z_size*segmentation.specs['z_shape'])),
                  )
    return fig



def plot_skeleton_3D(skeleton, fig=None, **kwargs):
    """
    Plot dynamic skeleton using plotly.

    Parameters
    ----------
    segmentation :
    **kwargs: metadata as x_size, y_size, z_size

    Returns
    -------
    """
    if fig is None:
        fig = go.Figure()

    image_skeleton = skeleton.create_binary_image()
    x_shape = image_skeleton.shape[2]
    y_shape = image_skeleton.shape[1]
    z_shape = image_skeleton.shape[0]

    x_size = 1
    y_size = 1
    z_size = 1
    if kwargs is None:
        warnings.warn("There is pixel/voxel size defined.")

    else:
        x_size = kwargs["x_size"]
        y_size = kwargs["y_size"]
        try:
            z_size = kwargs["z_size"]
        except:
            pass

    z0, y0, x0 = np.where(image_skeleton > 0)
    fig.add_trace(go.Scatter3d(x=x0,
                               y=y0,
                               z=z0,
                               mode='markers',
                               marker=dict(
                                    size=2,
                                    color='black',
                                    opacity=1
                               )
                               )
                  )

    fig.update_layout(title='',
                      autosize=False,
                      width=1000,
                      height=1000,
                      margin=dict(l=65, r=50, b=65, t=90),
                      showlegend=False,
                      scene=dict(aspectmode="manual",
                                 aspectratio=dict(x=x_size*x_shape,
                                                  y=y_size*y_shape,
                                                  z=z_size*z_shape)),
                      )

    return fig


def plot_face_3D(segmentation, fig=None, **kwargs):

    if fig is None:
        fig = initiate_figure(segmentation)
    

    rand = (np.random.rand(segmentation.face_df.shape[0], 3)*255).astype('int')
    rand[0] = 0
    cmap_rand = ListedColormap(rand)

    fill_colors = list(mpl.colors.cnames.values())
    random.shuffle(fill_colors)
    for f in (np.unique(segmentation.edge_df.face)[1:]):

        try:
            edges = segmentation.edge_df[segmentation.edge_df.face == f]
            nb_vert = len(edges)
            vert_order = [edges.iloc[0].srce]
            for i in range(nb_vert):
                vert_order.append(
                    edges[edges.trgt == vert_order[-1]]['srce'].to_numpy()[0])
            zs = segmentation.vert_df.loc[vert_order].z_pix.to_numpy()
            ys = segmentation.vert_df.loc[vert_order].y_pix.to_numpy()
            xs = segmentation.vert_df.loc[vert_order].x_pix.to_numpy()
            fig.add_trace(dict(
                type='scatter3d',
                mode='lines',
                x=xs,
                y=ys,
                z=zs,
                name='',
                # add a surface axis ('1' refers to axes[1] i.e. the y-axis)
                surfaceaxis=2,
                surfacecolor=fill_colors[f],
                line=dict(
                    color='black',
                    width=4
                ),
            ))
        except:
            print(f)

    return fig


def plot_face_analyse_3D(segmentation, column, normalize=True, normalize_max=None, border=True, fig=None, **kwargs):

    if fig is None:
        fig = initiate_figure(segmentation)
    
    if border: 
        face_id = segmentation.face_df.index.to_numpy()
    else:
        face_id = segmentation.face_df[segmentation.face_df.border==0].index.to_numpy()

    c = segmentation.face_df.loc[face_id, column]
    cmap = pl.cm.cividis
    
    if normalize:
        if normalize_max is None:
            c = c/np.max(c)
        else:
            c = c/normalize_max
    c = (c*255/np.max(c)).astype(int)
    
    for f in face_id:
        try:
            edges = segmentation.edge_df[segmentation.edge_df.face == f]
            nb_vert = len(edges)
            vert_order = [edges.iloc[0].srce]
            for i in range(nb_vert):
                vert_order.append(
                    edges[edges.trgt == vert_order[-1]]['srce'].to_numpy()[0])
            zs = segmentation.vert_df.loc[vert_order].z_pix.to_numpy()
            ys = segmentation.vert_df.loc[vert_order].y_pix.to_numpy()
            xs = segmentation.vert_df.loc[vert_order].x_pix.to_numpy()
            fig.add_trace(dict(
                type='scatter3d',
                mode='lines',
                x=xs,
                y=ys,
                z=zs,
                name='',
                # add a surface axis ('1' refers to axes[1] i.e. the y-axis)
                surfaceaxis=2,
                surfacecolor='rgb'+str(cmap(c[f])[0:3]),
                line=dict(
                    color='black',
                    width=4
                ),
            ))
        except Exception as ex:
            print (ex)
            print(f)

    return fig


def plot_junction_3D(segmentation, fig=None, **kwargs):

    if fig is None:
        fig = initiate_figure(segmentation)
        

    zs, ys, xs = segmentation.vert_df.loc[segmentation.edge_df['srce']][['z_pix', 'y_pix', 'x_pix']].values.flatten(
        order='F').reshape(3, segmentation.edge_df.shape[0])
    zt, yt, xt = segmentation.vert_df.loc[segmentation.edge_df['trgt']][['z_pix', 'y_pix', 'x_pix']].values.flatten(
        order='F').reshape(3, segmentation.edge_df.shape[0])

    rand = (np.random.rand(len(zs), 3)*255).astype('int')
    rand[0] = 0
    cmap_rand = ListedColormap(rand)
    for i in range(len(zs)):
        fig.add_trace(
            go.Scatter3d(
                x=[xs[i], xt[i]],
                y=[ys[i], yt[i]],
                z=[zs[i], zt[i]],
                mode='lines',
                line={"color": rand[[i, i, i]],
                      "width": 10,
                      },
            )
        )

    return fig


def plot_junction_analyse_3D(segmentation, column, normalize=True, normalize_max=None, border=True, fig=None, **kwargs):

    if fig is None:
        fig = initiate_figure(segmentation)
        
    if border: 
        edge_id = segmentation.edge_df.index.to_numpy()
    else:
        edge_id = segmentation.edge_df[segmentation.edge_df.opposite!=-1].index.to_numpy()

    zs,ys,xs = segmentation.vert_df.loc[segmentation.edge_df.loc[edge_id, 'srce']][["z_pix", "y_pix", "x_pix"]].values.flatten(order='F').reshape(3, segmentation.edge_df.loc[edge_id].shape[0])
    zt,yt,xt = segmentation.vert_df.loc[segmentation.edge_df.loc[edge_id, 'trgt']][["z_pix", "y_pix", "x_pix"]].values.flatten(order='F').reshape(3, segmentation.edge_df.loc[edge_id].shape[0])
    c = segmentation.edge_df.loc[edge_id, column].to_numpy()
    cmap = pl.cm.cividis
    
    if normalize:
        if normalize_max is None:
            c = c/np.max(c)
        else:
            c = c/normalize_max
 

    for i in range(len(zs)):

        fig.add_trace(
                go.Scatter3d(
                    x=[xs[i], xt[i]],
                    y=[ys[i], yt[i]],
                    z=[zs[i], zt[i]],
                    mode='lines',
                    line={"color":[cmap(c[i]), cmap(c[i])],
                          "width":10,
                         },                
                )
            )

    return fig


def plot_aniso_cell(segmentation, fig=None):
    if fig is None:
        fig = initiate_figure(segmentation)

    cmap = pl.cm.cividis
    max_aniso = segmentation.face_df['aniso'][1:].max()
    segmentation.face_df['norm_aniso'] = (segmentation.face_df['aniso']/max_aniso)
    factor = 2
    startx = (segmentation.face_df['fx'] - factor*segmentation.face_df['norm_aniso']*segmentation.face_df['orientationx'])/segmentation.specs["x_size"]
    starty = (segmentation.face_df['fy'] - factor*segmentation.face_df['norm_aniso']*segmentation.face_df['orientationy'])/segmentation.specs["y_size"]
    startz = (segmentation.face_df['fz'] - factor*segmentation.face_df['norm_aniso']*segmentation.face_df['orientationz'])/segmentation.specs["z_size"]
    endx = (segmentation.face_df['fx'] + factor*segmentation.face_df['norm_aniso']*segmentation.face_df['orientationx'])/segmentation.specs["x_size"]
    endy = (segmentation.face_df['fy'] + factor*segmentation.face_df['norm_aniso']*segmentation.face_df['orientationy'])/segmentation.specs["y_size"]
    endz = (segmentation.face_df['fz'] + factor*segmentation.face_df['norm_aniso']*segmentation.face_df['orientationz'])/segmentation.specs["z_size"]
    
    for i in segmentation.face_df.index:
        fig.add_trace(
                go.Scatter3d(
                    x=[startx[i], endx[i]],
                    y=[starty[i], endy[i]],
                    z=[startz[i], endz[i]],
                    mode='lines',
                    line={#"color":'red',
                          "color":'rgb'+str(cmap(int(segmentation.face_df.loc[i]['norm_aniso']*cmap.N))[0:3]),
                          "width":10,
                         },                
                )
            )
        

    return fig


