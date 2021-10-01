import random

import pandas as pd
import numpy as np

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import plotly.graph_objects as go


def plot_skeleton_3D(skeleton, fig=None, **kwargs):
    """
    Plot dynamic skeleton using plotly.

    Parameters
    ----------
    skeleton :
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

    fig.update_layout(title='Skeleton',
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


def plot_face_3D(skeleton, face_df, edge_df, vert_df, fig=None, **kwargs):

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

    rand = (np.random.rand(face_df.shape[0], 3)*255).astype('int')
    rand[0] = 0
    cmap_rand = ListedColormap(rand)

    fill_colors = list(mpl.colors.cnames.values())
    random.shuffle(fill_colors)
    for f in (np.unique(edge_df.face)[1:]):

        try:
            edges = edge_df[edge_df.face == f]
            nb_vert = len(edges)
            vert_order = [edges.iloc[0].srce]
            for i in range(nb_vert):
                vert_order.append(
                    edges[edges.trgt == vert_order[-1]]['srce'].to_numpy()[0])
            zs = vert_df.loc[vert_order].z_pix.to_numpy()
            ys = vert_df.loc[vert_order].y_pix.to_numpy()
            xs = vert_df.loc[vert_order].x_pix.to_numpy()
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

    fig.update_layout(title='face',
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


def plot_junction_3D(skeleton, face_df, edge_df, vert_df, fig=None, **kwargs):

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

    zs, ys, xs = vert_df.loc[edge_df['srce']][['z_pix', 'y_pix', 'x_pix']].values.flatten(
        order='F').reshape(3, edge_df.shape[0])
    zt, yt, xt = vert_df.loc[edge_df['trgt']][['z_pix', 'y_pix', 'x_pix']].values.flatten(
        order='F').reshape(3, edge_df.shape[0])

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

    fig.update_layout(title='edge',
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
