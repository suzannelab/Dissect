import numpy as np
import pandas as pd
from skimage import morphology, filters
from skimage import segmentation as ski_seg
from scipy import ndimage as ndi
from skimage.morphology import binary_dilation
from Dissects.image import thinning
from Dissects.image import dilation
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sci
import itertools

from sklearn import manifold
from skimage import morphology
#from .seg_2D import generate_mesh
from scipy import ndimage
import math
from scipy.ndimage.morphology import binary_dilation




def segmentation(mask, size_auto_remove=True, min_area=None, max_area=None, boundary_auto_remove=True):
    """
    Segment the cells of the image.

    Paramaters
    ----------
    mask: np.array, filament=1 and background=0
    size_auto_remove : bool, default: True, keep 95% of the cell area distribution. 
	The 2.5% smallest cells are set to 0 (skeletonization error), the 2.5% largest cells are set to 1 (background).
    min_area: integer, needs to be specified only if auto_remove is False. 
	Minimum number of pixels of a cell. When smaller, cells are set to 0 (counts as skeletonization error)
    max_area: interger, needs to be specified only if auto_remove is False. 
	Maximum number of pixels of a cell. When larger, cells are set to 1 (counts as background)
    Return
    ------
    segmented_im: np.array
        Pixels of filaments are equal to 0
        Pixels of the background = 1
        Pixels of cell i = i
    """
    edges = filters.sobel(mask)
    markers = np.zeros_like(mask)
    markers[mask == 0] = 2
    markers[mask > 0] = 1

    segmented_im = ski_seg.watershed(edges, markers)
    segmented_im, _ = ndi.label(segmented_im == 2)
 
    if size_auto_remove:
        l_areas = np.unique(segmented_im, return_counts=True)[1][2:]
        min_area = np.percentile(l_areas, 2.5)
        max_area = np.percentile(l_areas, 97.5)

        segmented_im = morphology.remove_small_objects(segmented_im, min_area)
        seg_max = morphology.remove_small_objects(segmented_im, max_area)
        segmented_im[np.where(seg_max!=0)] = 1
    else:
        try:
            segmented_im = morphology.remove_small_objects(segmented_im, min_area)
            seg_max = morphology.remove_small_objects(segmented_im, max_area)
            segmented_im[np.where(seg_max!=0)] = 1

        except TypeError:
            print("If size_auto_remove is False, you must specify min_area and max_area")

    if boundary_auto_remove:
        boundcells=[]
        for i in range(len(np.unique(segmented_im))):
            if (np.isin(0, np.where(segmented_im==i))
                or np.isin(segmented_im.shape[0], np.where(segmented_im==i)[0])
                or np.isin(segmented_im.shape[1]-1, np.where(segmented_im==i)[1])):
                    boundcells.append(i)
        if np.isin(0, boundcells):
            boundcells.remove(0)
        if np.isin(1, boundcells):
            boundcells.remove(1)
        for i in boundcells:
            segmented_im[np.where(segmented_im==i)] = 1
    return segmented_im

def junction_around_cell(mask, seg, cell):
    """Find junctions around cell i.

    Parameters
    ----------
    mask: np.array, filament=1 and background=0
    seg: np.array
        output of the segmentation function
    cell: integer
        number of the chosen cell

    Returns
    -------
    juncelli: np.array
        background = 0, one-pixel-width junction around cell i = 1

    """
    segmentationi = np.zeros_like(seg)
    segmentationi[np.where(seg == cell)] = 1

    juncelli = (ndi.binary_dilation(segmentationi).astype(
        segmentationi.dtype) * mask)

    return juncelli


def skel_vertices(skel):
    """Return 2 dataframes of the skeleton bifurcation points i.e. the vertices 

    Parameters
    ----------
    skel: skeleton object, has to be breakdowned ('skelconv -breakdown')

    Returns
    -------
    vert_df: dataframe of the float vertices
    vert_df_int: dataframe of the interger vertices 
    """
    df_skel_vertices = skel.critical_point[skel.critical_point.nfil>=3]
    
    vert_df=df_skel_vertices[['x','y']]
    vert_df=vert_df.reset_index(drop=True)
    df_test=pd.DataFrame(columns=['x_0','y_0'])
    df_test['x_0']=vert_df['y']
    df_test['y_0']=vert_df['x']
    vert_df=df_test
    vert_df_int=vert_df.astype('int32')
    
    return vert_df,vert_df_int


def find_vertex(mask,
                 dist,
                free_edges=False,
                kernel_path='../Dissects/segmentation/2d_pattern.csv'):
    """
    free_edges : if True, find vertex extremity
    warning :  make sure to have a skeletonize the output of disperse
    """
    seg= segmentation(mask, size_auto_remove=False, min_area=2, max_area=10000, boundary_auto_remove=True)
    
    # Need to be improve
    kernel = np.array(pd.read_csv(kernel_path, header=None))
    kernel = kernel.reshape((int(kernel.shape[0]/3), 3, 3))

    output_image = np.zeros(mask.shape)

    for i in np.arange(len(kernel)):
        out = sci.ndimage.binary_hit_or_miss(mask, kernel[i])
        output_image = output_image + out

    if free_edges == True:
        kernel = kernels_extremity()
        for i in np.arange(len(kernel)):
            out = sci.ndimage.binary_hit_or_miss(mask, kernel[i])
            output_image = output_image + out        
    list_vertices = np.where(output_image > 0)
       
    #create dataframe with cells info
    columns_name = ['x_0','y_0','Cell_1','Cell_2','Cell_3','Cell_4','Cell_5']
    nb_vertices = len(list_vertices[0])
    init = np.zeros((nb_vertices , len(columns_name)))
    df_vertices = pd.DataFrame(data=init, columns=columns_name)
    liste0 = []
    
    #get index of cell associated to each vertex
    for v in range(0, len(list_vertices[0])):
        carre = seg[max(0,list_vertices[0][v]-dist) : min(list_vertices[0][v]+(dist+1),seg.shape[0]-1),
                    max(0,list_vertices[1][v]-dist) : min(list_vertices[1][v]+(dist+1),seg.shape[1]-1)]
        cells = np.unique(carre)
        
        if len(cells)>3:
            df_vertices.loc[v]['x_0'] = list_vertices[0][v]
            df_vertices.loc[v]['y_0'] = list_vertices[1][v]
            df_vertices.loc[v]['Cell_1'] = cells[1]
            df_vertices.loc[v]['Cell_2'] = cells[2]
            df_vertices.loc[v]['Cell_3'] = cells[3]

            if len(np.unique(carre)) == 4 :
                df_vertices.loc[v]['Cell_4'] = 'Nan'
                df_vertices.loc[v]['Cell_5'] = 'Nan'
            if len(np.unique(carre)) == 5 :
                df_vertices.loc[v]['Cell_4'] = cells[4]
                df_vertices.loc[v]['Cell_5'] = 'Nan'
            if len(np.unique(carre)) == 6 :
                df_vertices.loc[v]['Cell_4'] = cells[4]
                df_vertices.loc[v]['Cell_5'] = cells[5]
        else:
            liste0.append(v)

    df_vertices=df_vertices.drop(liste0)
        
    ind = 0
    while ind < len(df_vertices) :
    #print(df_vertices.shape[0])
        cells_ind = np.array([df_vertices['Cell_1'].iloc[ind],
                              df_vertices['Cell_2'].iloc[ind],
                              df_vertices['Cell_3'].iloc[ind],
                              df_vertices['Cell_4'].iloc[ind],
                              df_vertices['Cell_5'].iloc[ind]])

        liste_x0 = [df_vertices['x_0'].iloc[ind]]
        liste_y0 = [df_vertices['y_0'].iloc[ind]]
        liste_i = []
        liste_cellsi = [cells_ind]


        for i in range(ind+1, len(df_vertices)):

            cells_i = np.array([df_vertices['Cell_1'].iloc[i],
                                df_vertices['Cell_2'].iloc[i],
                                df_vertices['Cell_3'].iloc[i],
                                df_vertices['Cell_4'].iloc[i],
                                df_vertices['Cell_5'].iloc[i]])


            mask_TrueFalse = np.isin(cells_ind, cells_i)



    
            if np.sum(mask_TrueFalse) > 3 and np.isin(1., cells_i) and np.isin(1., cells_ind):
                liste_cellsi.append(cells_i)
                liste_i.append(df_vertices.axes[0][i])
                liste_x0.append(df_vertices['x_0'].iloc[i])
                liste_y0.append(df_vertices['y_0'].iloc[i])

            if np.sum(mask_TrueFalse) >= 3 and np.isin(1., cells_i)==False and np.isin(1., cells_ind)==False:
                liste_cellsi.append(cells_i)
                liste_i.append(df_vertices.axes[0][i])
                liste_x0.append(df_vertices['x_0'].iloc[i])
                liste_y0.append(df_vertices['y_0'].iloc[i])

        df_vertices['x_0'].iloc[ind]=int(np.round(np.mean(liste_x0)))
        df_vertices['y_0'].iloc[ind]=int(np.round(np.mean(liste_y0)))

        ucells = np.unique(liste_cellsi)

        df_vertices['Cell_1'].iloc[ind] = ucells[0]
        
        if len(ucells)>1:
            df_vertices['Cell_2'].iloc[ind] = ucells[1]
        if len(ucells)>2:
            df_vertices['Cell_3'].iloc[ind] = ucells[2]
        if len(ucells)>3:
            df_vertices['Cell_4'].iloc[ind] = ucells[3]
        if len(ucells)>4:
            df_vertices['Cell_5'].iloc[ind] = ucells[4]

#print(liste_i)
        df_vertices=df_vertices.drop(liste_i)

        ind+=1
        
    nb_vertices = len(list_vertices[0])
    init = df_vertices.to_numpy()
    df_vertices2 = pd.DataFrame(data=init, columns=columns_name)

    vert_df= df_vertices2.drop( ['Cell_1','Cell_2','Cell_3','Cell_4', 'Cell_5'], axis=1)
    return vert_df, df_vertices2




def find_cells(mask, skel, dist):

    seg= segmentation(mask, size_auto_remove=False, min_area=2, max_area=10000, boundary_auto_remove=True)
    vert_df, vert_df_int = skel_vertices(skel)
    list_vertices1 = vert_df_int.values.tolist()
    list_vertices1=np.array(list_vertices1).T
    list_vertices1=tuple(list_vertices1)
    thistuple = (list_vertices1[0], list_vertices1[1])
    
    #create dataframe with cells info
    columns_name = ['x_0','y_0','Cell_1','Cell_2','Cell_3','Cell_4','Cell_5']
    nb_vertices = len(list_vertices1[0])
    init = np.zeros((nb_vertices , len(columns_name)))
    df_vertices = pd.DataFrame(data=init, columns=columns_name)
    liste0 = []
    
    #get index of cell associated to each vertex
    for v in range(0, len(thistuple[0])):
        carre = seg[max(0, thistuple[0][v]-dist) : min(thistuple[0][v]+(dist+1), seg.shape[0]-1),
            max(0, thistuple[1][v]-dist) : min(thistuple[1][v]+(dist+1), seg.shape[1]-1)]
        cells = np.unique(carre)
        
        if len(cells)>3:
            df_vertices.loc[v]['x_0'] = thistuple[0][v]
            df_vertices.loc[v]['y_0'] = thistuple[1][v]
            df_vertices.loc[v]['Cell_1'] = cells[1]
            df_vertices.loc[v]['Cell_2'] = cells[2]
            df_vertices.loc[v]['Cell_3'] = cells[3]

            if len(np.unique(carre)) == 4 :
                df_vertices.loc[v]['Cell_4'] = 'Nan'
                df_vertices.loc[v]['Cell_5'] = 'Nan'
            if len(np.unique(carre)) == 5 :
                df_vertices.loc[v]['Cell_4'] = cells[4]
                df_vertices.loc[v]['Cell_5'] = 'Nan'
            if len(np.unique(carre)) == 6 :
                df_vertices.loc[v]['Cell_4'] = cells[4]
                df_vertices.loc[v]['Cell_5'] = cells[5]
        else:
            liste0.append(v)

    df_vertices=df_vertices.drop(liste0)
        
    ind = 0
    while ind < len(df_vertices) :
    #print(df_vertices.shape[0])
        cells_ind = np.array([df_vertices['Cell_1'].iloc[ind],
                              df_vertices['Cell_2'].iloc[ind],
                              df_vertices['Cell_3'].iloc[ind],
                              df_vertices['Cell_4'].iloc[ind],
                              df_vertices['Cell_5'].iloc[ind]])

        liste_x0 = [df_vertices['x_0'].iloc[ind]]
        liste_y0 = [df_vertices['y_0'].iloc[ind]]
        liste_i = []
        liste_cellsi = [cells_ind]


        for i in range(ind+1, len(df_vertices)):

            cells_i = np.array([df_vertices['Cell_1'].iloc[i],
                                df_vertices['Cell_2'].iloc[i],
                                df_vertices['Cell_3'].iloc[i],
                                df_vertices['Cell_4'].iloc[i],
                                df_vertices['Cell_5'].iloc[i]])


            mask_TrueFalse = np.isin(cells_ind, cells_i)



    
            if np.sum(mask_TrueFalse) > 3 and np.isin(1., cells_i) and np.isin(1., cells_ind):
                liste_cellsi.append(cells_i)
                liste_i.append(df_vertices.axes[0][i])
                liste_x0.append(df_vertices['x_0'].iloc[i])
                liste_y0.append(df_vertices['y_0'].iloc[i])

            if np.sum(mask_TrueFalse) >= 3 and np.isin(1., cells_i)==False and np.isin(1., cells_ind)==False:
                liste_cellsi.append(cells_i)
                liste_i.append(df_vertices.axes[0][i])
                liste_x0.append(df_vertices['x_0'].iloc[i])
                liste_y0.append(df_vertices['y_0'].iloc[i])

        df_vertices['x_0'].iloc[ind]=int(np.round(np.mean(liste_x0)))
        df_vertices['y_0'].iloc[ind]=int(np.round(np.mean(liste_y0)))

        ucells = np.unique(liste_cellsi)

        df_vertices['Cell_1'].iloc[ind] = ucells[0]
        
        if len(ucells)>1:
            df_vertices['Cell_2'].iloc[ind] = ucells[1]
        if len(ucells)>2:
            df_vertices['Cell_3'].iloc[ind] = ucells[2]
        if len(ucells)>3:
            df_vertices['Cell_4'].iloc[ind] = ucells[3]
        if len(ucells)>4:
            df_vertices['Cell_5'].iloc[ind] = ucells[4]

#print(liste_i)
        df_vertices=df_vertices.drop(liste_i)

        ind+=1
            
        df_vertices=df_vertices.reset_index(drop=True)

    return df_vertices

def junctions(mask, df_vertices, max_area):

    df_junctions = pd.DataFrame(columns=['Cell1','Cell2','srce', 'trgt',
                                       'angle', 'length'])

    for ind in range (0, df_vertices.shape[0]): #pour chaque vertex
        cells_ind = np.array([df_vertices['Cell_1'][ind],
                              df_vertices['Cell_2'][ind],
                              df_vertices['Cell_3'][ind],
                              df_vertices['Cell_4'][ind],
                              df_vertices['Cell_5'][ind]])

        for i in range (ind+1, df_vertices.shape[0]): # pour chaque autre vertex


            cells_i = np.array([df_vertices['Cell_1'][i],
                                df_vertices['Cell_2'][i],
                                df_vertices['Cell_3'][i],
                                df_vertices['Cell_4'][i],
                                df_vertices['Cell_5'][i]])

            mask_TrueFalse = np.isin(cells_ind, cells_i)

            if np.isin(1., cells_ind) and np.isin(1., cells_i):  #si le fond(=cellule1) fait parti des cellules communes

                if np.sum(mask_TrueFalse) >= 3 :

                    dict_junctions={'Cell1': cells_ind[np.where(mask_TrueFalse)][0],
                                    'Cell2': cells_ind[np.where(mask_TrueFalse)][1],
                                    'sx' : df_vertices['x_0'][ind],
                                    'sy' : df_vertices['y_0'][ind],
                                    'tx' : df_vertices['x_0'][i],
                                    'ty' : df_vertices['y_0'][i],
                                    'angle' : (np.arctan((df_vertices['y_0'][ind]-df_vertices['y_0'][i])
                                        /(df_vertices['x_0'][ind]-df_vertices['x_0'][i])))*180/np.pi,
                                    'length' : ((df_vertices['y_0'][ind]-df_vertices['y_0'][i])
                                        /(df_vertices['x_0'][ind]-df_vertices['x_0'][i])),
                                    'srce': ind,
                                    'trgt' :i
                   }

                    df_junctions=df_junctions.append(dict_junctions, ignore_index = True)

            else:

                if np.sum(mask_TrueFalse) >= 2 :

                    dict_junctions={'Cell1': cells_ind[np.where(mask_TrueFalse)][0],
                    'Cell2': cells_ind[np.where(mask_TrueFalse)][1],
                    'sx' : df_vertices['x_0'][ind],
                    'sy' : df_vertices['y_0'][ind],
                    'tx' : df_vertices['x_0'][i],
                    'ty' : df_vertices['y_0'][i],
                    'angle' : (np.arctan((df_vertices['y_0'][ind]-df_vertices['y_0'][i])
                                        /(df_vertices['x_0'][ind]-df_vertices['x_0'][i])))*180/np.pi,
                    'length' : np.sqrt((df_vertices['y_0'][ind]-df_vertices['y_0'][i])**2
                                        +(df_vertices['x_0'][ind]-df_vertices['x_0'][i])**2),
                    'srce': ind,
                    'trgt': i
                   }

                    df_junctions=df_junctions.append(dict_junctions, ignore_index = True)
                    
    columns=['face', 'srce', 'trgt']
    init = np.zeros(((df_junctions.shape[0])*2, 3))
    edge_df = pd.DataFrame(data=init, columns=columns)
    ind=0
   
    for i in range (0, df_junctions.shape[0]):
    
        edge_df['face'][ind]=df_junctions['Cell1'][i]
        edge_df['srce'][ind]=df_junctions['srce'][i]
        edge_df['trgt'][ind]=df_junctions['trgt'][i]

        edge_df['face'][ind+1]=df_junctions['Cell2'][i]
        edge_df['srce'][ind+1]=df_junctions['trgt'][i]
        edge_df['trgt'][ind+1]=df_junctions['srce'][i]
    
        ind=ind+2
    


    seg_nobound = segmentation(mask, size_auto_remove=False, min_area=0, max_area=max_area, boundary_auto_remove=True)
    for i in range(edge_df.shape[0]):
        segmentationi = np.zeros_like(mask)
        segmentationi[np.where(seg_nobound == edge_df.face[i])] = 1
        com=ndimage.measurements.center_of_mass(segmentationi)

        #calculate angle
        p1=np.array([df_vertices.x_0[edge_df.srce[i]], df_vertices.y_0[edge_df.srce[i]]])
        p2=np.array([df_vertices.x_0[edge_df.trgt[i]], df_vertices.y_0[edge_df.trgt[i]]])
        angle=math.atan2(p2[1]-com[1], p2[0]-com[0])-math.atan2(p1[1]-com[1], p1[0]-com[0])
        
        if np.sin(angle)<0:
            d=edge_df.srce[i]
            e=edge_df.trgt[i]
            edge_df.srce[i]=e
            edge_df.trgt[i]=d  
                    
                
    return df_junctions, edge_df 

def generate_mesh(mask, seg=None, dilation_width=1):
    """ Generate mesh

    Parameters
    ----------
    mask: np.array, filament=1 and background=0

    Returns
    -------
    face_df
    edge_df
    vert_df

    """

    # Vertices detection
    image_vertex, list_vertices, df_vertices = vertices(mask, seg, dilation_width)

    # Edge detection
    df_junctions = junctions(list_vertices, df_vertices)

    vert_df = pd.DataFrame(columns=list('xy'))
    edge_df = pd.DataFrame(columns=['cell1',
                                    'cell2',
                                    'v1',
                                    'v2'])

    # change data output
    for _, e in df_junctions.iterrows():

        edge = {'cell1': e.Cell1,
                'cell2': e.Cell2}

        df_ = vert_df[vert_df['x'] == e.x0]
        if df_.empty:
            vert_df = vert_df.append({'x': e.x0,
                                      'y': e.y0},
                                     ignore_index=True)
            edge['v1'] = vert_df.index[-1]
        else:
            df_ = df_[df_['y'] == e.y0]
            if df_.empty:
                vert_df = vert_df.append({'x': e.x0,
                                          'y': e.y0},
                                         ignore_index=True)
                edge['v1'] = vert_df.index[-1]
            else:
                edge['v1'] = df_.index[0]

        df_ = vert_df[vert_df['x'] == e.x1]
        if df_.empty:
            vert_df = vert_df.append({'x': e.x1,
                                      'y': e.y1},
                                     ignore_index=True)
            edge['v2'] = vert_df.index[-1]
        else:
            df_ = df_[df_['y'] == e.y1]
            if df_.empty:
                vert_df = vert_df.append({'x': e.x1,
                                          'y': e.y1},
                                         ignore_index=True)
                edge['v2'] = vert_df.index[-1]
            else:
                edge['v2'] = df_.index[0]

        edge_df = edge_df.append(edge,
                                 ignore_index=True)

    edge_df['cell1'] = pd.to_numeric(edge_df['cell1'], downcast='integer')
    edge_df['cell2'] = pd.to_numeric(edge_df['cell2'], downcast='integer')
    edge_df['v1'] = pd.to_numeric(edge_df['v1'], downcast='integer')
    edge_df['v2'] = pd.to_numeric(edge_df['v2'], downcast='integer')

    vert_df['x'] = pd.to_numeric(vert_df['x'], downcast='integer')
    vert_df['y'] = pd.to_numeric(vert_df['y'], downcast='integer')

    face_id = np.sort(np.unique(np.concatenate((edge_df.cell1.sort_values().to_numpy(),
                                                edge_df.cell2.sort_values().to_list()))))
    face_df = pd.DataFrame(index=face_id, columns=list('xy'))

    for f in face_id:

        v_id = np.concatenate((edge_df[(edge_df['cell1'] == f) | (edge_df['cell2'] == f)]['v1'].to_numpy(),
                               edge_df[(edge_df['cell1'] == f) | (edge_df['cell2'] == f)]['v2'].to_numpy()))

        v_id = np.unique(v_id)

        face_df.loc[f]['x'] = vert_df.loc[v_id].x.mean()
        face_df.loc[f]['y'] = vert_df.loc[v_id].y.mean()

    return face_df, edge_df, vert_df
