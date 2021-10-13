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


def junctions_2(skel, clean=True):
    
    if not clean:
        #skel.critical_point['id_original'] = skel.critical_point.index
        skel.filament['id_original'] = skel.filament.index

    ## now try and do it for each node
    idx_nflsup3 = np.where(skel.critical_point.nfil >= 3)[0]

    lf = []
    end_fil = []
    lcp = []
    il = 0
    for idx_depart in idx_nflsup3:
        
        i_original_depart = skel.critical_point.iloc[idx_depart]['id_original']
   
        mask_clean = np.isin(skel.cp_fil_info.iloc[i_original_depart]['destcritid'],
                                 skel.critical_point.id_original)
        cp_filId = list(np.array(skel.cp_fil_info.iloc[i_original_depart]['fillId'])[mask_clean])
        #filament indices
        idx_fils = [np.where(skel.filament['id_original'] ==
                        cp_filId[i])[0][0]
                          for i in range(len(cp_filId))]      

        #destination cp indices
        cp_destId = list(np.array(skel.cp_fil_info.iloc[i_original_depart]['destcritid'])[mask_clean])

        #destination cp indices
        idx_destcrits = [np.where(skel.critical_point['id_original'] ==
                 cp_destId[i])[0][0]
                     for i in range(len(cp_destId))]
   
   

        for inum, fi, ci in zip(range(len(idx_fils)),idx_fils,idx_destcrits):
            if not np.isin(fi,end_fil):
                list_fils = [fi]
                list_cp = [idx_depart,ci]
                idx_thiscp = ci
                ior_previous = i_original_depart  
                dest_cps = skel.cp_fil_info.iloc[skel.critical_point.iloc[idx_thiscp]['id_original']]
                mask_clean = np.isin(dest_cps['destcritid'],skel.critical_point.id_original)
                dest_cps['destcritid']  = list(np.array(dest_cps['destcritid'])[mask_clean])
                dest_cps['fillId'] = list(np.array(dest_cps['fillId'])[mask_clean])
                if np.sum(np.isin(np.array(dest_cps['destcritid']),ior_previous)) > 1:
                    print('PROBLEM')
                    break
                else:    
                    next_cp = np.array(dest_cps['destcritid'])[~np.isin(np.array(dest_cps['destcritid']),ior_previous)][0]
                    idx_nextcp = np.where(np.isin(skel.critical_point['id_original'],
                                          next_cp))[0][0]

                iwhile=0
                while skel.critical_point.iloc[idx_thiscp]['nfil'] < 3:
                    ior_previous = skel.critical_point.iloc[idx_thiscp]['id_original']
                    idx_thiscp = idx_nextcp
                    ior_current = skel.critical_point.iloc[idx_nextcp]['id_original']

                    mask_clean = np.isin(skel.cp_fil_info.iloc[ior_previous]['destcritid'],
                                     skel.critical_point.id_original)
                    previous_cp_filId = list(
                        np.array(skel.cp_fil_info.iloc[ior_previous]['fillId'])[mask_clean])
           
                    idx_fili = np.array([np.where(skel.filament['id_original'] ==
                                skel.cp_fil_info.iloc[ior_previous]['fillId'][i])[0][0]
                          for i in range(len(skel.cp_fil_info.iloc[ior_previous]['fillId']))])
                    next_fil  = idx_fili[~np.isin(idx_fili,np.array(list_fils))][0]
                    list_fils.append(next_fil)
                    list_cp.append(idx_nextcp)
         
                    dest_cps = skel.cp_fil_info.iloc[skel.critical_point.iloc[idx_nextcp]['id_original']]
                    mask_clean = np.isin(dest_cps['destcritid'],skel.critical_point.id_original)
                    dest_cps['destcritid']  = list(np.array(dest_cps['destcritid'])[mask_clean])
                    dest_cps['fillId'] = list(np.array(dest_cps['fillId'])[mask_clean])
                    if np.sum(np.isin(np.array(dest_cps['destcritid']),ior_previous)) > 1:
                        print('PROBLEM')
                        break
                    else:    
                        next_cp = np.array(dest_cps['destcritid'])[
                            ~np.isin(np.array(dest_cps['destcritid']),ior_previous)][0]
                        idx_nextcp = np.where(np.isin(skel.critical_point['id_original'],
                                          next_cp))[0][0]

                    iwhile+=1
                lf.append(list_fils)
                end_fil.append(list_fils[-1])
                lcp.append(list_cp)
               
    vert_df = skel.critical_point.iloc[idx_nflsup3][['x','y']]
    columns_name = ['srce','trgt']
    nb_junctions = len(lf)
    init = np.zeros((nb_junctions , len(columns_name)))

    edge_df  = pd.DataFrame(data=init, columns=columns_name, dtype=np.int64)

    for i in range(len(lf)):
        edge_df.iloc[i]['srce'] = np.where(idx_nflsup3 == lcp[i][0])[0][0]
        edge_df.iloc[i]['trgt'] = np.where(idx_nflsup3 == lcp[i][-1])[0][0]
       
    Junctions  = np.empty(len(lf),dtype='object')
    junc_points = np.empty(len(lf),dtype='object')
    junc_cp_ends = np.empty(len(lf),dtype='object')
    lenght = np.zeros(len(lf))

    for ijunc in range(len(lf)):
        Junctions[ijunc] = []
        ifil = 0
        for fil in lf[ijunc]:
            if ifil == 0:
                ppoint = np.array(skel.point[skel.point['filament'] == fil][['x','y']])
                Junctions[ijunc].append(ppoint)
            if ifil == 1:
                ppoint_before = ppoint
                ppoint = np.array(skel.point[skel.point['filament'] == fil][['x','y']])
                ppoint1 = ppoint
                if np.all(ppoint[-1] == ppoint_before[0]):
                    ##flip 0+1
                    Junctions[ijunc][0] = np.flip(Junctions[ijunc][0],axis=0)
                    ppoint1 = np.flip(ppoint,axis=0)
                    Junctions[ijunc].append(ppoint1)
                elif np.all(ppoint[0] == ppoint_before[0]):
                    ##flip 0
                    Junctions[ijunc][0] = np.flip(Junctions[ijunc][0],axis=0)
                    Junctions[ijunc].append(ppoint)
                elif np.all(ppoint[-1] == ppoint_before[-1]):
                    ##flip 1
                    ppoint1 = np.flip(ppoint,axis=0)
                    Junctions[ijunc].append(ppoint1)
                else:
                    Junctions[ijunc].append(ppoint)
                ppoint = ppoint1
            if ifil > 1:
                ppoint_before = ppoint
                ppoint = np.array(skel.point[skel.point['filament'] == fil][['x','y']])
                ppoint1 = ppoint
                if np.all(ppoint[-1] == ppoint_before[-1]):
                    ##flip 1
                    ppoint1 = np.flip(ppoint,axis=0)
                    Junctions[ijunc].append(ppoint1)
                else:
                    Junctions[ijunc].append(ppoint)
                ppoint = ppoint1
            ifil += 1  
            junc_points[ijunc] = np.concatenate(Junctions[ijunc])
            junc_cp_ends[ijunc] = [np.where(idx_nflsup3 == lcp[ijunc][0])[0][0],
                                   np.where(idx_nflsup3 == lcp[ijunc][-1])[0][0]]
            lenght[ijunc] = np.sum(np.sqrt(np.sum(
                ((np.roll(junc_points[ijunc], 1, axis=0) -
                  junc_points[ijunc])[1:])**2, axis=1)))
       
    df_junc = pd.DataFrame(data={'vertices': junc_cp_ends,
                             'points_coords': junc_points,
                             'points_coords_binaire': [junc_points[ijunc].astype(int)
                                                for ijunc in range(len(junc_points))],
                             'lenght': lenght})

       
    return vert_df, edge_df, df_junc


def create_binary_image_junc(df_junc, ijunc, image):
    binary_image = np.zeros_like(image)
    for coord in df_junc.iloc[ijunc].points_coords_binaire:
        binary_image[coord[1],coord[0]] = 1
    return binary_image



def get_cells(seg, df_junc, image):
    c1=[]
    c2=[]

    for i in range (0,  df_junc.shape[0]):
    
        l=[]
        m=[]
        binary_junc_i = create_binary_image_junc(df_junc, i, image)
        dil=ndi.binary_dilation(binary_junc_i)*seg

        for ind in np.unique(dil)[1:]:
            t=(np.sum(np.isin(dil, ind)), ind)
            l.append(t)
        
        u=np.array(l)
        v=u[np.argsort(u[:,0])][-2:,1]
        c1.append(v[0])
        c2.append(v[1])
    df_junc['Cell 1'] = c1
    df_junc['Cell 2'] = c2

    return df_junc



def create_edge_df(junc_df, seg):

    columns=['face', 'srce', 'trgt']
    init = np.zeros(((df_junc.shape[0])*2, 3))
    edge_df = pd.DataFrame(data=init, columns=columns)
    ind=0

    for i in range (0, df_junc.shape[0]):
    
        edge_df['face'][ind]=df_junc['Cell 1'][i]
        edge_df['srce'][ind]=df_junc['vertices'][i][0]
        edge_df['trgt'][ind]=df_junc['vertices'][i][1]

        edge_df['face'][ind+1]=df_junc['Cell 2'][i]
        edge_df['srce'][ind+1]=df_junc['vertices'][i][1]
        edge_df['trgt'][ind+1]=df_junc['vertices'][i][0]
    
        ind=ind+2
    

    for i in np.arange(0, edge_df.shape[0], 2):
        print(i)
        segmentationi = np.zeros_like(mask)
        segmentationi[np.where(seg_nobound == edge_df.face[i])] = 1
        com=ndimage.measurements.center_of_mass(segmentationi) #com = center of mass

        #calculate angle
        p1=np.array([vert_df.x.iloc[int(edge_df.srce.iloc[i])], vert_df.y.iloc[int(edge_df.srce.iloc[i])]])
        p2=np.array([vert_df.x.iloc[int(edge_df.trgt.iloc[i])], vert_df.y.iloc[int(edge_df.trgt.iloc[i])]])
        angle=math.atan2(p2[1]-com[1], p2[0]-com[0])-math.atan2(p1[1]-com[1], p1[0]-com[0])


        if np.sin(angle)<0:

            e = edge_df.srce.iloc[i]
            edge_df.srce.iloc[i] = edge_df.srce.iloc[i+1]
            edge_df.srce.iloc[i+1] = e
        
            f = edge_df.trgt.iloc[i]
            edge_df.trgt.iloc[i] = edge_df.trgt.iloc[i+1]
            edge_df.trgt.iloc[i+1] = f
    return edge_df
