
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from tifffile import tifffile
from scipy import ndimage

def network_structure(skel, pixel_size, clean=False):
    """
    Follow each junction from one vertex to the next one, defines it as an edge and calcul its length.

    Parameters
    ----------
    skel : skeleton object, breakdowned (optional: smoothed)

    Return
    ------
    df_junc : Dataframe
    """
    
    skel.critical_point['z'] = skel.critical_point['z']*pixel_size['Z_SIZE']/pixel_size['X_SIZE']
    skel.point['z'] = skel.point['z']*pixel_size['Z_SIZE']/pixel_size['X_SIZE']
    
    if not clean:
        skel.critical_point['id_original'] = skel.critical_point.index
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
            dead_end = False
            if not np.isin(fi,end_fil):
                list_fils = [fi]
                list_cp = [idx_depart,ci]
                idx_thiscp = ci
                ior_previous = i_original_depart  
                dest_cps = skel.cp_fil_info.iloc[skel.critical_point.iloc[idx_thiscp]['id_original']]
                mask_clean = np.isin(dest_cps['destcritid'],skel.critical_point.id_original)
                dest_cps['destcritid']  = list(np.array(dest_cps['destcritid'])[mask_clean])
                dest_cps['fillId'] = list(np.array(dest_cps['fillId'])[mask_clean])
                if ((np.sum(np.isin(np.array(dest_cps['destcritid']),ior_previous)) > 1) and
                    (np.all((np.isin(np.array(dest_cps['destcritid']),ior_previous))))):
                    lf.append(list_fils)
                    end_fil.append(list_fils[-1])
                    lcp.append(list_cp)
                    continue  
                #    print('PROBLEM1')
                #    print(np.array(dest_cps['destcritid']),ior_previous)
                #    break
                if (len(np.isin(np.array(dest_cps['destcritid']),ior_previous)) == 1 and 
                    np.isin(np.array(dest_cps['destcritid']),ior_previous) == [True]):
                    #print('dead end outside while')
                    lf.append(list_fils)
                    end_fil.append(list_fils[-1])
                    lcp.append(list_cp)
                    #dead_end = True
                    continue   
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
                        print('PROBLEM2')
                        print(np.array(dest_cps['destcritid']),ior_previous)
                        break
                    if (len(np.isin(np.array(dest_cps['destcritid']),ior_previous)) == 1 and 
                        np.isin(np.array(dest_cps['destcritid']),ior_previous) == [True]):
                        lf.append(list_fils)
                        end_fil.append(list_fils[-1])
                        lcp.append(list_cp)
                        #print('dead end inside while')
                        dead_end = True
                        break
                    else:    
                        next_cp = np.array(dest_cps['destcritid'])[
                            ~np.isin(np.array(dest_cps['destcritid']),ior_previous)][0]
                        idx_nextcp = np.where(np.isin(skel.critical_point['id_original'],
                                          next_cp))[0][0]

                    iwhile+=1
                if not dead_end:
                    lf.append(list_fils)
                    end_fil.append(list_fils[-1])
                    lcp.append(list_cp) 
                      
    Junctions  = np.empty(len(lf),dtype='object')
    junc_points = np.empty(len(lf),dtype='object')
    junc_cp_ends = np.empty(len(lf),dtype='object')
    junc_cp_ends_srce = np.empty(len(lf),dtype='object')
    junc_cp_ends_trgt = np.empty(len(lf),dtype='object')
    length = np.zeros(len(lf))

    for ijunc in range(len(lf)):
        Junctions[ijunc] = []
        ifil = 0
        for fil in lf[ijunc]:
            if ifil == 0:
                ppoint = np.array(skel.point[skel.point['filament'] == fil][['x','y','z']])
                Junctions[ijunc].append(ppoint)
            if ifil == 1:
                ppoint_before = ppoint
                ppoint = np.array(skel.point[skel.point['filament'] == fil][['x','y','z']])
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
                ppoint = np.array(skel.point[skel.point['filament'] == fil][['x','y','z']])
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
            
            junc_cp_ends[ijunc] = [lcp[ijunc][0],lcp[ijunc][-1]]
            junc_cp_ends_srce[ijunc] = junc_cp_ends[ijunc][0]
            junc_cp_ends_trgt[ijunc] = junc_cp_ends[ijunc][1]
            #junc_cp_ends[ijunc] = [float_vert_df.index[np.where(idx_nflsup3 == lcp[ijunc][0])[0][0]],
            #                       float_vert_df.index[np.where(idx_nflsup3 == lcp[ijunc][-1])[0][0]]]
            length[ijunc] = np.sum(np.sqrt(np.sum(
                ((np.roll(junc_points[ijunc], 1, axis=0) -
                  junc_points[ijunc])[1:])**2, axis=1)))
       
    df_junc = pd.DataFrame(data={'srce' : junc_cp_ends_srce,
                                 'trgt' :junc_cp_ends_trgt,
                             'points_coords': junc_points,
                             'points_coords_binaire': [junc_points[ijunc].astype(int)
                                                for ijunc in range(len(junc_points))],
                             'length_AU': length,
                             'length_um': length*pixel_size['X_SIZE']
                                
                                })
    

    skel.critical_point['z']=skel.critical_point['z']*pixel_size['X_SIZE']/pixel_size['Z_SIZE']
    skel.point['z']=skel.point['z']*pixel_size['X_SIZE']/pixel_size['Z_SIZE']
       
    return df_junc


def create_network(df_junc, skel):
    
    """
    Create 2 dataframes. One for the nodes and one for the links

    Parameters
    ----------
    df_junc : dataframe output of the junctions function
    skel

    Return
    ------
    node_df : DataFrame
    link_df : Dataframe
    G : Graph (class of networkX)
    """
    #creation of node_df from df_junc
    ind_list = np.unique(np.concatenate((df_junc.srce, df_junc.trgt), axis=0)).tolist()
    node_df = skel.critical_point.iloc[ind_list]
    
    #creation of link_df from df_junc
    srce=df_junc['srce'].tolist()
    trgt=df_junc['trgt'].tolist()
    columns_name = ['srce', 'trgt']
    nb_node = len(srce+trgt)
    init = np.zeros((nb_node, len(columns_name)))
    link_df = pd.DataFrame(data=init, columns=columns_name)
    
    link_df['srce']=srce+trgt
    link_df['trgt']=trgt+srce
    
    #creation of graph object
    G = nx.from_pandas_edgelist(link_df,
                            source='srce',
                            target='trgt',
                            create_using=nx.Graph())
    dic_pos={}
    dic_pos = node_df[list("xyz")].T.to_dict('list')

    return node_df, link_df, G


def centrality_array(img, dil, node_df, measure, pixel_size) :
    centrality= np.zeros(img.shape)
    x_size = pixel_size['X_SIZE']
    y_size = pixel_size['Y_SIZE']
    try:
        z_size = pixel_size['Z_SIZE']
    except:
        pass
    for k, v in  measure.items():
        coord_xk0, coord_xk1 = int(node_df.x[k])-dil+1, int(node_df.x[k]) + dil
        coord_yk0, coord_yk1 = int(node_df.y[k])-dil+1, int(node_df.y[k]) + dil
        coord_zk0, coord_zk1 = (int(node_df.z[k]-(int(dil*x_size /z_size)-1))), int(node_df.z[k]) + (int(dil*x_size /z_size))
        centrality[coord_zk0:coord_zk1, coord_yk0:coord_yk1, coord_xk0:coord_xk1] = v
    return centrality
    

def branch_analysis(df_junc, skel, image_myo, pixel_size):
    """
    Calculate straight length, tortuosity, mean and direction for each branch (from node to node) of the network

    Parameters
    ----------
    skel
    df_junc : dataframe


    Return
    ------
    df_junc : completed df_junc

    """
    ls=[]
    lt=[]
    delta_st=[]
    t=[]
    lmean=[]

    for i in range(len(df_junc)):
        
        df_junc.points_coords_binaire[i][:,2] = (df_junc.points_coords.iloc[i][:,2]*pixel_size['Z_SIZE']).astype(int)
        
        s_xyz = (skel.critical_point.x.iloc[df_junc.srce[i]],
                 skel.critical_point.y.iloc[df_junc.srce[i]],
                 skel.critical_point.z.iloc[df_junc.srce[i]])
        ls.append(s_xyz)
    
        t_xyz = (skel.critical_point.x.iloc[df_junc.trgt[i]],
             skel.critical_point.y.iloc[df_junc.trgt[i]],
             skel.critical_point.z.iloc[df_junc.trgt[i]])
        lt.append(t_xyz)

        dst = np.sqrt((s_xyz[0]-t_xyz[0])**2
                  +(s_xyz[1]-t_xyz[1])**2
                  +(s_xyz[2]-t_xyz[2])**2)
    
        delta_st.append(dst)

        t.append(df_junc['length_AU'][i]/dst)

        lm = np.mean(image_myo.T[tuple((df_junc.points_coords_binaire.loc[i].T).astype(int))])
        lmean.append(lm)



    df_junc['s_xyz'] = ls 
    df_junc['t_xyz'] = lt 
    df_junc['straight_length'] = delta_st
    df_junc['tortuosity'] = t
    df_junc['mean']= lmean

    return df_junc

def compute_orientation(df_junc, xy=True, yz=False, xz=False):
    """
    Calculate orientation in degree (between 0 and 180) of the branches 

    Parameters
    ----------
    df_junc : dataframe after branch_analysis function according to the xy, yz, or xz plan

    Return
    ------
    df_junc : completed df_junc
    """
    if xy==True :
        angle=[]
        for i in range(len(df_junc)):
            if df_junc.t_xyz[i][0] == df_junc.s_xyz[i][0]:
                beta = 90
            else:
                alpha = np.arctan((df_junc.t_xyz[i][1]-df_junc.s_xyz[i][1])/(df_junc.t_xyz[i][0]-df_junc.s_xyz[i][0]))
                beta = (alpha)*180/np.pi
                if beta<0:
                    beta = (beta+180)
            angle.append(beta)
        df_junc['angle_xy']=angle

    if yz is True :
        angle=[]
        for i in range(len(df_junc)):
            if df_junc.t_xyz[i][1] == df_junc.s_xyz[i][1]:
                beta = 90
            else:
                alpha = np.arctan((df_junc.s_xyz[i][2]-df_junc.t_xyz[i][2])/(df_junc.s_xyz[i][1]-df_junc.t_xyz[i][1]))
                beta = (alpha)*180/np.pi
                if beta<0:
                    beta = (beta+180)
            angle.append(beta)
        df_junc['angle_yz']=angle

    if xz is True :
        angle=[]
        for i in range(len(df_junc)):
            if df_junc.t_xyz[i][0] == df_junc.s_xyz[i][0]:
                beta = 90
            else:
                alpha = np.arctan((df_junc.s_xyz[i][2]-df_junc.t_xyz[i][2])/(df_junc.s_xyz[i][0]-df_junc.t_xyz[i][0]))
                beta = (alpha)*180/np.pi
                if beta<0:
                    beta = (beta+180)
            angle.append(beta)
        df_junc['angle_xz']=angle

    return df_junc

def global_network_property(skel, df_junc, node_df, image_myo):
    """
    Calculate properties of the total network and store it in a datafarme. To be applied after branch_analysis for the global tortuosity

    Parameters
    ----------
    df_junc : 

    Return
    ------
    global_network_df : DataFrame

    """
    bin_skel = skel.create_binary_image()

    columns_name = ['length', 'nodes', 'end_nodes', 'branches', 'tortuosity', 'global_mean', 'branch_mean']
    init = np.zeros((1, len(columns_name)))
    global_network_df = pd.DataFrame(data=init, columns=columns_name)
    
    global_network_df.length = df_junc['length_um'].sum(axis = 0, skipna = True)
    global_network_df.nodes = len(node_df)
    global_network_df.branches = len(df_junc)
    global_network_df.end_nodes = len(node_df[node_df.nfil==1])
    global_network_df.tortuosity = df_junc['tortuosity'].mean(axis = 0, skipna = True)
    global_network_df.global_mean = np.mean(image_myo[bin_skel.astype(bool)])
    global_network_df.branch_mean = df_junc['mean'].mean(axis = 0, skipna = True)
        
    return global_network_df



def nb_node(img, 
            node_df,
           face_df, 
            it, 
            dil,
            segmentation):
    
    """
    Create an nd_array. Each dilated apical surface is equal to the number of contained node of signal 2

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um
    pixel_size: dict

    Return
    ------
    nb_nodes : nd_array

    """
    nb_nodes= np.zeros(img.shape)
    nodes = np.zeros(img.shape)
    
    nodes[node_df.z.to_numpy().astype(int),
          node_df.y.to_numpy().astype(int),
          node_df.x.to_numpy().astype(int)] = 1
    nb=[]
    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        #dil_fill_cell_i = ndimage.binary_dilation(fill_cell_i).astype(int)
        cross = nodes*fill_cell_i
        nb_nodes[np.where(fill_cell_i ==1)]= np.count_nonzero(cross)
        nb.append(np.count_nonzero(cross))
                  
    face_df['nb_nodes']=nb
        
    return nb_nodes



def mean_connection(img, 
            node_df,
           face_df, 
            it, 
            dil,
            segmentation):
    """
    Create an nd_array where each dilated apical surface is equal to mean of the number of connection of each node contained in this volume. 
    Fills face_df

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um
    pixel_size: dict

    Return
    ------
    mean_connection : nd_array

    """

    nodes = np.zeros(img.shape)
    for i in range(len(node_df)) :
        if node_df.nfil.iloc[i] != 1:
            nodes[int(node_df.z.iloc[i]), int(node_df.y.iloc[i]), int(node_df.x.iloc[i])] = node_df.nfil.iloc[i]
            
    mean_connection= np.zeros(img.shape)
    nghbr=[]

        
    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = nodes*fill_cell_i

        mean_connection[np.where(fill_cell_i ==1)] = np.sum(cross)/np.count_nonzero(cross)
        nghbr.append(np.sum(cross)/np.count_nonzero(cross))
                  
                  
    face_df['connection_mean'] = nghbr
        
    return mean_connection

def sum_betweenness_percell(img,
                           node_df,
                           face_df,
                            G,
                           it,
                           dil,
                           segmentation):
    """
    Create an nd_array. Each dilated apical surface is equal to the sum of the chosen centrality of signal 2 network contained in this volume

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    G : Graph from networkX
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um

    Return
    ------
    sum_centrality : nd_array, beware: the centrality has been multiplied by the number of node for representation purposes

    """
    
    sum_centrality = np.zeros(img.shape)
    centrality= np.zeros(img.shape)
    c=[]
    
    for k, v in  nx.betweenness_centrality(G).items():
        coord_xk = int(node_df.x[k])
        coord_yk= int(node_df.y[k])
        coord_zk= int(node_df.z[k])
        centrality[coord_zk, coord_yk, coord_xk] = v

    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = centrality*fill_cell_i
        sum_centrality[np.where(fill_cell_i ==1)]= np.sum(cross)*len(node_df)
        
        c.append(np.sum(cross))
        
    face_df['sum_btw']=c
    return sum_centrality

def sum_closeness_percell(img,
                           node_df,
                           face_df,
                            G,
                           it,
                           dil,
                           segmentation):
    """
    Create an nd_array. Each dilated apical surface is equal to the sum of the chosen centrality of signal 2 network contained in this volume

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    G : Graph from networkX
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um

    Return
    ------
    sum_centrality : nd_array, beware: the centrality has been multiplied by the number of node for representation purposes

    """
    
    sum_centrality = np.zeros(img.shape)
    centrality= np.zeros(img.shape)
    c=[]
    
    for k, v in  nx.closeness_centrality(G).items():
        coord_xk = int(node_df.x[k])
        coord_yk= int(node_df.y[k])
        coord_zk= int(node_df.z[k])
        centrality[coord_zk, coord_yk, coord_xk] = v

    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = centrality*fill_cell_i
        sum_centrality[np.where(fill_cell_i ==1)]= np.sum(cross)*len(node_df)
        
        c.append(np.sum(cross))
        
    face_df['sum_closeness']=c
    return sum_centrality

def sum_degree_percell(img,
                           node_df,
                           face_df,
                            G,
                           it,
                           dil,
                           segmentation):
    """
    Create an nd_array. Each dilated apical surface is equal to the sum of the chosen centrality of signal 2 network contained in this volume

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    G : Graph from networkX
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um

    Return
    ------
    sum_centrality : nd_array, beware: the centrality has been multiplied by the number of node for representation purposes

    """
    
    sum_centrality = np.zeros(img.shape)
    centrality= np.zeros(img.shape)
    c=[]
    
    for k, v in  nx.degree_centrality(G).items():
        coord_xk = int(node_df.x[k])
        coord_yk= int(node_df.y[k])
        coord_zk= int(node_df.z[k])
        centrality[coord_zk, coord_yk, coord_xk] = v

    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = centrality*fill_cell_i
        sum_centrality[np.where(fill_cell_i ==1)]= np.sum(cross)*len(node_df)
        
        c.append(np.sum(cross))
        
    face_df['sum_degree']=c
    return sum_centrality

def mean_closeness_pc(img,
                           node_df,
                           face_df,
                            G,
                           it,
                           dil,
                           segmentation):
    """
    Create an nd_array. Each dilated apical surface is equal to the mean of the chosen centrality of signal 2 network contained in this volume

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    G : Graph from networkX
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um

    Return
    ------
    sum_centrality : nd_array, beware: the centrality has been multiplied by the number of node for representation purposes

    """
    
    sum_centrality = np.zeros(img.shape)
    centrality= np.zeros(img.shape)
    c=[]
    
    for k, v in  nx.closeness_centrality(G).items():
        coord_xk = int(node_df.x[k])
        coord_yk = int(node_df.y[k])
        coord_zk = int(node_df.z[k])
        centrality[coord_zk, coord_yk, coord_xk] = v

    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = centrality*fill_cell_i
        sum_centrality[np.where(fill_cell_i ==1)]= cross[cross!=0].mean()*len(node_df)
        
        c.append(np.sum(cross))
        
    face_df['mean_closeness']=c
    return sum_centrality



def mean_degree_pc(img,
                           node_df,
                           face_df,
                            G,
                           it,
                           dil,
                           segmentation):
    """
    Create an nd_array. Each dilated apical surface is equal to the mean of the chosen centrality of signal 2 network contained in this volume

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    G : Graph from networkX
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um

    Return
    ------
    sum_centrality : nd_array

    """
    
    sum_centrality = np.zeros(img.shape)
    centrality= np.zeros(img.shape)
    c=[]
    
    for k, v in  nx.degree_centrality(G).items():
        coord_xk = int(node_df.x[k])
        coord_yk = int(node_df.y[k])
        coord_zk = int(node_df.z[k])
        centrality[coord_zk, coord_yk, coord_xk] = v

    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = centrality*fill_cell_i
        sum_centrality[np.where(fill_cell_i ==1)]= cross[cross!=0].mean()*len(node_df)
        
        c.append(np.sum(cross))
        
    face_df['mean_degree']=c
    return sum_centrality



def mean_betweenness_pc(img,
                           node_df,
                           face_df,
                            G,
                           it,
                           dil,
                          segmentation):
    """
    Create an nd_array. Each dilated apical surface is equal to the mean of the chosen centrality of signal 2 network contained in this volume

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    G : Graph from networkX
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um

    Return
    ------
    sum_centrality : nd_array

    """
    
    sum_centrality = np.zeros(img.shape)
    centrality= np.zeros(img.shape)
    c=[]
    
    for k, v in  nx.betweenness_centrality(G).items():
        coord_xk = int(node_df.x[k])
        coord_yk = int(node_df.y[k])
        coord_zk = int(node_df.z[k])
        centrality[coord_zk, coord_yk, coord_xk] = v

    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = centrality*fill_cell_i
        sum_centrality[np.where(fill_cell_i ==1)]= cross[cross!=0].mean()*len(node_df)
        
        c.append(np.sum(cross))
        
    face_df['mean_betweenness']=c
    return sum_centrality

def max_degree_pc(img,
                           node_df,
                           face_df,
                            G,
                           it,
                           dil,
                           segmentation):
    """
    Create an nd_array. Each dilated apical surface is equal to the mean of the chosen centrality of signal 2 network contained in this volume

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    G : Graph from networkX
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um

    Return
    ------
    sum_centrality : nd_array

    """
    
    sum_centrality = np.zeros(img.shape)
    centrality= np.zeros(img.shape)
    c=[]
    
    for k in range(len(node_df)):
        centrality[int(node_df.z.iloc[k]), 
                   int(node_df.y.iloc[k]), 
                   int(node_df.x.iloc[k])] = node_df.nfil.iloc[k]

    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = centrality*fill_cell_i
        sum_centrality[np.where(fill_cell_i ==1)]= cross.max()
        
        c.append(np.sum(cross))
        
    face_df['mean_degree']=c
    return sum_centrality
