
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from tifffile import tifffile

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
    for i, row in node_df.iterrows() :
        dic_pos[i] = np.array([ node_df.x[i], node_df.y[i], node_df.z[i] ])

     
    return node_df, link_df, G



def calculate_centrality(node_df, G, measure, measure_name):

    """
    Add centrality column to node_df

    Parameters
    ----------
    node_df: dataframe from create_network
    G: Graph from create_create_network
    measure: centrality measure (degree, betweenness, closeness). Write: nx.degree_centrality(G)
    measure_name: string. Name of the new colomn inthe dataframe
    """
    node_df[measure_name] = np.nan
    for k, v in measure.items():
        node_df[measure_name][k]=v
    



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
    

def branch_analysis(df_junc, skel):
    """
    Calculate straight length and tortuosity for each branch (from node to node) of the network

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

    for i in range(len(df_junc)):
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
    
    df_junc['s_xyz'] = ls 
    df_junc['t_xyz'] = lt 
    df_junc['straight_length'] = delta_st
    df_junc['tortuosity'] = t
    
    return df_junc

def global_network_property(df_junc, node_df):
    """
    Calculate properties of the total network and store it in a datafarme. To be applied after branch_analysis for the global tortuosity

    Parameters
    ----------
    df_junc : 

    Return
    ------
    global_network_df : DataFrame

    """
    
    columns_name = ['Length', 'Nodes', 'End_nodes', 'Branches', 'Tortuosity']
    init = np.zeros((1, len(columns_name)))
    global_network_df = pd.DataFrame(data=init, columns=columns_name)
    
    global_network_df.Length = df_junc['length_um'].sum(axis = 0, skipna = True)
    global_network_df.Nodes = len(node_df)
    global_network_df.Branches = len(df_junc)
    global_network_df.End_nodes = len(node_df[node_df.nfil==1])
    
    #Tortuosity
    global_network_df.Tortuosity = df_junc['tortuosity'].mean(axis = 0, skipna = True)
        
    return global_network_df

def nb_node(img, 
            node_df, 
            face_df, 
            edge_df, 
            vert_df,
            points_df, 
            it, 
            dil,
            pixel_size):
    """
    Create an nd_array. Each dilated apical surface is equal to the number of contained node of signal 2

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df:
    edge_df:
    vert_df:
    points_df:
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

    for i in range(len(face_df)) : 
        cell_i = enlarge_face_plane(img,
                       face_df,
                       edge_df,
                       vert_df,
                       points_df,
                       i,
                       dil,
                       pixel_size)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        #dil_fill_cell_i = ndimage.binary_dilation(fill_cell_i).astype(int)
        cross = nodes*fill_cell_i
        nb_nodes[np.where(fill_cell_i ==1)]= np.count_nonzero(cross)
        
    return nb_nodes

def sum_centrality_percell(img,
                           measure,
                           node_df,
                           face_df,
                           edge_df,
                           vert_df,
                           points_df,
                           it,
                           dil,
                           pixel_size):
    """
    Create an nd_array. Each dilated apical surface is equal to the sum of the chosen centrality of signal 2 network contained in this volume

    Parameters
    ----------
    img: nd_array
    measure : nx.degree_centrality(G) or nx.closeness_centrality(G) or nx.betweenness_centrality(G)
    node_df: dataframe from create_network
    face_df:
    edge_df:
    vert_df:
    points_df:
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um
    pixel_size: dict

    Return
    ------
    sum_centrality : nd_array

    """
    
    sum_centrality = np.zeros(img.shape)
    
    centrality= np.zeros(img.shape)
    for k, v in  measure.items():
        coord_xk = int(node_df.x[k])
        coord_yk= int(node_df.y[k])
        coord_zk= int(node_df.z[k])
        centrality[coord_zk, coord_yk, coord_xk] = v

    for i in range(len(face_df)) : 
        cell_i = enlarge_face_plane(img,
                       face_df,
                       edge_df,
                       vert_df,
                       points_df,
                       i,
                       dil,
                       pixel_size)

        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        #dil_fill_cell_i = ndimage.binary_dilation(fill_cell_i).astype(int)
        cross = centrality*fill_cell_i
        sum_centrality[np.where(fill_cell_i ==1)]= np.count_nonzero(cross)
        
    return sum_centrality

def mean_centrality_percell(img,
                           measure,
                           node_df,
                           face_df,
                           edge_df,
                           vert_df,
                           points_df,
                           it,
                           dil,
                           pixel_size):
    
    mean_centrality = np.zeros(img.shape)
    
    centrality= np.zeros(img.shape)
    for k, v in  measure.items():
        coord_xk = int(node_df.x[k])
        coord_yk= int(node_df.y[k])
        coord_zk= int(node_df.z[k])
        centrality[coord_zk, coord_yk, coord_xk] = v

    for i in range(len(face_df)) : 
        cell_i = enlarge_face_plane(img,
                       face_df,
                       edge_df,
                       vert_df,
                       points_df,
                       i,
                       dil,
                       pixel_size)

        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        #dil_fill_cell_i = ndimage.binary_dilation(fill_cell_i).astype(int)
        cross = centrality*fill_cell_i
        

        mean_centrality[np.where(fill_cell_i ==1)]=  cross[cross!=0].mean()
        
    return mean_centrality
