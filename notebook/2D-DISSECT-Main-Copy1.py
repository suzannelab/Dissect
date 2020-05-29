#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('/home/tmerle/DISSECT/Code2D')


# In[ ]:


from Dissect_allfunctions import *
import skel


# In[3]:


#%matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'inline')


#import pingouin as pg


# In[4]:


def RunDisperse2D(ImDir,ImName,Threshold,MSC=False):
    if MSC:
        os.system('mse '+ImDir+ImName+' -outDir '+ImDir+' -loadMSC '+ImDir+ImName+'.MSC'+' -cut '+Threshold+' -periodicity 0 -upSkl')
    else:
        os.system('mse '+ImDir+ImName+' -outDir '+ImDir+' -cut '+Threshold+' -periodicity 0 -upSkl')
                
    os.system('skelconv '+ImDir+ImName+'_c'+Threshold+'.up.NDskl -outDir '+ImDir+' -toFITS')
    os.system('skelconv '+ImDir+ImName+'_c'+Threshold+'.up.NDskl -outDir '+ImDir+' -to NDskl_ascii')
    
    return


# In[5]:


ImDir='/home/tmerle/DISSECT/Code2D/Test1/'
ImName1='C1-20171214_sqh-GFP_ap-alpha-cat-RFP_Dapi_WP2h-002-dorsal_AiSc_croped-bin4_red.tif'
ImName2='C2-20171214_sqh-GFP_ap-alpha-cat-RFP_Dapi_WP2h-002-dorsal_AiSc_croped-bin4_green.tif'


# In[6]:


import exifread

with open(ImDir+ImName1, 'rb') as f:
    tags = exifread.process_file(f)
print(tags['Image XResolution'])
print(tags['Image YResolution'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


N_2=2
tif2fits(ImDir+ImName1+'.fits',ImDir+ImName1,stack=True,N=N_2)
tif2fits(ImDir+ImName2+'.fits',ImDir+ImName2,stack=True,N=N_2)


# In[ ]:





# In[8]:


ImName1fits='C1-20171214_sqh-GFP_ap-alpha-cat-RFP_Dapi_WP2h-002-dorsal_AiSc_croped-bin4_red.tif.fits'
ImName2fits='C2-20171214_sqh-GFP_ap-alpha-cat-RFP_Dapi_WP2h-002-dorsal_AiSc_croped-bin4_green.tif.fits'


# In[9]:


Im1 = fits.getdata(ImDir+ImName1fits)
plt.figure()
plt.imshow(Im1,origin='lower')


# In[ ]:





# In[10]:


#RUN DISPERSE
Threshold='500'
#RunDisperse2D(ImDir,ImName1,Threshold,MSC=False)
Skel1 = fits.getdata(ImDir+ImName1fits+'_c'+Threshold+'.up.NDskl.fits')
skeleton = skel.Skel(ImDir+ImName1fits+'_c'+Threshold+'.up.NDskl.a.NDskl')

#binarisation du skelet pour vérifier ce qu'on a fait 
#(et constater qu'on a des filaments au milieu des cellules non rattachés d'un côté)
Skel1[np.where(Skel1 != 0)] = 1


fig = plt.figure(figsize=(15,10))
ax1= plt.subplot(1,2,1)
ax1.imshow(Im1,origin='lower')
ax2= plt.subplot(1,2,2,sharex=ax1,sharey=ax1)
ax2.imshow(Skel1,origin='lower')



# In[11]:


#Nettoyage du squelette. Fonction codée par Sophie en utilisant le module skel de Didier Vibert (astrophysics)

skeleton = skel.Skel(ImDir+ImName1fits+'_c'+Threshold+'.up.NDskl.a.NDskl')
clean_skeleton(skeleton, save=False)

#Tout ce qui suit sert à récupérer une image 

#les coordonnées des extrémités des segments composant le filament 0 :
print(skeleton.fil[0].points)


plt.figure(figsize=(15,15))
plt.imshow(Im1,origin='lower')
plt.scatter(skeleton.fil[0].points[:,0],skeleton.fil[0].points[:,1],color='red')

plt.figure(figsize=(15,15))
plt.imshow(Im1,origin='lower')
for i in range(skeleton.nfil):
    plt.scatter(skeleton.fil[i].points[:,0],skeleton.fil[i].points[:,1],color='red')
    
    

#Masque binaire et propre :  
MaskFil = FilMask_int(skeleton,Im1)

plt.figure(figsize=(17,17))
ax1 = plt.subplot(1,2,1)
ax1.imshow(Im1,origin='lower')
ax2 = plt.subplot(1,2,2,sharex=ax1,sharey=ax1)
ax2.imshow(MaskFil,origin='lower')
ax2.set_title('MaskFil')


# In[12]:


#Inversion de MaskFil pour le watershed
InvMaskFil = np.copy(MaskFil)
InvMaskFil[np.where(MaskFil == 0)] = 1
InvMaskFil[np.where(MaskFil == 1)] = 0

plt.figure(figsize=(10,10))
plt.imshow(InvMaskFil,origin='lower')
plt.title('InvMaskFil')


# In[ ]:





# In[13]:


#Elargissement des filaments Top Hat
KernelSize = 1
MASKFILTop = convolve(MaskFil,Tophat2DKernel(KernelSize))
MASKFILTop[np.where(MASKFILTop != 0.)] = 1.


plt.figure(figsize=(10,3))
ax1 = plt.subplot(1,2,1)
ax1.imshow(Im1,origin='lower')
ax1 = plt.subplot(1,2,2,sharex=ax1,sharey=ax1)
ax1.imshow(MASKFILTop,origin='lower')

#Elargissement des filaments Box
#KernelBoxSize =2
#MASKFILBox = convolve(Skel1,Box2DKernel(KernelBoxSize))
#SKEL1Box[np.where(SKEL1Box != 0.)] = 1.

#plt.figure()
#plt.imshow(SKEL1Box)
#plt.title('Skel1Box')

plt.figure(figsize=(10,10))
plt.imshow(MASKFILTop,origin='lower')
plt.title('MASKFILTop')


# In[14]:


#OTSU FILTER SUR IM1

KernelSize = 50
Lis1 = convolve(Im1,Tophat2DKernel(KernelSize))
plt.figure()

plt.imshow(Lis1,origin='lower')
plt.title('Lis1')

OtsuLis1 = otsufilter(Lis1)
plt.figure()
plt.imshow(OtsuLis1,origin='lower')
plt.title('OtsuLis1')

#AntiOtsu: Fond1 pour normalisation
AntiOtsu1 = ~OtsuLis1
plt.figure()
plt.imshow(AntiOtsu1,origin='lower')
plt.title('AntiOtsu1')


# In[ ]:





# In[15]:


Mask1 = MASKFILTop * OtsuLis1

plt.figure(figsize=(10,6))
ax1 = plt.subplot(1,2,1)
ax1.imshow(Mask1,origin='lower')
ax1.set_title('Mask1')
ax2 = plt.subplot(1,2,2,sharex=ax1,sharey=ax1)
ax2.imshow(MASKFILTop,origin='lower')
ax2.set_title('MASKFILTOP')


# In[16]:


AntiMask = np.copy(MASKFILTop)
AntiMask[np.where(MASKFILTop == 0)] = 1
AntiMask[np.where(MASKFILTop == 1)] = 0

plt.figure()
plt.imshow(AntiMask,origin='lower')
plt.title('AntiMask')


# In[17]:


Im1WoJunc = AntiMask * Im1
plt.figure()
plt.imshow(Im1WoJunc,origin='lower')
plt.title('Im1WoJunc')


# In[ ]:





# In[18]:


#Junc1
Junc1 = MASKFILTop * Im1
plt.figure()
plt.imshow(Junc1,origin='lower')
plt.title('Junc1')


#Fond 1
Fond1 = AntiOtsu1*Im1
plt.figure()
plt.imshow(Fond1,origin='lower')
plt.title('Fond1')


# In[19]:


#Calcul moy1
meanSig1_JuncSig1 = np.mean(Junc1[np.where(Junc1 != 0)]) / np.mean(Fond1[np.where(Fond1 != 0)])
print('mean sig1 normalised=',meanSig1_JuncSig1)

mean = np.mean (Im1[np.where(MASKFILTop == 1)]) / np.mean(Im1[np.where(~OtsuLis1)])
print('mean with 0', mean)


# In[20]:


#OTSU FILTER SUR IM2
Im2 = fits.getdata(ImDir+ImName2fits)
plt.figure()
plt.imshow(Im2,origin='lower')
plt.title('Im2')

#Lissage du signal
KernelSize = 20
Lis2 = convolve(Im2,Tophat2DKernel(KernelSize))
plt.figure()
plt.imshow(Lis2,origin='lower')
plt.title('Lis2')

OtsuLis2 = otsufilter(Lis2)
plt.figure()
plt.imshow(OtsuLis2,origin='lower')
plt.title('OtsuLis2')


# In[21]:



#AntiOtsu: Fond1 pour normalisation
AntiOtsu2 = ~OtsuLis2
plt.figure()
plt.imshow(AntiOtsu2,origin='lower')
plt.title('AntiOtsu2')


Fond2 = AntiOtsu2 * Im2
plt.figure()
plt.imshow(Fond2,origin='lower')
plt.title('Fond2')


OtsuIm2 = OtsuLis2 * Im2
plt.figure()
plt.imshow(OtsuIm2,origin='lower')
plt.title('OtsuIm2')


# In[22]:


#Apply MASKFILTop to Im2
Junc2 = MASKFILTop * Im2


Cell2 = AntiMask*Im2

plt.figure(figsize=(11,6))
ax1 = plt.subplot(1,2,1)
ax1.imshow(Junc2,origin='lower')
ax1.set_title('Junc2')
ax2 = plt.subplot(1,2,2,sharex=ax1,sharey=ax1)
ax2.imshow(Cell2,origin='lower')
ax2.set_title('Cell2')


plt.figure()
plt.imshow(Cell2,origin='lower')
plt.title('Cell2')


# In[23]:


Cell22 = AntiMask * Im2 * OtsuLis1

plt.figure(figsize=(9,10))
ax1 = plt.subplot(2,1,1)
ax1.imshow(Cell2,origin='lower')
ax1.set_title('Cell2')
ax2 = plt.subplot(2,1,2,sharex=ax1,sharey=ax1)
ax2.imshow(Cell22,origin='lower')
ax2.set_title('Cell22')



# In[24]:


#Mean filaments Im2 signal where filament Im1
meanSig2_JuncSig1 = np.mean(Im2[np.where(Junc2 != 0)]) / np.mean(Im2[np.where(Fond2 != 0)])
print(meanSig2_JuncSig1)

mean = np.mean (Im2[np.where(MASKFILTop == 1)]) / np.mean(Im2[np.where(~OtsuLis2)])
print(mean)
print('mean sig2 auc jonctions de sig1 normalised=',np.mean(Junc2[np.where(Junc2 != 0)]) / np.mean(Fond2[np.where(Fond2 != 0)]))


# In[25]:


# Disperse sur Im2 pour le fun


# Disperse 
Threshold='1.5e+03'
RunDisperse2D(ImDir,ImName2fits,Threshold,MSC=False)
Fil2 = fits.getdata(ImDir+ImName2fits+'_c'+Threshold+'.up.NDskl.fits')


plt.figure(figsize=(9,10))
ax1 = plt.subplot(2,1,1)
ax1.imshow(Im2,origin='lower')
ax1.set_title('Im2')
ax2 = plt.subplot(2,1,2,sharex=ax1,sharey=ax1)
ax2.imshow(Fil2,origin='lower')
ax2.set_title('Fil2')


# In[26]:


#Binarisation et élargissemment des filaments
Fil2[np.where(Fil2 != 0.)] = 1.
plt.figure()
plt.imshow(Fil2,origin='lower')
plt.title('Fil2')

KernelBoxSize = 2
Mask2 = convolve(Fil2,Box2DKernel(KernelBoxSize))
Mask2[np.where(Mask2 != 0.)] = 1.
plt.figure()
plt.imshow(Mask2,origin='lower')
plt.title('Mask2')



Sig2 = Mask2 * Im2
plt.figure(figsize=(15,15))
plt.imshow(Sig2,origin='lower')
plt.title('Sig2')


# In[27]:


print('mean sig2 =',np.mean(Sig2[np.where(Mask2 != 0)]) / np.mean(Im2[np.where(~OtsuLis2)]))


# In[ ]:





# In[28]:


#Nettoyage
Skel2 = fits.getdata(ImDir+ImName2fits+'_c'+Threshold+'.up.NDskl.fits')
skeleton = skel.Skel(ImDir+ImName2fits+'_c'+Threshold+'.up.NDskl.a.NDskl')
skeleton2 = skel.Skel(ImDir+ImName2fits+'_c'+Threshold+'.up.NDskl.a.NDskl')
clean_skeleton(skeleton2, save=False)
#Masque binaire et propre :  
MaskFil2 = FilMask_int(skeleton2,Im2)
plt.figure(figsize=(9,10))
ax1 = plt.subplot(2,1,1)
ax1.imshow(Fil2,origin='lower')
ax1.set_title('Fil2')
ax2 = plt.subplot(2,1,2,sharex=ax1,sharey=ax1)
ax2.imshow(MaskFil2,origin='lower')
ax2.set_title('MaskFil2')


# In[29]:


TopMaskFil2 = convolve(MaskFil2,Tophat2DKernel(1))
TopMaskFil2[np.where(TopMaskFil2 != 0.)] = 1.
plt.figure(figsize=(15,15))
plt.imshow(TopMaskFil2,origin='lower')
plt.title('TopMaskFil2')


# In[30]:





# In[31]:


#Global means in filaments for Sig1 and Sig2

normIm1 = NormaliseImage(Im1, 50) 
meanFil1 = np.mean (normIm1[np.where(MASKFILTop == 1)])
print('Sig1: mean in Sig1 filaments ',meanFil1)

normIm2 = NormaliseImage(Im2, 50) 
meanFil2_whereFil1 = np.mean (normIm2[np.where(MASKFILTop == 1)])
print('Sig2: mean in Sig1 filaments',meanFil2_whereFil1)


meanFil2 = np.mean (normIm2[np.where(Mask2 == 1)])
print('Sig2: mean in Sig2 filaments ',meanFil2)

meanFil2_clean = np.mean (normIm2[np.where(TopMaskFil2 == 1)])
print('Sig2: mean in clean Sig2 filaments ',meanFil2_clean)


# # Passons à la segmentation
# 

# In[32]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[33]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[34]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[35]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[36]:


min_area = 150
seg = segmentation(InvMaskFil,min_area)


# In[37]:


from matplotlib.colors import ListedColormap
rand = np.random.rand(256,3)
rand[0] = 0
cmap_rand = ListedColormap(rand)


plt.figure(figsize=(9,10))
ax1 = plt.subplot(2,1,1)
ax1.imshow(Im1,origin='lower')
ax1.set_title('Im1')
ax2 = plt.subplot(2,1,2,sharex=ax1,sharey=ax1)
ax2.imshow(seg,origin='lower',cmap=cmap_rand)
ax2.set_title('seg')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[38]:


junc_particular = JuncCell(seg, MaskFil, 58)

plt.figure()
plt.imshow(junc_particular)
plt.title('junc_particular')


# In[ ]:





# In[ ]:





# In[39]:


sigMain1='alphacat'
Dataframe1 = CellStatsMain(seg,MaskFil,normIm1,1,sigMain1)


# In[40]:


sigMain2 ="myosin"
Dataframe2 = CellStatsMain(seg,MaskFil,normIm2,1,sigMain2)


# In[41]:


pd.set_option('display.max_rows', None)


# In[42]:


Dataframe1


# In[113]:


Dataframe2


# In[114]:


import exifread

with open(ImDir+ImName1, 'rb') as f:
    tags = exifread.process_file(f)
print('X px/um =',tags['Image XResolution'])
print('Y px/um =',tags['Image YResolution'])


# In[115]:


Conv = tags['Image XResolution'].values[0].num/tags['Image XResolution'].values[0].den
print(Conv)


# In[116]:


Dataframe1['perimeter-um']=Dataframe1['perimeter']/Conv


# In[47]:


Dataframe1['area-um2']=Dataframe1['areaCell']/Conv**2


# In[48]:


Dataframe1


# In[49]:


plt.figure()
sns.set(style='white', font_scale=1.2)
sns.regplot(y='perimeter', x='meanJunc_myosin', fit_reg=True, data=Dataframe2, robust=True, n_boot=100);
plt.xlabel(' myosin mean in junctions');
plt.ylabel('perimeter');
plt.title('');
fig.tight_layout()
plt.show()


# In[50]:


plt.figure()
sns.set(style='white', font_scale=1.2)
sns.regplot(x='perimeter', y='meanJunc_myosin', fit_reg=True, data=Dataframe2, robust=True, n_boot=100);
plt.xlabel(' perimeter');
plt.ylabel('myosin mean in junctions');
plt.title('');
fig.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[51]:


plt.figure()
plt.scatter(Dataframe1['perimeter'],Dataframe1['areaCell'],color='black')
plt.xlabel('cell perimeter')
plt.ylabel('cell area')
xx = np.linspace(0,200,2000)
plt.plot(xx,xx**2/(4*np.pi))


# In[52]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[ ]:



    


# In[ ]:





# In[ ]:





# In[53]:


plt.figure()
sns.set(style='white', font_scale=1.2)
sns.regplot(y='areaCell', x='meanJunc_myosin', fit_reg=True, data=Dataframe2, robust=True, n_boot=100);
plt.xlabel('meanJunc_myosin');
plt.ylabel('areaCell');
plt.title('');

'''a, b = linear_fit(Dataframe2,ycol='areaCell', xcol='meanJunc_myosin')
xx = np.linspace(Dataframe2['meanJunc_myosin'].min(),Dataframe2['meanJunc_myosin'].max(),100)
plt.plot(xx,a*xx+b,color='red')
'''

fig.tight_layout()
plt.show()


# In[54]:


import scipy.stats as st


# In[55]:


st.spearmanr(Dataframe1['perimeter'],Dataframe1['areaCell'])


# In[56]:


st.pearsonr(Dataframe1['perimeter'],Dataframe1['areaCell'])


# In[57]:


sigMain='alphacat'
init = np.zeros((len(np.unique(seg)[2:]),8))
EmbryoStats = pd.DataFrame(data=init,columns=['meanCell_'+sigMain,'stdCell_'+sigMain,'semCell_'+sigMain,'areaCell','meanJunc_'+sigMain,'stdJunc_'+sigMain,'semJunc_'+sigMain,'lenJunc']) 
for i in np.unique(seg)[2:]:
    EmbryoStats['meanCell_'+sigMain][i] = i


# # Ellipses

# In[58]:


from astropy.modeling import models, fitting
from astropy.modeling.models import Ellipse2D
from astropy.coordinates import Angle
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
# fitting procedure
fit = fitting.SimplexLSQFitter() 
#fit = fitting.LevMarLSQFitter()


# In[ ]:





# In[59]:


def Cell(seg,i):   
    segmentationi = np.zeros_like(seg)
    #for each cell get contour pixels
    segmentationi[np.where(seg == i)] = 1
    return segmentationi

y, x = np.mgrid[0:np.shape(seg)[0], 0:np.shape(seg)[1]] #grille 
init = np.zeros((len(np.unique(seg)[2:]),9))
#Dataframe_geo = pd.DataFrame(data=init,columns=['areaCell','perimeter','x_0','y_0','a','b','theta'])
Dataframe_geo = pd.DataFrame(data=init,columns=['perimeter','perimeter um','cellArea','cellArea um2',
                                                    'x_0','y_0','a','b','theta'])

for ind,i in enumerate(np.unique(seg)[2:]):
    print(i)
    JuncCellMaski = JuncCell(seg,MaskFil,i) 
    # enlarge through smoothing 2*KernelSize+1
    JuncCellMaski_conv = convolve(JuncCellMaski,Tophat2DKernel(KernelSize))
    JuncCellMaski_conv[np.where(JuncCellMaski_conv != 0)] = 1
        
    ### multiply this mask of filaments around cell i with pixels from Im
    #JuncCell_i=JuncCellMaski_conv*Im
        
    #compute the cell area and perimeter          
    Dataframe_geo ['perimeter'][ind] = len(np.where(JuncCellMaski == 1)[0])
    Dataframe_geo ['perimeter um'][ind] = len(np.where(JuncCellMaski == 1)[0]) / Conv
    Dataframe_geo ['cellArea'][ind] =  len(np.where(seg == i)[0])
    Dataframe_geo ['cellArea um2'][ind] =  len(np.where(seg == i)[0]) / Conv**2
   

    #Ellipse
    imCell = Cell(seg,i)
    # gaussian fit (to estimate x_0, y_0 and theta)
    gi = models.Gaussian2D(amplitude = 1., x_mean = np.where(imCell==1)[1][int(len(np.where(imCell == 1)[0]) / 2) + 1],
                           y_mean = np.where(imCell == 1)[0][int(len(np.where(imCell == 1)[0]) / 2) + 1], 
                           x_stddev=10, y_stddev=10, theta=0.0) #modèle initial
    g1 = fit(gi, x, y, imCell, maxiter=100000) #fit une gaussienne avec les parametres initiaux donnés
    # initial model: fais une ellipse fixe avec les parametres trouvés par le fit gaussien
    ei1 = models.Ellipse2D(amplitude=1., x_0=g1.x_mean, y_0=g1.y_mean, a=g1.x_stddev, b=g1.y_stddev, theta=g1.theta, fixed={'x_0': True, 'y_0':True, 'theta':True})
    #fitted model : on fit une ellipse et donc on change a et b 
    e1 = fit(ei1, x, y, imCell, maxiter=100000)
    e1.amplitude = 1
    z1 = e1(x, y)
    
    if e1.b.value > e1.a.value:
        e1.theta.value = e1.theta.value + np.pi/2 
        c = e1.a.value  
        e1.a.value = e1.b.value 
        e1.b.value = c

    Dataframe_geo['x_0'][ind] = e1.x_0.value
    Dataframe_geo['y_0'][ind] = e1.y_0.value
    Dataframe_geo['a'][ind] = e1.a.value
    Dataframe_geo['b'][ind] = e1.b.value
    Dataframe_geo['theta'][ind] = e1.theta.value
    
Dataframe_geo['e'] = np.sqrt( 1 - (Dataframe_geo['b']**2 / (Dataframe_geo['a']**2) ))
Dataframe_geo['an'] = Dataframe_geo['a'] / Dataframe_geo['b']
Dataframe_geo['a um'] = Dataframe_geo['a'] / Conv
Dataframe_geo['b um'] = Dataframe_geo['b'] / Conv 


# In[60]:


Dataframe_geo 


# In[95]:


from multiprocessing import Pool
import time


# In[96]:


def Cell(seg,i):   
    segmentationi = np.zeros_like(seg)
    #for each cell get contour pixels
    segmentationi[np.where(seg == i)] = 1
    return segmentationi

def cellgeoi(inputs):
    ind,i,df = inputs
    print(i)

    JuncCellMaski = JuncCell(seg,MaskFil,i) 
    # enlarge through smoothing 2*KernelSize+1
    JuncCellMaski_conv = convolve(JuncCellMaski,Tophat2DKernel(KernelSize))
    JuncCellMaski_conv[np.where(JuncCellMaski_conv != 0)] = 1
        
    ### multiply this mask of filaments around cell i with pixels from Im
    #JuncCell_i=JuncCellMaski_conv*Im
        
    #compute the cell area and perimeter          
    df['perimeter'][ind] = len(np.where(JuncCellMaski == 1)[0])
    df['perimeter um'][ind] = len(np.where(JuncCellMaski == 1)[0]) / Conv
    df['cellArea'][ind] =  len(np.where(seg == i)[0])
    df['cellArea um'][ind] =  len(np.where(seg == i)[0]) / Conv**2
   

    #Ellipse
    imCell = Cell(seg,i)
    # gaussian fit (to estimate x_0, y_0 and theta)
    gi = models.Gaussian2D(amplitude = 1., x_mean = np.where(imCell==1)[1][int(len(np.where(imCell == 1)[0]) / 2) + 1],
                           y_mean = np.where(imCell == 1)[0][int(len(np.where(imCell == 1)[0]) / 2) + 1], 
                           x_stddev=10, y_stddev=10, theta=0.0) #modèle initial
    g1 = fit(gi, x, y, imCell, maxiter=100000) #fit une gaussienne avec les parametres initiaux donnés
    # initial model: fais une ellipse fixe avec les parametres trouvés par le fit gaussien
    ei1 = models.Ellipse2D(amplitude=1., x_0=g1.x_mean, y_0=g1.y_mean, a=g1.x_stddev, b=g1.y_stddev, theta=g1.theta, fixed={'x_0': True, 'y_0':True, 'theta':True})
    #fitted model : on fit une ellipse et donc on change a et b 
    e1 = fit(ei1, x, y, imCell, maxiter=100000)
    e1.amplitude = 1
    z1 = e1(x, y)
    
    if e1.b.value > e1.a.value:
        e1.theta.value = e1.theta.value + np.pi/2 
        c = e1.a.value  
        e1.a.value = e1.b.value 
        e1.b.value = c

    df['x_0'][ind] = e1.x_0.value
    df['y_0'][ind] = e1.y_0.value
    df['a'][ind] = e1.a.value
    df['b'][ind] = e1.b.value
    df['theta'][ind] = e1.theta.value
    
    return

start = time.time()
y, x = np.mgrid[0:np.shape(seg)[0], 0:np.shape(seg)[1]] #grille 
init = np.zeros((len(np.unique(seg)[2:]),9))
Dataframe_geo_para = pd.DataFrame(data=init,columns=['areaCell','perimeter','x_0','y_0','a','b','theta'])

all_inputs = zip(range(len(np.unique(seg)[2:])),np.unique(seg)[2:],repeat(Dataframe_geo_para))

pool = Pool()
CellGeo= pool.map_async(cellgeoi, all_inputs)
pool.close()
pool.join()
CellGeo.get()

Dataframe_geo_para['e'] = np.sqrt( 1 - (Dataframe_geo_para['b']**2 / (Dataframe_geo_para['a']**2) ))
Dataframe_geo_para['an'] = Dataframe_geo_para['a'] / Dataframe_geo_para['b']

end=time.time()

print('Cell Geo 4 cores ran in ',(end-start)/60.,' min')


# In[ ]:





# In[114]:


Dataframe_geo.to_csv('Dataframe_geo.csv',index=False)


# In[61]:


Dataframe_geo = pd.read_csv('/home/tmerle/Dataframe_geo.csv')


# In[62]:


Dataframe_geo


# In[63]:


np.mean(Dataframe_geo['a um'])


# In[64]:


from matplotlib.colors import LogNorm,PowerNorm


# In[65]:


plt.figure(figsize=(10,10))
plt.imshow(seg,origin='lower',cmap=cmap_rand)
for i in range(len(Dataframe_geo)):
    startx = Dataframe_geo['x_0'][i] - Dataframe_geo['a'][i] * np.cos(Dataframe_geo['theta'][i])
    starty = Dataframe_geo['y_0'][i] - Dataframe_geo['a'][i] * np.sin(Dataframe_geo['theta'][i])
    endx = Dataframe_geo['x_0'][i] + Dataframe_geo['a'][i] * np.cos(Dataframe_geo['theta'][i])
    endy = Dataframe_geo['y_0'][i] + Dataframe_geo['a'][i] * np.sin(Dataframe_geo['theta'][i])
    
    
    plt.plot([startx, endx], [starty, endy], color='black')


# In[66]:


fig, ax = plt.subplots(figsize=(10,10))
ax.set_aspect('equal')
#plt.imshow(seg,origin='lower',cmap=cmap_rand)
for i in range(len(Dataframe_geo)):
    startx = Dataframe_geo['x_0'][i] - Dataframe_geo['a'][i] * np.cos(Dataframe_geo['theta'][i])
    starty = Dataframe_geo['y_0'][i] - Dataframe_geo['a'][i] * np.sin(Dataframe_geo['theta'][i])
    endx = Dataframe_geo['x_0'][i] + Dataframe_geo['a'][i] * np.cos(Dataframe_geo['theta'][i])
    endy = Dataframe_geo['y_0'][i] + Dataframe_geo['a'][i] * np.sin(Dataframe_geo['theta'][i])

    colori = plt.cm.jet(Dataframe_geo['e'][i]) # r is 0 to 1 inclusive
        
    plt.plot([startx, endx], [starty, endy], color=colori)
   
#plt.colorbar(Dataframe_geo_bis['e'])


# In[67]:


from matplotlib.collections import LineCollection

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


# In[68]:


'''Fonction pour la jolie color bar'''

from mpl_toolkits import axes_grid1

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


# In[69]:


fig = plt.figure(figsize=(10,9))
ax.set_aspect('equal')
#plt.imshow(seg,origin='lower',cmap='viridis')
plt.imshow(InvMaskFil,origin='lower',cmap='gist_gray')
xs = []
ys = []
for i in range(len(Dataframe_geo)):
    startx = Dataframe_geo['x_0'][i] - Dataframe_geo['a'][i] * np.cos(Dataframe_geo['theta'][i])
    starty = Dataframe_geo['y_0'][i] - Dataframe_geo['a'][i] * np.sin(Dataframe_geo['theta'][i])
    endx = Dataframe_geo['x_0'][i] + Dataframe_geo['a'][i] * np.cos(Dataframe_geo['theta'][i])
    endy = Dataframe_geo['y_0'][i] + Dataframe_geo['a'][i] * np.sin(Dataframe_geo['theta'][i])
    
    xs.append([startx,endx])
    ys.append([starty,endy])
c = Dataframe_geo['an']
lc = multiline(xs, ys, c, cmap='jet', lw=2, clim = (0,8))
add_colorbar(lc, aspect=0.05, pad_fraction=0.5, label = 'Anisotropy (a/b)')


# In[70]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[71]:


cmap1 = plt.cm.jet(np.arange(256))
cmap1[0] = [1., 1., 1., 1.]
cmap1 = ListedColormap(cmap1, name='myColorM', N=cmap1.shape[0])



fig, ax = plt.subplots(figsize=(10,10))
plt.imshow(InvMaskFil,origin='lower',cmap='gist_gray')
q = ax.quiver(Dataframe_geo['x_0'], Dataframe_geo['y_0'], 
              0.5*Dataframe_geo['a'] * np.cos(Dataframe_geo['theta']), 
              0.5*Dataframe_geo['b'] * np.sin(Dataframe_geo['theta']), 
              Dataframe_geo['a']/Dataframe_geo['b'],
              units='xy' ,scale=0.5, 
              angles = Dataframe_geo['theta'] * 180 / np.pi,
              cmap='jet',
              clim = (0,8)
                                                    )
ax.set_aspect('equal')
#lc = multiline(xs, ys, c, cmap='jet', lw=2)
#axcb = fig.colorbar(lc)at's why it's called a la Turk.

add_colorbar(q, aspect=0.05, pad_fraction=0.5, label = 'Anisotropy (a/b)')


# In[77]:


#Longueur de la flêche -> demi grand axe
#Couleur de la flêche -> Orientation (theta)


fig, ax = plt.subplots(figsize=(10,10))
plt.imshow(InvMaskFil,origin='lower',cmap='gist_gray')
q = ax.quiver(Dataframe_geo['x_0'], Dataframe_geo['y_0'], 
              0.5*Dataframe_geo['a'] * np.cos(Dataframe_geo['theta']), 
              0.5*Dataframe_geo['b'] * np.sin(Dataframe_geo['theta']), 
              Dataframe_geo['theta'],
              units='xy' ,scale=0.5, 
              angles = Dataframe_geo['theta'] * 180 / np.pi,
              cmap='jet',
              clim = (0,np.pi/2)
                                                    )
ax.set_aspect('equal')
#lc = multiline(xs, ys, c, cmap='jet', lw=2)
#axcb = fig.colorbar(lc)at's why it's called a la Turk.

add_colorbar(q, aspect=0.05, pad_fraction=0.5, label = 'Orientation (0;Pi/2)')


# In[1]:


print('blabla')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




