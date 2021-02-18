#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
   Warning: I work with images containing a black background
'''

import mahotas as mh
import glob, os
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import KFold

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from skimage import io

## unit test is ok ###

## nb class == 4 
## Alternaria , Mycosphaerella, Pseudocercosporella, Sans_Symptomes  
## not L_maculans

## extract haralick features ##

## 'copy-pasta'

Alt = '/home/mpalerme/Documents/Atipical_traitement_element_Lydia/photo_jeu_reduit/Alt_bdb_cut2_max' 
Big = '/home/mpalerme/Documents/Atipical_traitement_element_Lydia/photo_jeu_reduit/Big_bdb_cut2_max'
Mac = '/home/mpalerme/Documents/Atipical_traitement_element_Lydia/photo_jeu_reduit/Mac_bdb_cut2_max'
Mil = '/home/mpalerme/Documents/Atipical_traitement_element_Lydia/photo_jeu_reduit/Mil_bdb_cut2_max'
Myc = '/home/mpalerme/Documents/Atipical_traitement_element_Lydia/photo_jeu_reduit/Myc_bdb_cut2_max'
Pse = '/home/mpalerme/Documents/Atipical_traitement_element_Lydia/photo_jeu_reduit/Pse_bdb_cut2_max'
Syl = '/home/mpalerme/Documents/Atipical_traitement_element_Lydia/photo_jeu_reduit/Syl_bdb_cut2_max'

y=np.array([])
X=[]
u=1.0
for where in [Alt, Big, Mac, Mil, Myc, Pse, Syl]:
    ones=[]
    os.chdir(where)
    k=1.0
    for file in sorted(glob.glob("*ecto*.tiff")):
        print(k/len(glob.glob("*ecto*.tiff"))) 
        image = mh.imread(file)
        ## NB : skimage to mahotas, careful with differences in type 'image'       
        ############
        ## test of features from hsv instead of rgb  ###      
        ############
        # image=skimage.color.rgb2hsv(image)
        # r=(image[:,:,0]*255).astype(int) #  h
        # g=(image[:,:,1]*255).astype(int) #  s
        # b=(image[:,:,2]*255).astype(int) #  v
        rR=image[:,:,0] # r
        gR=image[:,:,1] # g
        bR=image[:,:,2] # b 
        #rR=np.where(r==255, 0, r)
        #gR=np.where(g==255, 0, g)
        #bR=np.where(b==255, 0, b)
        ############
        ## test of other features in addition to haralick ### 
        ############        
        # imgG = mh.colors.rgb2grey(image)
        # imgGi = imgG.astype(int)
        # imgGiR=np.where(imgGi==255, 0, imgGi)
        # res1=mh.features.haralick(imgGiR,ignore_zeros=True,return_mean=True)
        # res2=mh.features.zernike_moments(imgGiR,radius=1000)
        # res=np.concatenate((res1,res2))
        res1 = np.zeros((13,))
        res2 = np.zeros((13,))
        res3 = np.zeros((13,))
        im_gray = io.imread(file, True, 'matplotlib')
        if len(np.nonzero(im_gray)[0]) > 0:
            try:
                res1=mh.features.haralick(rR,ignore_zeros=True,return_mean=True)
            except ValueError:
                pass
            try:
                res2=mh.features.haralick(gR,ignore_zeros=True,return_mean=True)
            except ValueError:
                pass
            try:
                res3=mh.features.haralick(bR,ignore_zeros=True,return_mean=True)
            except ValueError:
                pass
        # res1=mh.features.haralick(rR,ignore_zeros=False,return_mean=True)
        # res2=mh.features.haralick(gR,ignore_zeros=False,return_mean=True)
        # res3=mh.features.haralick(bR,ignore_zeros=False,return_mean=True)
        res=np.concatenate((res1,res2,res3))
        ones.append(res)
        k=k+1.0
    print("Done "+where)
    y=np.concatenate([y,np.zeros(len(glob.glob("*ecto*.tiff")))+u])
    ones=np.stack(ones)
    X.append(ones)
    u=u+1.0

Xrecto=X
yrecto=y

y=np.array([])
X=[]
u=1.0
for where in [Alt, Big, Mac, Mil, Myc, Pse, Syl]:
    ones=[]
    os.chdir(where)
    k=1.0
    for file in sorted(glob.glob("*erso*.tiff")):
        print(k/len(glob.glob("*erso*.tiff"))) 
        image = mh.imread(file)
        rR=image[:,:,0] # r
        gR=image[:,:,1] # g
        bR=image[:,:,2] # b 
        #rR=np.where(r==255, 0, r)
        #gR=np.where(g==255, 0, g)
        #bR=np.where(b==255, 0, b)  
        res1 = np.zeros((13,))
        res2 = np.zeros((13,))
        res3 = np.zeros((13,))
        im_gray = io.imread(file, True, 'matplotlib')
        if len(np.nonzero(im_gray)[0]) > 0:
            try:
                res1=mh.features.haralick(rR,ignore_zeros=True,return_mean=True)
            except ValueError:
                pass
            try:
                res2=mh.features.haralick(gR,ignore_zeros=True,return_mean=True)
            except ValueError:
                pass
            try:
                res3=mh.features.haralick(bR,ignore_zeros=True,return_mean=True)
            except ValueError:
                pass
        # res1=mh.features.haralick(rR,ignore_zeros=False,return_mean=True)
        # res2=mh.features.haralick(gR,ignore_zeros=False,return_mean=True)
        # res3=mh.features.haralick(bR,ignore_zeros=False,return_mean=True)
        res=np.concatenate((res1,res2,res3))
        ones.append(res)
        k=k+1.0
    print("Done "+where)
    y=np.concatenate([y,np.zeros(len(glob.glob("*erso*.tiff")))+u])
    ones=np.stack(ones)
    X.append(ones)
    u=u+1.0

Xverso=X
yverso=y

Xr=np.vstack(Xrecto)
Xv=np.vstack(Xverso)


### Alternaria :  56
### Myco :  84 
### dernier : 106 (erreur en 149/2)
### 214

## Careful : 1 image add a Recto but no Verso (?!)

#idx=list(range(214))+list(range(215,270))
#Xrb=Xr[idx]
Xrb=Xr

Xtot=[]
for i in range(Xrb.shape[0]):
    Xtot.append(np.concatenate((Xrb[i],Xv[i])))
    
## Try a simplistic classification (both in term of ROI, features and classifier)
    


# from sklearn.datasets import make_classification

Xb=np.vstack(Xtot) # (X)
#yb=y[idx]
yb=y
kf = KFold(len(yb))

acc1=[]
acc2=[]
acc3=[]
preds1=[]
trues1=[]
preds2=[]
trues2=[]
preds3=[]
trues3=[]
for train, test in kf.split(Xb):
    print("Train : ", train)
    print('test : ', test) 
    X_train, X_test, y_train, y_test = Xb[train], Xb[test], yb[train], yb[test]    
    clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X_train, y_train) 
    clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X_train, y_train)    
    clf3 = QuadraticDiscriminantAnalysis().fit(X_train, y_train)    
    acc1.append(clf1.score(X_test,y_test))
    acc2.append(clf2.score(X_test,y_test))
    acc3.append(clf3.score(X_test,y_test))
    preds1.append(clf1.predict(X_test))
    trues1.append(y_test)
    preds2.append(clf2.predict(X_test))
    trues2.append(y_test)
    preds3.append(clf3.predict(X_test))
    trues3.append(y_test)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    names=['Alt', 'Big', 'Mac', 'Mil', 'Myc', 'Pse', 'Syl']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def cal_visu_conf(preds, trues):
    p=np.concatenate(preds)
    t=np.concatenate(trues)

    con=confusion_matrix(t,p) #,labels=)

    cm_normalized = con.astype('float') / con.sum(axis=1)[:, np.newaxis]

    plot_confusion_matrix(cm_normalized)

    print(cm_normalized)  




print("LINEAR Auto")
print("Acc = " + str(np.array(acc1).mean()))
print("Kappa = " + str(cohen_kappa_score(preds1, trues1)))
cal_visu_conf(preds1, trues1)

print("LINEAR")
print("Acc = " + str(np.array(acc2).mean()))
print("Kappa = " + str(cohen_kappa_score(preds2, trues2)))
cal_visu_conf(preds2, trues2)

print("QUADRATIC")
print("Acc = " + str(np.array(acc3).mean()))
print("Kappa = " + str(cohen_kappa_score(preds3, trues3)))
cal_visu_conf(preds3, trues3)



