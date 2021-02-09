#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
   Warning: I work with images containing a black background
'''

import mahotas as mh
import glob, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import KFold

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from skimage import io


import SimpleITK as sitk
from radiomics.glrlm import RadiomicsGLRLM
## unit test is ok ###

## nb class == 4
## Alternaria , Mycosphaerella, Pseudocercosporella, Sans_Symptomes
## not L_maculans

## extract haralick features ##

## 'copy-pasta'

Alt = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Alt'
Big = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Big'
Mac = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Mac'
Mil = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Mil'
Myc = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Myc'
Pse = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Pse'
Syl = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Syl'

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
        a_glrlm = np.zeros((16,))
        # Not use background for moments
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

            # Transform numpy grayscale image to simpleITK grayscale imad
            im_itk = sitk.GetImageFromArray(im_gray)
            # Create mask
            mask = im_gray.copy()
            mask[mask > 0] = 1
            mask = sitk.GetImageFromArray(mask)
            # Calculate Gray Level Run Length Matrix (GLRLM) Features
            glrlm = RadiomicsGLRLM(im_itk, mask)
            glrlm._initCalculation()
            # Extract features
            a_glrlm = [glrlm.getSmallAreaEmphasisFeatureValue()[0],
                       glrlm.getLargeAreaEmphasisFeatureValue()[0],
                       glrlm.getGrayLevelNonUniformityFeatureValue()[0],
                       glrlm.getGrayLevelNonUniformityNormalizedFeatureValue()[0],
                       glrlm.getSizeZoneNonUniformityFeatureValue()[0],
                       glrlm.getSizeZoneNonUniformityNormalizedFeatureValue()[0],
                       glrlm.getZonePercentageFeatureValue()[0],
                       glrlm.getGrayLevelVarianceFeatureValue()[0],
                       glrlm.getZoneVarianceFeatureValue()[0],
                       glrlm.getZoneEntropyFeatureValue()[0],
                       glrlm.getLowGrayLevelZoneEmphasisFeatureValue()[0],
                       glrlm.getHighGrayLevelZoneEmphasisFeatureValue()[0],
                       glrlm.getSmallAreaLowGrayLevelEmphasisFeatureValue()[0],
                       glrlm.getSmallAreaHighGrayLevelEmphasisFeatureValue()[0],
                       glrlm.getLargeAreaLowGrayLevelEmphasisFeatureValue()[0],
                       glrlm.getLargeAreaHighGrayLevelEmphasisFeatureValue()[0]
                ]

            rR = rR[np.nonzero(im_gray)]
            gR = gR[np.nonzero(im_gray)]
            bR = bR[np.nonzero(im_gray)]

        # define moment mean, standar deviation, variance, min, max
        momt_R = [np.mean(rR), np.std(rR), np.var(rR), np.min(rR), np.max(rR)]
        momt_G = [np.mean(gR), np.std(gR), np.var(gR), np.min(gR), np.max(gR)]
        momt_B = [np.mean(bR), np.std(bR), np.var(bR), np.min(bR), np.max(bR)]
        res=np.concatenate((res1, res2, res3, momt_R, momt_G, momt_B, a_glrlm))
        # res=np.concatenate((momt_R, momt_G, momt_B))
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
        a_glrlm = np.zeros((16,))

        # Not use background for moments
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

            # Transform numpy grayscale image to simpleITK grayscale imad
            im_itk = sitk.GetImageFromArray(im_gray)
            # Create mask
            mask = im_gray.copy()
            mask[mask > 0] = 1
            mask = sitk.GetImageFromArray(mask)
            # Calculate Gray Level Run Length Matrix (GLRLM) Features
            glrlm = RadiomicsGLRLM(im_itk, mask)
            glrlm._initCalculation()
            # Extract features
            a_glrlm = [glrlm.getSmallAreaEmphasisFeatureValue()[0],
                       glrlm.getLargeAreaEmphasisFeatureValue()[0],
                       glrlm.getGrayLevelNonUniformityFeatureValue()[0],
                       glrlm.getGrayLevelNonUniformityNormalizedFeatureValue()[0],
                       glrlm.getSizeZoneNonUniformityFeatureValue()[0],
                       glrlm.getSizeZoneNonUniformityNormalizedFeatureValue()[0],
                       glrlm.getZonePercentageFeatureValue()[0],
                       glrlm.getGrayLevelVarianceFeatureValue()[0],
                       glrlm.getZoneVarianceFeatureValue()[0],
                       glrlm.getZoneEntropyFeatureValue()[0],
                       glrlm.getLowGrayLevelZoneEmphasisFeatureValue()[0],
                       glrlm.getHighGrayLevelZoneEmphasisFeatureValue()[0],
                       glrlm.getSmallAreaLowGrayLevelEmphasisFeatureValue()[0],
                       glrlm.getSmallAreaHighGrayLevelEmphasisFeatureValue()[0],
                       glrlm.getLargeAreaLowGrayLevelEmphasisFeatureValue()[0],
                       glrlm.getLargeAreaHighGrayLevelEmphasisFeatureValue()[0]
                ]

            rR = rR[np.nonzero(im_gray)]
            gR = gR[np.nonzero(im_gray)]
            bR = bR[np.nonzero(im_gray)]
        # define moment mean, standar deviation, variance, min, max
        momt_R = [np.mean(rR), np.std(rR), np.var(rR), np.min(rR), np.max(rR)]
        momt_G = [np.mean(gR), np.std(gR), np.var(gR), np.min(gR), np.max(gR)]
        momt_B = [np.mean(bR), np.std(bR), np.var(bR), np.min(bR), np.max(bR)]
        res=np.concatenate((res1, res2, res3, momt_R, momt_G, momt_B, a_glrlm))
        # res=np.concatenate((momt_R, momt_G, momt_B))
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
    # clf1 = GradientBoostingClassifier().fit(X_train, y_train)
    clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X_train, y_train)
    clf3 = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
    # acc1.append(clf1.score(X_test,y_test))
    acc2.append(clf2.score(X_test,y_test))
    acc3.append(clf3.score(X_test,y_test))
    # preds1.append(clf1.predict(X_test))
    # trues1.append(y_test)
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




# print("GRADIENT")
# print(np.array(acc1).mean())
# cal_visu_conf(preds1, trues1)

print("LINEAR")
print(np.array(acc2).mean())
cal_visu_conf(preds2, trues2)

print("QUADRATIC")
print(np.array(acc3).mean())
cal_visu_conf(preds3, trues3)
