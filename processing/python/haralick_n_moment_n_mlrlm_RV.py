#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
   Warning: I work with images containing a black background
'''
import argparse
import mahotas as mh
import glob, os
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from skimage import io


import SimpleITK as sitk
from radiomics.glrlm import RadiomicsGLRLM

import json

##############################################################################
### Constants
##############################################################################
LT_CLASS = ["Alt", "Big", "Mac", "Mil", "Myc", "Pse", "Syl"]
RECTO = "recto"
VERSO = "verso"
TRAIN = "train"
TEST = "test"
SYMPTOM = 'symptom'

Alt = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Alt_bdb_cut2_max'
Big = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Big_bdb_cut2_max'
Mac = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Mac_bdb_cut2_max'
Mil = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Mil_bdb_cut2_max'
Myc = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Myc_bdb_cut2_max'
Pse = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Pse_bdb_cut2_max'
Syl = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Syl_bdb_cut2_max'


###############################################################################
### Manage arguments input
###############################################################################
def arguments ():
    """
    manage input arguments

    Returns
    -------
    namespace

    """
    parser = argparse.ArgumentParser()

    # Add argument for source directory
    parser.add_argument('-i', '--dir_in', type=str, help="source directory",
                        required=True, dest='dir_in')

    # Add argument for output directory
    parser.add_argument('-o', '--dir_out', type=str, help="output directory",
                        required=True, dest='dir_out')

    # Take all arguments
    return parser.parse_args()


##############################################################################
### Additional function
##############################################################################
def create_dataset (dir_in, rectoverso):
    """
    From directory recreate dataset's list

    Parameters
    ----------
    dir_in : str
        input directory where there are splits with images.

    rectoverso : bool
        work with verso.

    Returns
    -------
    list of dictonary.

    """
    # Create list of dataset
    lt_dataset = []

    # Take all split
    lt_split = os.listdir(dir_in)

    for split in lt_split:
        # Add a dataset
        lt_dataset.append({})
        for part in [TRAIN, TEST]:
            # Add dataset's part
            lt_dataset[int(split)][part] = {RECTO:[], SYMPTOM:[]}
            if rectoverso:
                # Add verso
                lt_dataset[int(split)][part][VERSO] = []

            for index_sympt, symptom in enumerate(LT_CLASS):
                # Take path of all recto images of this symptom
                lt_images = sorted(glob.glob(os.path.join(dir_in, split, part,
                                                          symptom, RECTO)))

                # Add images in dataset
                lt_dataset[int(split)][part][RECTO] += lt_images

                if rectoverso:
                    # Take path of all verso images of this symptom
                    lt_images = sorted(glob.glob(os.path.join(dir_in, split,
                                                              part, symptom,
                                                              VERSO)))

                    # Add images in dataset
                    lt_dataset[int(split)][part][VERSO] += lt_images

                # Add index of symptom
                lt_dataset[int(split)][part][SYMPTOM] += list(np.ones(len(lt_images),
                                                                      np.int8) *
                                                              index_sympt)

    return lt_dataset


def lrlm(in_im, in_mask):
    """
    Parameters
    ----------
    in_im : numpy array
        DESCRIPTION.
    in_mask : numpy array
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Transform numpy  image to simpleITK  image
    im_itk = sitk.GetImageFromArray(in_im)
    mask = sitk.GetImageFromArray(in_mask)
    # Calculate Gray Level Run Length Matrix (GLRLM) Features
    glrlm = RadiomicsGLRLM(im_itk, mask)
    glrlm._initCalculation()
    # Extract features
    return [glrlm.getShortRunEmphasisFeatureValue()[0],
            glrlm.getLongRunEmphasisFeatureValue()[0],
            glrlm.getGrayLevelNonUniformityFeatureValue()[0],
            glrlm.getGrayLevelNonUniformityNormalizedFeatureValue()[0],
            glrlm.getRunLengthNonUniformityFeatureValue()[0],
            glrlm.getRunLengthNonUniformityNormalizedFeatureValue()[0],
            glrlm.getRunPercentageFeatureValue()[0],
            glrlm.getGrayLevelVarianceFeatureValue()[0],
            glrlm.getRunVarianceFeatureValue()[0],
            glrlm.getRunEntropyFeatureValue()[0],
            glrlm.getLowGrayLevelRunEmphasisFeatureValue()[0],
            glrlm.getHighGrayLevelRunEmphasisFeatureValue()[0],
            glrlm.getShortRunLowGrayLevelEmphasisFeatureValue()[0],
            glrlm.getShortRunHighGrayLevelEmphasisFeatureValue()[0],
            glrlm.getLongRunLowGrayLevelEmphasisFeatureValue()[0],
            glrlm.getLongRunHighGrayLevelEmphasisFeatureValue()[0]
           ]


def extract_features(path_im):
    """
    extract all feature of image

    Parameters
    ----------
    path_im : str
        path of image.

    Returns
    -------
    list of features.

    """

    # Read image
    image = mh.imread(file)

    # Extract RGB channels
    rR=image[:,:,0] # r
    gR=image[:,:,1] # g
    bR=image[:,:,2] # b

    # inialize feature's lists
    res1 = np.zeros((13,))
    res2 = np.zeros((13,))
    res3 = np.zeros((13,))
    a_glrlm = np.zeros((16,))
    a_rlrlm = np.zeros((16,))
    a_vlrlm = np.zeros((16,))
    a_blrlm = np.zeros((16,))
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

        # Create mask
        mask = im_gray.copy()
        mask[mask > 0] = 1

        # Calculate Gray Level Run Length Matrix (GLRLM) Features
        a_glrlm = lrlm(im_gray, mask)
        a_rlrlm = lrlm(rR, mask)
        a_vlrlm = lrlm(gR, mask)
        a_blrlm = lrlm(bR, mask)

        rR = rR[np.nonzero(im_gray)]
        gR = gR[np.nonzero(im_gray)]
        bR = bR[np.nonzero(im_gray)]

    # define moment mean, standar deviation, variance, min, max
    momt_R = [np.mean(rR), np.std(rR), np.var(rR), np.min(rR), np.max(rR)]
    momt_G = [np.mean(gR), np.std(gR), np.var(gR), np.min(gR), np.max(gR)]
    momt_B = [np.mean(bR), np.std(bR), np.var(bR), np.min(bR), np.max(bR)]

    return np.concatenate((res1, res2, res3, momt_R, momt_G, momt_B,
                           a_glrlm, a_rlrlm, a_vlrlm, a_blrlm))


def features_dataset(dataset, rectoverso):
    """


    Parameters
    ----------
    dataset : dictionnary
        dataset for a split.

    rectoverso : bool
        work with verso.

    Returns
    -------
    list of array.

    """
    lt_features = []
    for index in range(len(dataset[RECTO])):
        features = extract_features(dataset[RECTO][index])

        # if woks with verso
        if rectoverso:
            features_verso = extract_features(dataset[VERSO][index])

            # concatenate features of image
            features = np.concatenate((features, features_verso))

        lt_features.append(features)

    return lt_features

y=np.array([])
X=[]
u=1.0
id_im = []
for where in [Alt, Big, Mac, Mil, Myc, Pse, Syl]:
    ones=[]
    os.chdir(where)
    k=1.0
    for file in sorted(glob.glob("*ecto*.tiff")):
        print(k/len(glob.glob("*ecto*.tiff")))

        # Take Id of image
        id_im.append(file[0:8])

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
        a_rlrlm = np.zeros((16,))
        a_vlrlm = np.zeros((16,))
        a_blrlm = np.zeros((16,))
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

            # Create mask
            mask = im_gray.copy()
            mask[mask > 0] = 1

            # Calculate Gray Level Run Length Matrix (GLRLM) Features
            a_glrlm = lrlm(im_gray, mask)
            a_rlrlm = lrlm(rR, mask)
            a_vlrlm = lrlm(gR, mask)
            a_blrlm = lrlm(bR, mask)

            rR = rR[np.nonzero(im_gray)]
            gR = gR[np.nonzero(im_gray)]
            bR = bR[np.nonzero(im_gray)]

        # define moment mean, standar deviation, variance, min, max
        momt_R = [np.mean(rR), np.std(rR), np.var(rR), np.min(rR), np.max(rR)]
        momt_G = [np.mean(gR), np.std(gR), np.var(gR), np.min(gR), np.max(gR)]
        momt_B = [np.mean(bR), np.std(bR), np.var(bR), np.min(bR), np.max(bR)]
        res=np.concatenate((res1, res2, res3,
                            momt_R, momt_G, momt_B,
                            a_glrlm, a_rlrlm, a_vlrlm, a_blrlm))
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
        a_rlrlm = np.zeros((16,))
        a_vlrlm = np.zeros((16,))
        a_blrlm = np.zeros((16,))
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

            # Create mask
            mask = im_gray.copy()
            mask[mask > 0] = 1

           # Calculate Gray Level Run Length Matrix (GLRLM) Features
            a_glrlm = lrlm(im_gray, mask)
            a_rlrlm = lrlm(rR, mask)
            a_vlrlm = lrlm(gR, mask)
            a_blrlm = lrlm(bR, mask)

            rR = rR[np.nonzero(im_gray)]
            gR = gR[np.nonzero(im_gray)]
            bR = bR[np.nonzero(im_gray)]
        # define moment mean, standar deviation, variance, min, max
        momt_R = [np.mean(rR), np.std(rR), np.var(rR), np.min(rR), np.max(rR)]
        momt_G = [np.mean(gR), np.std(gR), np.var(gR), np.min(gR), np.max(gR)]
        momt_B = [np.mean(bR), np.std(bR), np.var(bR), np.min(bR), np.max(bR)]
        res=np.concatenate((res1, res2, res3,
                            momt_R, momt_G, momt_B,
                            a_glrlm, a_rlrlm, a_vlrlm, a_blrlm))
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
Xb = np.nan_to_num(Xb)
#yb=y[idx]
yb=y


kf = StratifiedShuffleSplit(random_state=159)

acc1=[]
acc2=[]
acc3=[]
preds1=[]
trues1=[]
preds2=[]
trues2=[]
preds3=[]
trues3=[]
the_dict = {}
for index, my_split in enumerate(kf.split(Xb, yb)):
    train = my_split[0]
    test = my_split[1]
    print("Train : ", train)
    print('test : ', test)
    X_train, X_test, y_train, y_test = Xb[train], Xb[test], yb[train], yb[test]
    id_train, id_test = id_im[train], id_im[test]
    clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X_train, y_train)
    clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X_train, y_train)
    clf3 = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
    cl_svc = SVC()
    acc1.append(clf1.score(X_test,y_test))
    acc2.append(clf2.score(X_test,y_test))
    acc3.append(clf3.score(X_test,y_test))
    preds1.append(clf1.predict(X_test))
    trues1.append(y_test)
    preds2.append(clf2.predict(X_test))
    trues2.append(y_test)
    preds3.append(clf3.predict(X_test))
    trues3.append(y_test)
    the_dict["l_auto_acc_" + str(index)] = clf1.score(X_test,y_test)
    the_dict["l_none_acc_" + str(index)] = clf2.score(X_test,y_test)
    the_dict["quadra_acc_" + str(index)] = clf3.score(X_test,y_test)
    the_dict["l_auto_pred_" + str(index)] = clf1.predict(X_test)
    the_dict["l_none_pred_" + str(index)] = clf2.predict(X_test)
    the_dict["quadra_pred_" + str(index)] = clf3.predict(X_test)
    the_dict["true_" + str(index)] = y_test


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
#cal_visu_conf(preds1, trues1)

print("LINEAR")
print("Acc = " + str(np.array(acc2).mean()))
print("Kappa = " + str(cohen_kappa_score(preds2, trues2)))
#cal_visu_conf(preds2, trues2)

print("QUADRATIC")
print("Acc = " + str(np.array(acc3).mean()))
print("Kappa = " + str(cohen_kappa_score(preds3, trues3)))
#cal_visu_conf(preds3, trues3)

# Save dictionary
DICT_FILE = "pred_true.json"
with open(DICT_FILE, 'w') as file:
    json.dump(the_dict, file)

def run():
    """


    Returns
    -------
    None.

    """
    # Take input arguments
    args =  arguments()

    # Take absolu path of input directory
    abs_dir_in = os.path.abspath(os.path.expanduser(args.dir_in))

    # Take dataset
    lt_dataset = create_dataset(abs_dir_in, args.rv)

    # scan dataset
    for dataset in lt_dataset:
        lt_features = features_dataset(dataset, args.rv)


if __name__=='__main__':
    run()
