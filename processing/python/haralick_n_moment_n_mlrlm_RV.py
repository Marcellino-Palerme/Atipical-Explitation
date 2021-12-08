#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
   Warning: I work with images containing a black background
'''
import argparse
import mahotas as mh
import glob, os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skimage import io
import SimpleITK as sitk
from radiomics.glrlm import RadiomicsGLRLM
import multiprocessing as mp
import copy
import pickle
import tools_file as tsf

##############################################################################
### Constants
##############################################################################
LT_CLASS = ["Alt", "Big", "Mac", "Mil", "Myc", "Pse", "Syl"]
RECTO = "recto"
VERSO = "verso"
TRAIN = "train"
TEST = "test"
SYMPTOM = 'symptom'

# Alt = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Alt_bdb_cut2_max'
# Big = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Big_bdb_cut2_max'
# Mac = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Mac_bdb_cut2_max'
# Mil = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Mil_bdb_cut2_max'
# Myc = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Myc_bdb_cut2_max'
# Pse = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Pse_bdb_cut2_max'
# Syl = '/home/port-mpalerme/Documents/Atipical/Traitement/photos/Syl_bdb_cut2_max'


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

    # Add argument to choose Recto or Recto/Verso
    parser.add_argument('--rv', action='store_true', help="Recto/Verso",
                        dest='rv')

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
    lt_split = sorted(os.listdir(dir_in))

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
                                                          symptom, RECTO,
                                                          '*ecto*.*')))

                # Add images in dataset
                lt_dataset[int(split)][part][RECTO] += lt_images

                if rectoverso:
                    # Take path of all verso images of this symptom
                    lt_images = sorted(glob.glob(os.path.join(dir_in, split,
                                                              part, symptom,
                                                              VERSO,
                                                              '*erso*.*')))

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
    # Define output
    output = np.zeros((16,))
    try:
        # Calculate Gray Level Run Length Matrix (GLRLM) Features
        glrlm = RadiomicsGLRLM(im_itk, mask)
        glrlm._initCalculation()
    except ValueError:
        return output

    functions = [glrlm.getShortRunEmphasisFeatureValue,
                 glrlm.getLongRunEmphasisFeatureValue,
                 glrlm.getGrayLevelNonUniformityFeatureValue,
                 glrlm.getGrayLevelNonUniformityNormalizedFeatureValue,
                 glrlm.getRunLengthNonUniformityFeatureValue,
                 glrlm.getRunLengthNonUniformityNormalizedFeatureValue,
                 glrlm.getRunPercentageFeatureValue,
                 glrlm.getGrayLevelVarianceFeatureValue,
                 glrlm.getRunVarianceFeatureValue,
                 glrlm.getRunEntropyFeatureValue,
                 glrlm.getLowGrayLevelRunEmphasisFeatureValue,
                 glrlm.getHighGrayLevelRunEmphasisFeatureValue,
                 glrlm.getShortRunLowGrayLevelEmphasisFeatureValue,
                 glrlm.getShortRunHighGrayLevelEmphasisFeatureValue,
                 glrlm.getLongRunLowGrayLevelEmphasisFeatureValue,
                 glrlm.getLongRunHighGrayLevelEmphasisFeatureValue
                ]

    # Extract features
    for index, func in enumerate(functions):
        try:
            output[index] = func()[0]
        except ValueError:
            output[index] = 0
  
    output = np.nan_to_num(output, False, nan=0, posinf=np.iinfo('int64').max,
                           neginf=np.iinfo('int64').min)

    return output


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
    print(path_im)
    # Read image
    image = mh.imread(path_im)

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
    im_gray = io.imread(path_im, True, 'matplotlib')
    if len(np.nonzero(im_gray)[0]) > 0:
        try:
            res1=mh.features.haralick(rR,ignore_zeros=True,return_mean=True)
            res1 = np.nan_to_num(res1, False, nan=0,
                                 posinf=np.iinfo('int64').max,
                                 neginf=np.iinfo('int64').min)
        except ValueError:
            pass
        try:
            res2=mh.features.haralick(gR,ignore_zeros=True,return_mean=True)
            res2 = np.nan_to_num(res2, False, nan=0,
                                 posinf=np.iinfo('int64').max,
                                 neginf=np.iinfo('int64').min)
        except ValueError:
            pass
        try:
            res3=mh.features.haralick(bR,ignore_zeros=True,return_mean=True)
            res3 = np.nan_to_num(res3, False, nan=0,
                                 posinf=np.iinfo('int64').max,
                                 neginf=np.iinfo('int64').min)
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


def part_features_dataset(dataset, part, rectoverso):
    """


    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    part : TYPE
        DESCRIPTION.
    rectoverso : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    lt_features = []
    for index in range(len(dataset[part][RECTO])):
        features = extract_features(dataset[part][RECTO][index])

        # if woks with verso
        if rectoverso:
            id_recto = os.path.basename(dataset[part][RECTO][index])[0:8]
            id_verso = os.path.basename(dataset[part][VERSO][index])[0:8]
            # Verify we work with same leaf
            if id_recto != id_verso:
                raise Exception('Not same id, recto: ' + id_recto +
                                ' verso: ' + id_verso)
            features_verso = extract_features(dataset[part][VERSO][index])

            # concatenate features of image
            features = np.concatenate((features, features_verso))

        lt_features.append(features)

    return lt_features


def features_dataset(lt_dataset, rectoverso):
    """


    Parameters
    ----------
    lt_dataset : list of dictionnary
        list of dataset for a split.

    rectoverso : bool
        work with verso.

    Returns
    -------
    list of array.

    """
    ###########################################################################
    ### Extract features of all images
    ###########################################################################
    feat_image = {}
    for part in [TRAIN,TEST]:
        # Take list name
        lt_name = [os.path.basename(name)[0:8]\
                   for name in lt_dataset[0][part][RECTO]]

        # Extract feature of all images
        lt_features = part_features_dataset(lt_dataset[0], part, rectoverso)
        # Link image name and it features
        feat_image.update({os.path.basename(name):features\
                           for name, features in zip(lt_name, lt_features)})

    ###########################################################################
    ### Create dataset with only features of each image
    ###########################################################################
    lt_features = []
    for dataset in lt_dataset:
        train = [feat_image[os.path.basename(name)[0:8]]\
                 for name in dataset[TRAIN][RECTO]]
        test = [feat_image[os.path.basename(name)[0:8]]\
                for name in dataset[TEST][RECTO]]
        lt_features.append({TRAIN:train, TEST:test})

    return lt_features


def all_fit(classifier, lt_features, lt_dataset):
    """


    Parameters
    ----------
    classifier : TYPE
        DESCRIPTION.
    lt_features : TYPE
        DESCRIPTION.
    lt_dataset : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    history = {'classifier':[], 'predict':[], 'score': [], 'true':[]}
    for index, feature in enumerate(lt_features):
        # create a copy of classifier
        in_classifier = copy.deepcopy(classifier)
        # fit classifier
        in_classifier.fit(feature[TRAIN], lt_dataset[index][TRAIN][SYMPTOM])
        # Save classifier
        history['classifier'].append(in_classifier)
        # Save predict
        history['predict'].append(in_classifier.predict(feature[TEST]))
        # Save true
        history['true'].append(lt_dataset[index][TEST][SYMPTOM])
        # Save score
        history['score'].append(in_classifier.score(feature[TEST],
                                                    lt_dataset[index][TEST][SYMPTOM]))

    return history


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
    lt_features = features_dataset(lt_dataset, args.rv)

    # Define all classifier
    lt_classifier = [["linear_auto",
                      LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')],
                     ["linear_none",
                      LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)],
                     ["quatra",
                      QuadraticDiscriminantAnalysis()],
                     ["svc", SVC()],
                     ["dtree", DecisionTreeClassifier(random_state=159)]]

    pool = mp.Pool(processes=len(lt_classifier))
    results = []
    out_fits = {}
    for name, classifier in lt_classifier:
        out_fits[name] = []
        results.append(pool.apply_async(all_fit, (classifier,
                                                  lt_features,
                                                  lt_dataset),
                                        callback=out_fits[name].append))

    for result in results:
        result.wait()

    # Take absolu path of input directory
    abs_dir_out = os.path.abspath(os.path.expanduser(args.dir_out))

    # Create out directory
    tsf.create_directory(abs_dir_out)

    # Save data
    with open(os.path.join(abs_dir_out, 'save'), 'wb') as fsave :
        pickle.dump(out_fits, fsave)

    # print result
    for name, classifier in lt_classifier:
        print("Classifier: " + name)
        print("score: " + str(np.mean(out_fits[name][0]['score'])))

if __name__=='__main__':
    run()
