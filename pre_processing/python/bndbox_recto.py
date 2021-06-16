#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 08:48:35 2021

@author: mpalerme
"""
from tkinter.filedialog import askdirectory
from tools_file import file_list_ext, file_list
from defusedxml.ElementTree import parse
from os.path import join, basename, splitext
from skimage import io, feature, filters, color, transform
import numpy as np
from geometry import translation, bounding_box, rotate
from functools import partial
from pycpd import affine_registration
import xml.etree.cElementTree as ET

RECTO = 'Recto'
VERSO = 'Verso'

def extract_edge(path_img, scale, flip="ud"):
    """
    Extract edge of images

    Parameters
    ----------
    path_img : str
        path of image.
    scale : float
        scale.
    flip : boolean
        flip horizontal the image

    Returns
    -------
    array : coordonate of each point of edge.

    """
    # img_h= io.imread(path_img, True,'pil')
    img = io.imread(path_img)
    # io.imshow(img_h)
    # Keep hue of image
    img_h = color.rgb2hsv(img)[:,:,0]
    # Calcule Otsu thresold
    tso = filters.threshold_otsu(img_h)
    # rescale image
    img_h = transform.rescale(img_h, scale)
    # Transform image to have black foreground and white background
    img_h[img_h<tso] = 0
    img_h[img_h>=tso] = 1

    # flip image
    if(flip == "ud"):
        img_h = np.flipud(img_h)
    else:
        img_h = np.fliplr(img_h)

    # Extract edge
    edge = feature.canny(img_h,3)
    # Get coordonate of edge
    nonzero = np.nonzero(edge)
    xnonzero = np.reshape(nonzero[0], (-1,1))
    ynonzero = np.reshape(nonzero[1], (-1,1))
    return np.concatenate((xnonzero, ynonzero), axis=1)


# Callback function
# Function to see the progress of the registration
def trace(iteration, error,X,Y):
    print('Iteration: {:d}\nError: {:06.8f}'.format(iteration, error))


def create_xml(directory, name_img, bnd_box, class_name):
    """
    Create xml for define bounding_box of image

    Parameters
    ----------
    directory : TYPE
        DESCRIPTION.
    name_img : TYPE
        DESCRIPTION.
    bnd_box : TYPE
        DESCRIPTION.
    class_name : str
        DESCRIPTION

    Returns
    -------
    None.

    """
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = basename(directory)
    ET.SubElement(root, "filename").text = name_img
    ET.SubElement(root, "path").text = join(directory, name_img)
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    img = io.imread(join(directory, name_img))
    ET.SubElement(size, "width").text = str(img.shape[1])
    ET.SubElement(size, "height").text = str(img.shape[0])
    ET.SubElement(size, "depth").text = str(img.shape[2])
    ET.SubElement(root, "segmented").text = "0"
    object_ = ET.SubElement(root, "object")
    ET.SubElement(object_, "name").text = class_name
    ET.SubElement(object_, "pose").text = "Unspecified"
    ET.SubElement(object_, "truncated").text = "0"
    ET.SubElement(object_, "difficult").text = "0"
    bndbox = ET.SubElement(object_, "bndbox")
    xymin = np.amin(bnd_box, axis=0)
    xymax = np.amax(bnd_box, axis=0)
    ET.SubElement(bndbox, "xmin").text = str(xymin[1])
    ET.SubElement(bndbox, "ymin").text = str(xymin[0])
    ET.SubElement(bndbox, "xmax").text = str(xymax[1])
    ET.SubElement(bndbox, "ymax").text = str(xymax[0])

    tree = ET.ElementTree(root)
    xml_name = splitext(name_img)[0] + ".xml"
    tree.write(join(directory, xml_name))



def run():
    directory = askdirectory()
    # Take all xml files
    xml_files = file_list_ext(directory, 'xml')
    # Take all files
    all_files = file_list(directory)

    # Follow the progress of the rigid registration function
    callback = partial(trace)

    # For each recto xml we create verso xml
    # Verso xml containt the bounding box for verso image
    for xml_file in xml_files:
        # parse the xml
        xml = parse(join(directory, xml_file))
        for tag in xml.findall('path'):
            # Take recto image name
            recto_img_name = basename(tag.text)
        # Verify recto image exits
        if not (recto_img_name in all_files):
            continue
        # Create recto image name
        verso_img_name = recto_img_name.replace(RECTO, VERSO, 1)
        # Verify verso image exits
        if not (verso_img_name in all_files):
            continue

        # Verify xml for verso image not exist
        img_name = splitext(verso_img_name)[0]
        if img_name + ".xml" in all_files:
            continue

        # define multi start
        multi_start = {"ud_0":None, "ud_90":None, "ud_180":None, "ud_270":None,
                       "lr_0":None, "lr_90":None, "lr_180":None, "lr_270":None}

        for key in multi_start:
            # Take kind flip and angle rotation
            flip, angle = key.split("_")

            # Extract recto edge coodonate
            recto_coord = extract_edge(join(directory, recto_img_name),0.1, flip)
            # Extract verso edge coodonate
            verso_coord = extract_edge(join(directory, verso_img_name),0.1)

            recto_coord = rotate(recto_coord,degres=int(angle))

            reg = affine_registration(**{'X': verso_coord,
                                        'Y': recto_coord,
                                        'max_iterations':200, 'tolerance':0.0001})
            reg.register(callback)
            # save result
            multi_start[key] = reg

            # visualize(recto_coord_reduc, verso_coord_reduc, res[0])

        # Take the start with the minner error
        key_min = min(multi_start, key=lambda key : multi_start[key].q)
        reg = multi_start[key_min]
        # Take kind flip and angle rotation
        flip, angle = key_min.split("_")
        # Get extremum of bounding box
        dir_extrem = {}
        for extrem in ['xmin', 'ymin', 'xmax', 'ymax']:
            for tag in xml.findall('object/bndbox/' + extrem):
                dir_extrem[extrem] = int(tag.text)

        # Recreate bounding box
        bnd_box = bounding_box([[dir_extrem['xmin'], dir_extrem['ymin']],
                                [dir_extrem['xmax'], dir_extrem['ymax']]])

        # take image recto
        img_recto = io.imread(join(directory, recto_img_name), True)
        # put image at zero
        img_recto[img_recto>0] = 0
        # put bounding box
        img_recto[dir_extrem['ymin']:dir_extrem['ymax'],
                  dir_extrem['xmin']:dir_extrem['xmax']] = 1

        # rescale image
        img_recto = transform.rescale(img_recto, 0.1)
        img_recto[img_recto>0] = 1

        # flip bounding box
        if(flip == "ud"):
            img_recto = np.flipud(img_recto)
        else:
            img_recto = np.fliplr(img_recto)

        # Get new coordonates of bounding box
        nonzero = np.nonzero(img_recto)
        xnonzero = np.reshape(nonzero[0], (-1,1))
        ynonzero = np.reshape(nonzero[1], (-1,1))
        bnd_box = np.concatenate((xnonzero, ynonzero), axis=1)

        bnd_box = rotate(bnd_box, degres=int(angle))
        # Rotate the bounding box
        bnd_box = rotate(bnd_box, rot_mat=reg.B)
        # bnd_box = rotate(bnd_box, rot_mat=reg.R)

        # Translate the bounding box
        bnd_box = translation(bnd_box, np.array(reg.t))

        # put image at zero
        img_recto[img_recto>0] = 0
        # rescale image
        #img_recto = transform.rescale(img_recto, 10)

        img_recto = np.zeros((1000,1000))

        # put bounding box on image
        for coord in bnd_box:
            if coord[0]>0 and coord[1]>0 :
                img_recto[int(np.around(coord[0])),
                          int(np.around(coord[1]))] = 1

        # rescale image
        img_recto = transform.rescale(img_recto, 10)

        # Get new coordonates of bounding box
        nonzero = np.nonzero(img_recto)
        if len(nonzero[0]) == 0:
            continue
        xnonzero = np.reshape(nonzero[0], (-1,1))
        ynonzero = np.reshape(nonzero[1], (-1,1))
        bnd_box = np.concatenate((xnonzero, ynonzero), axis=1)

        # Create bounding box of bounding box
        bnd_box = bounding_box(bnd_box)

        create_xml(directory, verso_img_name, bnd_box, "Alt")

if __name__ == "__main__":
    run()
