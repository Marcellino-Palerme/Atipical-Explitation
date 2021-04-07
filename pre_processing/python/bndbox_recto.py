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
from skimage import io, feature, filters, color
import numpy as np
from geometry import center, axial_symmetry, translation_vector, translation,\
                     bounding_box
from functools import partial
from pycpd import AffineRegistration
import xml.etree.cElementTree as ET

EXT_IMG = 'bmp'
RECTO = 'Recto'
VERSO = 'Verso'

def visualize(points):
    """


    Parameters
    ----------
    points : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    points = np.rint(points)
    print(points)
    shape = np.amax(points, axis=0).astype(int) + 10
    print(shape)
    img = np.zeros(shape)
    for x, y in points:
        img[int(x),int(y)] = 1

    io.imshow(img)

def extract_edge(path_img):
    """
    Extract edge of images

    Parameters
    ----------
    path_img : str
        path of image.

    Returns
    -------
    array : coordonate of each point of edge.

    """
    img = io.imread(path_img)

    # Keep hue of image
    img_h = color.rgb2hsv(img)[:,:,0]
    # Calcule Otsu thresold
    tso = filters.threshold_otsu(img_h)
    # Transform image to have black foreground and white background
    img_h[img_h<tso] = 0
    img_h[img_h>=tso] = 1
    # Extract edge
    edge = feature.canny(img_h)
    # Get coordonate of edge
    nonzero = np.nonzero(edge)
    xnonzero = np.reshape(nonzero[0], (-1,1))
    ynonzero = np.reshape(nonzero[1], (-1,1))
    return np.concatenate((xnonzero, ynonzero), axis=1)


# Callback function
# Function to see the progress of the registration
def trace(iteration, error,X,Y):
    print('Iteration: {:d}\nError: {:06.4f}'.format(iteration, error))


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
    ET.SubElement(size, "width").text = str(img.shape[0])
    ET.SubElement(size, "height").text = str(img.shape[1])
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
    ET.SubElement(bndbox, "xmin").text = str(xymin[0])
    ET.SubElement(bndbox, "ymin").text = str(xymin[1])
    ET.SubElement(bndbox, "xmax").text = str(xymax[0])
    ET.SubElement(bndbox, "ymax").text = str(xymax[1])

    tree = ET.ElementTree(root)
    xml_name = splitext(img_name)[0] + ".xml"
    tree.write(join(directory, xml_name))



if __name__ == "__main__":
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

        # Extract recto edge coodonate
        recto_coord = extract_edge(join(directory, recto_img_name))
        # Extract verso edge coodonate
        verso_coord = extract_edge(join(directory, verso_img_name))

        # Extract center of recto
        recto_center = center(recto_coord)
        print(recto_center)
        # Extract center of verso
        verso_center = center(verso_coord)

        # Flip recto image about y axis of center
        for index, coords in enumerate(recto_coord):
            recto_coord[index] = axial_symmetry(recto_center,
                                                [recto_center[0] + 10,
                                                 recto_center[1]],
                                                coords)

        # Extract translation vector
        trans_vec = translation_vector(recto_center, verso_center)
        # Apply the translation to recto
        recto_coord = translation(recto_coord, trans_vec)

        reg = AffineRegistration(**{'X': verso_coord, 'Y': recto_coord,
                                    'max_iterations':200, 'tolerance':0.0001})
        res = reg.register(callback)

        # Get extremum of bounding box
        dir_extrem = {}
        for extrem in ['xmin', 'ymin', 'xmax', 'ymax']:
            for tag in xml.findall('object/bndbox/' + extrem):
                dir_extrem[extrem] = int(tag.text)

        # Recreate bounding box
        bnd_box = bounding_box([[dir_extrem['xmin'], dir_extrem['ymin']],
                                [dir_extrem['xmax'], dir_extrem['ymax']]])

        # Flip recto bounding box about y axis of recto center
        for index, coords in enumerate(bnd_box):
            bnd_box[index] = axial_symmetry(recto_center,
                                            [recto_center[0] + 10,
                                             recto_center[1]],
                                            coords)

        # Translate the bounding box
        bnd_box = translation(bnd_box, trans_vec)

        # Rotate the bounding box
        for index, coords in enumerate(bnd_box):
            bnd_box[index] = np.dot(res[1][0], coords)

        # Create bounding box of bounding box
        bnd_box = bounding_box(bnd_box)

        create_xml(directory, verso_img_name, bnd_box, "Alt")
