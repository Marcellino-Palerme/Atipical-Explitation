#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:30:02 2021

@author: mpalerme
"""
import numpy as np
from itertools import product

__all__ = ['translation_vector', 'translation', 'center',
           'bounding_box', 'axial_symmetry']

def translation_vector(pointA, pointB):
    """
    Calculate the translation vector between two points

    Parameters
    ----------
    pointA : Float array
        Start point.
    pointB : Float array
        End point.

    Returns
    -------
    array.

    Examples
    --------

    """
    #Verify we have same dimension
    if len(pointA) != len(pointB):
        raise ValueError('Not same dimension')

    trans_vec = []
    for ptA, ptB in zip(pointA, pointB):
        trans_vec.append(ptB - ptA)

    return trans_vec

def translation(points, trans_vec):
    """
    Apply translation vector to all points

    Parameters
    ----------
    points : array
        coordonnates of each points.
    trans_vec : array
        translation vector.

    Returns
    -------
    array : new points.

    Examples
    --------

    """
    points = np.array(points)

    #Verify we have same dimention
    if points.shape[-1] != len(trans_vec):
        raise ValueError("not same dimension")

    return points + np.array(trans_vec)

def center(points):
    """
    Calculate the center of points

    Parameters
    ----------
    points : array
        coordonnates of each points.

    Returns
    -------
    array : coordonate of center.

    Examples
    --------

    """
    points = np.array(points)
    return np.mean(points, axis=0)

def bounding_box(points):
    """
    Give extrem points of bounding box encompass all points

    Parameters
    ----------
    points : array
        coordonnates of each points.

    Returns
    -------
    array : coordonate of each points of bounding box.

    Examples
    --------
    >>> bounding_box([[1,2,3],[0,2,9],[4,-8,10]])
    [[0, -8, 3],
    [0, -8, 10],
    [0, 2, 3],
    [0, 2, 10],
    [4, -8, 3],
    [4, -8, 10],
    [4, 2, 3],
    [4, 2, 10]]

    >>> bounding_box([[1,2],[0,2],[4,-8]])
    [[0, -8], [0, 2], [4, -8], [4, 2]]

    >>> bounding_box([[1,2,7,8],[0,2,7,41],[4,-8,14,7],[7,8,10,0],[5,8,9,1]])
    [[0, -8, 7, 0],
     [0, -8, 7, 41],
     [0, -8, 14, 0],
     [0, -8, 14, 41],
     [0, 8, 7, 0],
     [0, 8, 7, 41],
     [0, 8, 14, 0],
     [0, 8, 14, 41],
     [7, -8, 7, 0],
     [7, -8, 7, 41],
     [7, -8, 14, 0],
     [7, -8, 14, 41],
     [7, 8, 7, 0],
     [7, 8, 7, 41],
     [7, 8, 14, 0],
     [7, 8, 14, 41]]
    """
    minmax = {}
    # Take coordonate min of each dimension
    minmax['0'] = np.amin(points, axis=0)
    # Taka coordonate max of each dimension
    minmax['1'] = np.amax(points, axis=0)
    # Dimension of point
    dimension = len(points[0])
    # define all extrem points of bounding box
    bbox = []
    for point in product('01', repeat=dimension):
        temp = []
        for dim, mnmx in enumerate(point):
            temp.append(minmax[mnmx][dim])
        bbox.append(temp)
    return bbox

def axial_symmetry(pointA, pointB, pointM):
    """
    Calculate coordonate of M' axial symmetry of M around AB

    Parameters
    ----------
    pointA : 2d array
        Coordonate of point A .
    pointB : 2d array
        Coordonate of point B.
    pointM : 2d array
        Coordonate of point M.

    Returns
    -------
    2d array : coordonate of M'.

    Examples
    --------
    >>> axial_symmetry([0,5],[0,6],[1,1])
    [-1, 1]

    >>> axial_symmetry([5,5],[6,6],[1,1])
    [1.0, 1.0]

    >>> axial_symmetry([5,5],[6,6],[0,1])
    [1.0, 0.0]

    >>> axial_symmetry([8,8],[6,6],[0,1])
    [1.0, 0.0]

    """
    pointMr = [0,0]
    # Verify if xA == xB
    if pointA[0] == pointB[0]:
        pointMr[0] = 2*pointA[0] - pointM[0]
        pointMr[1] = pointM[1]
    else:
        # calculate a and b of axis AB where y = ax + b
        a = (pointB[1] - pointA[1]) / (pointB[0] - pointA[0])
        b = pointA[1] - (a * pointA[0])

        # xM'
        pointMr[0] = (((1 - (a * a)) * pointM[0] + (2 * a * pointM[1]) - (2 * a * b)) /
                      (1 + (a * a)))

        # yM'
        pointMr[1] = (((2 * a * pointM[0]) - ((1 - (a * a)) * pointM[1]) + (2 * b)) /
                      (1 + (a * a)))

    return pointMr


def rotate(points, **kargs):
    """


    Parameters
    ----------
    points : TYPE
        DESCRIPTION.
    **kargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # create rotation matrice
    if "degres" in kargs:
        # take rotation angle
        deg = kargs["degres"]
        # convert degres to radians
        rad = np.deg2rad(deg)
        # Create rotation matrice
        rot_mat = [[np.cos(rad), -np.sin(rad)],
                   [np.sin(rad), np.cos(rad)]]
    else:
        rot_mat = kargs["rot_mat"]

    # apply rotation at all points
    for index, coords in enumerate(points):
        points[index] = np.dot(rot_mat, coords)

    return points

