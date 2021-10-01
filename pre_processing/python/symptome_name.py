#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:51:58 2021

@author: mpalerme

Modify symptome name of verso 
"""

from tkinter.filedialog import askdirectory
from defusedxml.ElementTree import parse
from tools_file import file_list_ext, file_list
from os.path import join

RECTO = 'Recto'
VERSO = 'Verso'

def run():
    directory = askdirectory()
    # Take all xml files
    xml_files = file_list_ext(directory, 'xml')
    # Take all files
    all_files = file_list(directory)

    for xml_file in xml_files:
        
        # Work only with recto
        if VERSO in xml_file:
            continue

        # parse the recto xml
        recto_xml = parse(join(directory, xml_file))

        
        verso_xml_name = xml_file.replace(RECTO, VERSO, 1)
        # Verify verso xml exits
        if not (verso_xml_name in all_files):
            continue
        
        # parse the verso wml
        verso_xml = parse(join(directory, verso_xml_name))
        
        # Get name of symptom on recto        
        recto_root = recto_xml.getroot()
        recto_symptom_name = recto_root.find("object/name")
        
        # Modify name of symptom on verso
        verso_root = verso_xml.getroot()
        verso_symptom_name = verso_root.find("object/name")
        verso_symptom_name.text = recto_symptom_name.text
        
        # Save modification verso xml
        verso_xml.write(join(directory, verso_xml_name))


if __name__ == "__main__":
    run()