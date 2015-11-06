#!/usr/bin/env python

import os
import sys

import cv2
import numpy as np

# c.f. http://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html

def show(img,name="img"):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)

def save(img,infile,append_name):
    n,ext = os.path.splitext(infile)
    img_name = "{0}_{1}{2}".format(n,append_name,ext)
    cv2.imwrite(img_name,img)
    print("save image: {0}".format(img_name))

def main(infile):
    img = cv2.imread(infile)
    assert img is not None

    gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # THRESH_OTSU: use Otsu's algorithm for optimal threashold value
    ret,th = cv2.threshold(gimg,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    op = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel,iterations=2)
    # sure background area
    sure_bg = cv2.dilate(op,kernel,iterations=3)
    # sure foregronud area
    dist_transform = cv2.distanceTransform(op,cv2.DIST_L2,5)
    ret,sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    #show(sure_bg)
    #show(sure_fg)
    #show(dist_transform)
    #show(unknown)

    # marker labelling
    ret,markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    # mark unknown region 0
    markers[unknown==255] = 0
    # apply watershed
    markers = cv2.watershed(img,markers)
    # boundary region is -1
    img[markers==-1] = [255,0,0]

    show(img)

    save(img,infile,"watershed_seg")

if __name__ == "__main__":
    if len(sys.argv) <= 1:
       print("Usage: {0} <file}".format(sys.argv[0]))
       sys.exit(1)
    main(sys.argv[1])
