#!/usr/bin/env python

import os
import sys

import numpy as np
import cv2
import cv2.cv as cv

def show(img,name="img"):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)

def main(infile):
    img = cv2.imread(infile)
    assert img is not None
    # smoothing
    img = cv2.medianBlur(img,5)

    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gimg, cv.CV_HOUGH_GRADIENT,1,50,
                                param1=50,param2=30,minRadius=0,maxRadius=200)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

    show(img)
    n,ext = os.path.splitext(infile)
    img_name = "{0}_{1}{2}".format(n,"hough_circle",ext)
    cv2.imwrite(img_name,img)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
       print("Usage: {0} <file}".format(sys.argv[0]))
       sys.exit(1)
    main(sys.argv[1])
