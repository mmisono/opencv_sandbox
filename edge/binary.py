#!/usr/bin/env python

import cv2
import sys
import os

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

    # show(th)
    save(th,infile,"binary")

if __name__ == "__main__":
    if len(sys.argv) <= 1:
       print("Usage: {0} <file}".format(sys.argv[0]))
       sys.exit(1)
    main(sys.argv[1])
