#!/usr/bin/env python

import cv2
import sys
import os

def sobelX(img,ksize=5):
    return cv2.Sobel(img,cv2.CV_32F,1,0,ksize=ksize)

def sobelY(img,ksize=5):
    return cv2.Sobel(img,cv2.CV_32F,0,1,ksize=ksize)

def canny(img,th1=50,th2=150):
    return cv2.Canny(img,th1,th2)

def lap(img,ksize=5):
    return cv2.Laplacian(img,cv2.CV_32F,ksize=ksize)

def show(img,name="img"):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)

def main(infile):
    img = cv2.imread(infile)
    assert img is not None

    for f in (sobelX,sobelY,lap,canny):
        fname = f.__name__
        n,ext = os.path.splitext(infile)
        img_name = "{0}_{1}{2}".format(n,fname,ext)
        filtered = f(img)
        cv2.imwrite(img_name,filtered)
        print("save image: {0}".format(img_name))
        # show(img,fname)

if __name__ == "__main__":
    if len(sys.argv) <= 1:
       print("Usage: {0} <file}".format(sys.argv[0]))
       sys.exit(1)
    main(sys.argv[1])
