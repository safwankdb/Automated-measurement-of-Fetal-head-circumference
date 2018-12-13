import cv2
import os

edge_annotations = [f for f in os.listdir(
    './training_set/') if 'Annotation' in f]

for pic in edge_annotations:
    im = cv2.imread('./training_set/' + pic)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ellipse = cv2.fitEllipse(contours[0])
    im1 = cv2.ellipse(im, ellipse, (255, 255, 255), -1)
    cv2.imwrite('./training_set/' + pic.replace('Annotation', 'Mask'), im1)
