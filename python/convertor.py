
# -*- coding:utf-8 -*-

import logging
import os
import numpy
import math
import sys
import unittest
import json
import inspect
import time
import timeit

# FTS格式
# --------------------
# face:

# rect:
# 500,466,1037,1002

# face106:
# 500.79984,553.0546
# 504.12244,588.2755
# 508.37396,623.6479
# .... 106条
#
# faceext:
# 612.8252,593.6181
# 622.48413,592.6478
# 632.28735,591.66846
# .... 134条
#
# eyeballcenter:
# 650.07587,579.62305
# 917.5341,555.63916
#
# eyeballcontour:
# 666.9835,581.6166
# 665.6209,587.2577
# .... 38条
# --------------------
class FTS2Xml(object):
    def __init__(self, imgDir, ftsDir, xmlPath):
        self.data = {'imgDir':imgDir, 'ftsDir':ftsDir, 'xmlPath':xmlPath}

    def MainProc(self):
        ftsDir = self.data['ftsDir']
        logging.debug(ftsDir)
        for root, dirs, files in os.walk(ftsDir):
            for name in files:
                ftsDir = os.path.join(root, name)
                logging.debug(ftsDir)
                if not name.endswith('.fts'):
                    logging.warn('abandon file : %s' % ftsDir)
                    continue

                imgName = name[:-4]
                imgPath = os.path.join(self.data['imgDir'], imgName)
                logging.debug('ftsDir[%s] imgPath[%s]' % (ftsDir, imgPath))
                return
                

class ConvertUT(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

    def Fts2xml(self):
        fpsDir = '/Users/palance/Downloads/face data sets/ibug_300W_large_face_landmark_dataset/helen/annotation240'
        imgDir = '/Users/palance/Downloads/face data sets/ibug_300W_large_face_landmark_dataset/helen/trainset'
        xmlPath = '/Users/palance/Downloads/face data sets/ibug_300W_large_face_landmark_dataset/helen/train.xml'
        f5s2xml = FTS2Xml(imgDir, fpsDir, xmlPath)
        f5s2xml.MainProc()

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main
    # cmd: python -m unittest mpcap.CapUT.test01
