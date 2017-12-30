
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
import xml.sax
import cv2
import subprocess
from subprocess import Popen, PIPE

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

class FTSParser(object):
    def __init__(self, ftsPath):
        self.currTag = None
        self.ftsData = []
        self.load(ftsPath)

    def handleTag(self, line):
        if line.startswith('face:'):
            self.ftsData.append({'rect':[], 'face106':[], 'faceext':[], 'eyeballcenter':[], 'eyeballcontour':[]})
            return 'face:'
        if len(self.ftsData) <= 0:
            raise Exception('no data constructed for num')

        lastFTSData = self.ftsData[-1]
        for k in lastFTSData.keys():
            if line.startswith(k):
                self.currTag = k
                return k
        return None

    def handleNums(self, line):
        data = line.split(',')
        key = self.currTag
        value = None
        if len(self.ftsData) <= 0:
            raise Exception('no data constructed for num')

        lastFTSData = self.ftsData[-1]
        if self.currTag == 'rect' :
            if len(data) != 4:
                raise Exception('line error: %s' % line)
            if len(lastFTSData['rect']) != 0:
                raise Exception('multi rect')

            value = [int(d) for d in data]
            lastFTSData[key] = value
        else:
            if len(data) != 2:
                raise Exception('line error: %s' % line)

            value = [int(float(d)) for d in data]
            lastFTSData[key].append(value)
        return (key, value)

    def load(self, ftsPath):
        with open(ftsPath, 'rb') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                if self.handleTag(line) != None:
                    continue
                try:
                    self.handleNums(line)
                except Exception, e:
                    logging.error(ftsPath)
                    logging.error(e.args)


class FTS2Xml(object):
    # 将FTS转成dlib训练所识别的xml
    def __init__(self, imgDir, ftsDir, xmlPath):
        self.data = {'imgDir':imgDir, 'ftsDir':ftsDir, 'xmlPath':xmlPath}

    def MainProc(self):
        ftsDir = self.data['ftsDir']
        xmlPath = self.data['xmlPath']
        with open(xmlPath, 'w') as xmlFile:
            xmlFile.write('''<?xml version='1.0' encoding='ISO-8859-1'?>
<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
<dataset>
<name></name>
<comment></comment>
<images>
''')
            for root, dirs, files in os.walk(ftsDir):
                for name in files:
                    ftsPath = os.path.join(root, name)
                    if not name.endswith('.fts'):
                        logging.warn('abandon file : %s' % ftsPath)
                        continue

                    imgName = name[:-4]
                    imgPath = os.path.join(self.data['imgDir'], imgName)
                    if not os.path.exists(imgPath):
                        logging.warn('img not found ftsPath[%s] imgPath[%s]' % (ftsPath, imgPath))
                        continue

                    ftsParser = FTSParser(ftsPath)
                    xmlFile.write("  <image file='%s'>\n" % imgPath)
                    for ftsData in ftsParser.ftsData:
                        left = ftsData['rect'][0]
                        top = ftsData['rect'][1]
                        width = ftsData['rect'][2] - left
                        height = ftsData['rect'][3] - top
                        xmlFile.write("    <box top='%d' left='%d' width='%d' height='%d'>\n" % (left, top, width, height))

                        faceIdx = 0
                        for facebase in ftsData['face106']:
                            xmlFile.write("      <part name='%03d' x='%d' y='%d'/>\n" % (faceIdx, facebase[0], facebase[1]))
                            faceIdx += 1
                            
                        for faceext in ftsData['faceext']:
                            xmlFile.write("      <part name='%03d' x='%d' y='%d'/>\n" % (faceIdx, faceext[0], faceext[1]))
                            faceIdx += 1
                            
                        for eyeballcenter in ftsData['eyeballcenter']:
                            xmlFile.write("      <part name='%03d' x='%d' y='%d'/>\n" % (faceIdx, eyeballcenter[0], eyeballcenter[1]))
                            faceIdx += 1
                            
                        for eyeballcontour in ftsData['eyeballcontour']:
                            xmlFile.write("      <part name='%03d' x='%d' y='%d'/>\n" % (faceIdx, eyeballcontour[0], eyeballcontour[1]))
                            faceIdx += 1
                        xmlFile.write("    </box>\n")
                    xmlFile.write("  </image>\n")
            xmlFile.write('''
</images>
</dataset>
''')


class XmlHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.mImages = []
        self.mCurrImage = None
        self.mCurrBox = None
        self.mCurrPart = None

    def startElement(self, tag, attributes):
        if tag == 'image':
            file = attributes['file']
            self.mCurrImage = {'file':file, 'box':[]}
        elif tag == 'box':
            top = int(attributes['top'])
            left = int(attributes['left'])
            width = int(attributes['width'])
            height = int(attributes['height'])
            self.mCurrBox = {'top':top, 'left':left, 'width':width, 'height':height, 'part':[]}
        elif tag == 'part':
            name = int(attributes['name'])
            x = int(attributes['x'])
            y = int(attributes['y'])
            self.mCurrBox['part'].append({'name':name, 'x':x, 'y':y})

    def endElement(self, tag):
        if tag == 'box':
            self.mCurrImage['box'].append(self.mCurrBox)
        elif tag == 'image':
            self.mImages.append(self.mCurrImage)

class ConvertUT(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

    def waitToClose(self, img):
        def mouseCallback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                logging.debug('(%3d, %3d)' % (x, y))

        cv2.imshow('image', img)
        cv2.setMouseCallback('image', mouseCallback, img)

        while True:
            cv2.imshow('image', img)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        
        cv2.destroyAllWindows()

    def Fts2xml(self):
        # 将fts文件转成训练需要的xml
        ftsDir = '/Users/palance/Downloads/FaceDataset/ibug_300W_large_face_landmark_dataset/helen/annotation240'
        imgDir = '/Users/palance/Downloads/FaceDataset/ibug_300W_large_face_landmark_dataset/helen/trainset'
        xmlPath = '/Users/palance/Downloads/FaceDataset/ibug_300W_large_face_landmark_dataset/helen/train.xml'
        f5s2xml = FTS2Xml(imgDir, ftsDir, xmlPath)
        f5s2xml.MainProc()

    def Xml2Img(self):
        # 将xml中标注的landmarks点画到img上
        xmlPath = '/Users/palance/Downloads/FaceDataset/ibug_300W_large_face_landmark_dataset/helen/train.xml'

        # 将起笔、运笔、抬笔交给画笔处理

        parser = xml.sax.make_parser()
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        handler = XmlHandler()
        parser.setContentHandler(handler)
        parser.parse(xmlPath)
        nImage = 0
        for image in handler.mImages:
            nImage += 1
            if nImage != 10:
                continue
            imgPath = image['file']
            logging.debug(imgPath)
            img = cv2.imread(imgPath)
            for box in image['box']:
                top, left, width, height = box['top'], box['left'], box['width'], box['height']
                right = left + width
                bottom = top + height
                logging.debug('(%d, %d, %d, %d)' % (top, left, width, height))
                cv2.rectangle(img, (top, left), (bottom, right), (0, 255, 0), 1)

                msg = ''
                landmarks = []
                for part in box['part']:
                    landmarks.append([part['x'], part['y']])

                pts = numpy.array(landmarks, numpy.int32)
                cv2.polylines(img, pts.reshape(-1, 1, 2), True, (0, 255, 0), 2)
            self.waitToClose(img)
            return # 只显示一张图片，就退出

    def Xml2Model(self):
        # 根据xml完成训练
        xmlDir = '/Users/palance/Downloads/FaceDataset/ibug_300W_large_face_landmark_dataset/helen'
        trainerPath = '/Users/palance/Subversion/dlib-19.8/examples/build/Release/train_shape_predictor_ex'
        subprocess.call([trainerPath, xmlDir])

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main
    # cmd: python -m unittest mpcap.CapUT.test01
