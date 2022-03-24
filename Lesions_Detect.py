import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, color
from skimage.morphology import disk
import skimage.filters.rank as sfr
from PIL import Image, ImageEnhance
from skimage import exposure
import collections
from phasepack import phasecong, phasecongmono, phasesym, phasesymmono
import pyfftw
import math
from math import pi

def Principle_Vessel_Segmentation_for_MA_and_HE(path):  #主干血管分割
    img = cv2.imread(path)
    _, green, _ = cv2.split(img)
    r1 = cv2.morphologyEx(green, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    morph_contrast_enhanced_green = R3
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    f5 = clahe.apply(cv2.subtract(morph_contrast_enhanced_green, green))
    ret, f6 = cv2.threshold(f5, 30, 255, cv2.THRESH_BINARY)  # 用来计算 mask
    mask = np.ones(f6.shape[:2], dtype="uint8") * 255
    im = cv2.bitwise_and(f6, f6, mask=mask)
    ret, fin = cv2.threshold(im, 11, 255, cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(newfin, kernel)
    dst = cv2.dilate(erosion, kernel)
    dst = 255 - dst
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 2000:
            cv2.drawContours(dst, [cnt], -1, 0, -1)
    dst = cv2.erode(dst, np.ones((2, 2), np.uint8))
    dst = cv2.dilate(dst, np.ones((2, 2), np.uint8))
    dst = cv2.erode(dst, np.ones((5, 5), np.uint8))
    dst = cv2.dilate(dst, np.ones((5, 5), np.uint8))
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 1500:
            cv2.drawContours(dst, [cnt], -1, 0, -1)
    return dst

def Principle_Vessel_Segmentation_for_EX_and_SE(path):
    img = cv2.imread(path)
    _, green, _ = cv2.split(img)
    r1 = cv2.morphologyEx(green, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    morph_contrast_enhanced_green = R3
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    f5 = clahe.apply(cv2.subtract(morph_contrast_enhanced_green, green))
    ret, f6 = cv2.threshold(f5, 15, 255, cv2.THRESH_BINARY)  # 用来计算 mask
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(f6, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 2000:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(newfin, kernel)
    dst = cv2.dilate(erosion, kernel)
    return dst

def Ocular_boundary_Pextraction(path):  #眼球边界检测
    img = cv2.imread(path)
    _, green, _ = cv2.split(img)
    erosion = cv2.dilate(green, kernel=np.ones((3, 3), np.uint8))
    dst = cv2.erode(erosion, kernel=np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        area = []
        for k in range(len(contours)):
            area.append(cv2.contourArea(contours[k]))
        max_idx = np.argmax(np.array(area))
        dst = cv2.drawContours(dst, contours, max_idx, 255, cv2.FILLED)
    h = dst.shape[0]
    w = dst.shape[1]
    c = max(h, w)
    cv2.circle(dst, (round(w / 2), round(h / 2)), round(c / 2 - 8), (255, 255, 255), -1)
    dst = cv2.erode(dst, kernel=np.ones((12, 12), np.uint8))
    return dst

def GetCenterLocation(filepath, result):
    img = cv2.imread(filepath)
    _, green, _ = cv2.split(img)
    ret, thresh1 = cv2.threshold(green, 230, 255, cv2.THRESH_BINARY)
    imgadd = cv2.bitwise_and(result, thresh1)
    final = cv2.bitwise_xor(imgadd, thresh1)
    dilation = cv2.dilate(final, np.ones((10, 10), np.uint8))
    final = cv2.erode(dilation, np.ones((5, 5), np.uint8))
    contours, hierarch = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if (len(contours)!=1 and area < 550):
            cv2.drawContours(final, [contours[i]], 0, 0, -1)
    contours, hierarch = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if (len(contours)!=0):
       a = b = discs = 0
       for i in range(len(contours)):
          rect = (cv2.minAreaRect(contours[i]))[0]
          a += rect[0]; b += rect[1]
       a = a / len(contours); b = b / len(contours)
       zxzb = (round(a), round(b))
       arr1 = np.array(zxzb, dtype=float)
       for i in range(len(contours[0])):
           ax = contours[0][i][0][0]
           ay = contours[0][i][0][1]
           point = (ax, ay)
           arr2 = np.array(point, dtype=float)
           disl = arr2 - arr1
           dis = math.hypot(disl[0], disl[1])
           if dis > discs:
               discs = dis
    else:
        zxzb = discs = 0
    return zxzb, discs

def FindCircleImg_for_EX(img,zxzb):
    cv2.circle(img, zxzb, 1, (0, 0, 255), 3)
    _, green, _ = cv2.split(img)
    ret, thresh1 = cv2.threshold(green, 147, 255, cv2.THRESH_BINARY)
    contours, hierarch = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 50:
            cv2.drawContours(thresh1, [contours[i]], 0, 0, -1)
    dilation = cv2.dilate(thresh1, np.ones((5, 5), np.uint8))
    final = cv2.erode(dilation, np.ones((5, 5), np.uint8))
    ret, thresh2 = cv2.threshold(green, 255, 255, cv2.THRESH_BINARY)
    cv2.circle(thresh2, zxzb, 156, (255, 255, 255), -1)
    imgpre = cv2.bitwise_and(thresh2, final)
    contours, hierarch = cv2.findContours(imgpre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 2050:
            cv2.drawContours(imgpre, [contours[i]], 0, 0, -1)
    dilation = cv2.dilate(imgpre, np.ones((3, 3), np.uint8))
    contours, hierarch = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    a = b = discs = 0
    arr1 = np.array(zxzb, dtype=float)
    for i in range(len(contours[0])):    #寻找圆的最大半径
        ax = contours[0][i][0][0]
        ay = contours[0][i][0][1]
        point = (ax, ay)
        arr2 = np.array(point, dtype=float)
        disl = arr2 - arr1
        dis = math.hypot(disl[0], disl[1])
        if dis > discs:
            discs = dis
    cv2.circle(dilation, zxzb, round(discs), (255, 255, 255), -1)
    dilation = cv2.bitwise_and(dilation, final)
    contours, hierarch = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 1550:
            cv2.drawContours(dilation, [contours[i]], 0, 0, -1)
    return dilation

def FindCircleImg_for_SE(img,zxzb):
    cv2.circle(img, zxzb, 1, (0, 0, 255), 3)
    _, green, _ = cv2.split(img)
    ret, thresh1 = cv2.threshold(green, 147, 255, cv2.THRESH_BINARY)
    contours, hierarch = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 50:
            cv2.drawContours(thresh1, [contours[i]], 0, 0, -1)
    dilation = cv2.dilate(thresh1, np.ones((5, 5), np.uint8))
    final = cv2.erode(dilation, np.ones((5, 5), np.uint8))
    ret, thresh2 = cv2.threshold(green, 255, 255, cv2.THRESH_BINARY)
    cv2.circle(thresh2, zxzb, 125, (255, 255, 255), -1)
    imgpre = cv2.bitwise_and(thresh2, final)
    contours, hierarch = cv2.findContours(imgpre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 2050:
            cv2.drawContours(imgpre, [contours[i]], 0, 0, -1)
    dilation = cv2.dilate(imgpre, np.ones((3, 3), np.uint8))
    contours, hierarch = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    a = b = discs = 0
    arr1 = np.array(zxzb, dtype=float)
    for i in range(len(contours[0])):    #寻找圆的最大半径
        ax = contours[0][i][0][0]
        ay = contours[0][i][0][1]
        point = (ax, ay)
        arr2 = np.array(point, dtype=float)
        disl = arr2 - arr1
        dis = math.hypot(disl[0], disl[1])
        if dis > discs:
            discs = dis
    cv2.circle(dilation, zxzb, round(discs), (255, 255, 255), -1)
    dilation = cv2.bitwise_and(dilation, final)
    contours, hierarch = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 1550:
            cv2.drawContours(dilation, [contours[i]], 0, 0, -1)
    return dilation

def Primary_ML_segmentation(img):
    _, green, _ = cv2.split(img)
    ret, thresh1 = cv2.threshold(green, 25, 255, cv2.THRESH_BINARY)
    dilation = cv2.dilate(thresh1, np.ones((3, 3), np.uint8))
    final = cv2.erode(dilation, np.ones((3, 3), np.uint8))
    final = 255 - final
    contours, hierarchy = cv2.findContours(final, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        for i in range(len(contours)):
            discs = 0
            area = cv2.contourArea(contours[i])
            rect = (cv2.minAreaRect(contours[i]))[0]
            a = rect[0];b = rect[1]
            zxzb = (round(a), round(b))
            arr1 = np.array(zxzb, dtype=float)
            for j in range(len(contours[i])):
                ax = contours[i][j][0][0]
                ay = contours[i][j][0][1]
                point = (ax, ay)
                arr2 = np.array(point, dtype=float)
                disl = arr2 - arr1
                dis = math.hypot(disl[0], disl[1])
                if dis > discs:
                    discs = dis
            areapercent = area / (pi * discs * discs)
            if (area <= 2500 or area > 15000 or discs > 100 or areapercent < 0.3):
                cv2.drawContours(final, [contours[i]], 0, 0, -1)
    dilation = cv2.dilate(final, np.ones((11, 11), np.uint8))
    final = cv2.erode(dilation, np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(final, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours) > 1 and len(contours) != 0):
        areapercentori = 0
        for i in range(len(contours)):
            discs = 0
            area = cv2.contourArea(contours[i])
            rect = (cv2.minAreaRect(contours[i]))[0]
            a = rect[0]; b = rect[1]
            zxzb = (round(a), round(b))
            arr1 = np.array(zxzb, dtype=float)
            for j in range(len(contours[i])):
                ax = contours[i][j][0][0]
                ay = contours[i][j][0][1]
                point = (ax, ay)
                arr2 = np.array(point, dtype=float)
                disl = arr2 - arr1
                dis = math.hypot(disl[0], disl[1])
                if dis > discs:
                    discs = dis
            areapercent = area / (pi * discs * discs)
            if (areapercent > areapercentori and i != 0):
                cv2.drawContours(final, [contours[i]], 0, 0, -1)
            if (areapercent <= areapercentori):
                areapercentori = areapercent
                cv2.drawContours(final, [contours[i - 1]], 0, 0, -1)
    if len(contours) == 0:
        ret, thresh3 = cv2.threshold(green, 5, 255, cv2.THRESH_BINARY)
        dilation = cv2.dilate(thresh3, np.ones((5, 5), np.uint8))
        final = cv2.erode(dilation, np.ones((3, 3), np.uint8))
        final = 255 - final
        contoursn, hierarchy = cv2.findContours(final, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contoursn) != 0:
            for i in range(len(contoursn)):
                area = cv2.contourArea(contoursn[i])
                discs = 0
                rect = (cv2.minAreaRect(contoursn[i]))[0]
                a = rect[0]; b = rect[1]
                zxzb = (round(a), round(b))
                arr1 = np.array(zxzb, dtype=float)
                for j in range(len(contoursn[i])):
                    ax = contoursn[i][j][0][0]
                    ay = contoursn[i][j][0][1]
                    point = (ax, ay)
                    arr2 = np.array(point, dtype=float)
                    disl = arr2 - arr1
                    dis = math.hypot(disl[0], disl[1])
                    if dis > discs:
                        discs = dis
                areapercent = area / (pi * discs * discs)
                if (areapercent < 0.2 or area > 50000 or area < 1000):
                    cv2.drawContours(final, [contoursn[i]], 0, 0, -1)
        final = cv2.dilate(final, np.ones((19, 19), np.uint8))
        final = cv2.erode(final, np.ones((3, 3), np.uint8))
        contoursn, hierarchy = cv2.findContours(final, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contoursn) > 1 and len(contoursn) != 0):
            areapercentorin = 0
            for i in range(len(contoursn)):
                discs = 0
                area = cv2.contourArea(contoursn[i])
                rect = (cv2.minAreaRect(contoursn[i]))[0]
                a = rect[0]; b = rect[1]
                zxzb = (round(a), round(b))
                arr1 = np.array(zxzb, dtype=float)
                for j in range(len(contoursn[i])):
                    ax = contoursn[i][j][0][0]
                    ay = contoursn[i][j][0][1]
                    point = (ax, ay)
                    arr2 = np.array(point, dtype=float)
                    disl = arr2 - arr1
                    dis = math.hypot(disl[0], disl[1])
                    if dis > discs:
                        discs = dis
                areapercentn = area / (pi * discs * discs)
                if (areapercentn < areapercentorin):
                    cv2.drawContours(final, [contoursn[i]], 0, 0, -1)
                if (areapercentn >= areapercentorin and i != 0):
                    areapercentorin = areapercentn
                    cv2.drawContours(final, [contoursn[i - 1]], 0, 0, -1)
    finalori = final.copy()
    contours, hierarchy = cv2.findContours(final, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        for i in range(len(contours)):
            rect = (cv2.minAreaRect(contours[i]))[0]
            a = rect[0]; b = rect[1]
            zxzb = (round(a), round(b))
            cv2.drawContours(final, [contours[i]], 0, 0, -1)
            cv2.circle(final, zxzb, 70, (255, 255, 255), -1)
            final = cv2.bitwise_and(final, finalori)
    return final

def Primary_MA_segmentation(green):
    dst = phasesymmono(green)
    dst = dst[0]
    h = dst.shape[0]
    w = dst.shape[1]
    erosion = cv2.erode(dst, kernel=np.ones((3, 3), np.uint8))
    dst = cv2.dilate(erosion, kernel=np.ones((3, 3), np.uint8))
    dst = dst * 255
    ret, dst = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)
    dst = dst.astype(np.uint8)
    erosion = cv2.dilate(dst, kernel=np.ones((7, 7), np.uint8))
    dst = cv2.erode(erosion, kernel=np.ones((7, 7), np.uint8))
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        for i in range(len(contours)):
            discs = 0
            area = cv2.contourArea(contours[i])
            rect = (cv2.minAreaRect(contours[i]))[0]
            a = rect[0]; b = rect[1]
            zxzb = (round(a), round(b))
            arr1 = np.array(zxzb, dtype=float)
            for j in range(len(contours[i])):
                ax = contours[i][j][0][0]
                ay = contours[i][j][0][1]
                point = (ax, ay)
                arr2 = np.array(point, dtype=float)
                disl = arr2 - arr1
                dis = math.hypot(disl[0], disl[1])
                if dis > discs:
                    discs = dis
            if (area <= 8 or area > 53 or discs > 6):
                cv2.drawContours(dst, [contours[i]], 0, 0, -1)
    dst = cv2.dilate(dst, kernel=np.ones((5, 5), np.uint8))
    dst = cv2.erode(dst, kernel=np.ones((5, 5), np.uint8))
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        discs = 0
        area = cv2.contourArea(contours[i])
        rect = (cv2.minAreaRect(contours[i]))[0]
        a = rect[0];  b = rect[1]
        zxzb = (round(a), round(b))
        arr1 = np.array(zxzb, dtype=float)
        for j in range(len(contours[i])):
            ax = contours[i][j][0][0]
            ay = contours[i][j][0][1]
            point = (ax, ay)
            arr2 = np.array(point, dtype=float)
            disl = arr2 - arr1
            dis = math.hypot(disl[0], disl[1])
            if dis > discs:
                discs = dis
        areapercent = area/(pi*discs*discs)
        if (areapercent<0.4 or green[round(b), round(a)] > 36 or abs(a-w) < 15 or abs(b-h) < 15 ):
            cv2.drawContours(dst, [contours[i]], 0, 0, -1)
    return dst

def Primary_HE_segmentation(green,xgfg,result,MApredetect):
    dst = phasesymmono(green)
    dst = dst[0]
    h = dst.shape[0]; w = dst.shape[1]
    erosion = cv2.erode(dst, kernel=np.ones((3, 3), np.uint8))
    dst = cv2.dilate(erosion, kernel=np.ones((3, 3), np.uint8))
    dst = dst * 255
    ret, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)
    dst = dst.astype(np.uint8)
    erosion = cv2.dilate(dst, kernel=np.ones((3, 3), np.uint8))
    dst = cv2.erode(erosion, kernel=np.ones((3, 3), np.uint8))
    cut = cv2.bitwise_and(dst, xgfg)
    dst = cv2.subtract(dst, cut)
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        for i in range(len(contours)):
            discs = 0
            area = cv2.contourArea(contours[i])
            rect = (cv2.minAreaRect(contours[i]))[0]
            a = rect[0]; b = rect[1]
            zxzb = (round(a), round(b))
            arr1 = np.array(zxzb, dtype=float)
            for j in range(len(contours[i])):
                ax = contours[i][j][0][0]
                ay = contours[i][j][0][1]
                point = (ax, ay)
                arr2 = np.array(point, dtype=float)
                disl = arr2 - arr1
                dis = math.hypot(disl[0], disl[1])
                if dis > discs:
                    discs = dis
            if (area <= 40 or area > 480 or discs > 45):
                cv2.drawContours(dst, [contours[i]], 0, 0, -1)
    dst = cv2.dilate(dst, kernel=np.ones((5, 5), np.uint8))
    dst = cv2.erode(dst, kernel=np.ones((5, 5), np.uint8))
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        discs = 0
        area = cv2.contourArea(contours[i])
        rect = (cv2.minAreaRect(contours[i]))[0]
        a = rect[0]; b = rect[1]
        c = (cv2.minAreaRect(contours[i]))[1][0]
        d = (cv2.minAreaRect(contours[i]))[1][1]
        zxzb = (round(a), round(b))
        arr1 = np.array(zxzb, dtype=float)
        for j in range(len(contours[i])):
            ax = contours[i][j][0][0]
            ay = contours[i][j][0][1]
            point = (ax, ay)
            arr2 = np.array(point, dtype=float)
            disl = arr2 - arr1
            dis = math.hypot(disl[0], disl[1])
            if dis > discs:
                discs = dis
        areapercent = area / (pi * discs * discs)
        if (areapercent<0.1 or green[round(b), round(a)] > 36 or abs(a-w) < 20 or abs(b-h) < 20 ):
            cv2.drawContours(dst, [contours[i]], 0, 0, -1)
        if (area <= 380 and abs(c-d)>30):
            cv2.drawContours(dst, [contours[i]], 0, 0, -1)
        if (area <= 200 and abs(c-d)>10):
            cv2.drawContours(dst, [contours[i]], 0, 0, -1)
        if (area <= 60 and green[round(b), round(a)] > 26 ):
            cv2.drawContours(dst, [contours[i]], 0, 0, -1)
    dst = cv2.bitwise_and(dst, result)
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        for i in range(len(contours)):
            rect = (cv2.minAreaRect(contours[i]))[0]
            a = rect[0]; b = rect[1]
            if (MApredetect[round(b), round(a)] > 0):
                cv2.drawContours(dst, [contours[i]], 0, 0, -1)
    return dst

def Primary_EX_segmentation(img, circlearea, arr1):
    _, green, _ = cv2.split(img)
    ret, thresh1 = cv2.threshold(green, 200, 255, cv2.THRESH_BINARY)
    result_f = cv2.subtract(thresh1, circlearea)
    result_f= cv2.erode(result_f, np.ones((5, 5), np.uint8))
    result_f = cv2.dilate(result_f, np.ones((5, 5), np.uint8))
    contours, hierarch = cv2.findContours(result_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        for i in range(len(contours)):
            discs = 10000
            area = cv2.contourArea(contours[i])
            for j in range(len(contours[i])):
                ax = contours[i][j][0][0]
                ay = contours[i][j][0][1]
                point = (ax, ay)
                arr2 = np.array(point, dtype=float)
                disl = arr2 - arr1
                dis = math.hypot(disl[0], disl[1])
                if dis < discs:
                    discs = dis
            if (discs < 300 and area > 5500):
                cv2.drawContours(result_f, [contours[i]], 0, 0, -1)
            if (area < 50 or area > 12000):
                cv2.drawContours(result_f, [contours[i]], 0, 0, -1)
    ret, thresh2 = cv2.threshold(green, 1, 255, cv2.THRESH_BINARY)
    thresh2 = cv2.bitwise_xor(thresh2, thresh1)
    h = thresh2.shape[0]
    w = thresh2.shape[1]
    c = max(h, w)
    cv2.circle(thresh2, (round(w / 2), round(h / 2)), round(c / 2), (255, 255, 255), -1)
    result_f = cv2.bitwise_and(thresh2, result_f)
    return result_f

def Primary_SE_segmentation(img, circlearea, result):
    _, green, _ = cv2.split(img)
    ret, thresh1 = cv2.threshold(green, 160, 255, cv2.THRESH_BINARY)
    ret, thresh3 = cv2.threshold(green, 159, 255, cv2.THRESH_BINARY)
    result_f = cv2.subtract(thresh1, circlearea)
    result_f= cv2.erode(result_f, np.ones((5, 5), np.uint8))
    result_f = cv2.dilate(result_f, np.ones((5, 5), np.uint8))
    result_f1 = cv2.subtract(thresh3, circlearea)
    result_f1 = cv2.erode(result_f1, np.ones((5, 5), np.uint8))
    result_f1 = cv2.dilate(result_f1, np.ones((5, 5), np.uint8))
    base = result_f1.copy()
    cut = result_f1 - result_f
    ret, thresh1 = cv2.threshold(green, 200, 255, cv2.THRESH_BINARY)
    ret, thresh3 = cv2.threshold(green, 199, 255, cv2.THRESH_BINARY)
    result_f2 = cv2.subtract(thresh1, circlearea)
    result_f2 = cv2.erode(result_f2, np.ones((5, 5), np.uint8))
    result_f2 = cv2.dilate(result_f2, np.ones((5, 5), np.uint8))
    result_f3 = cv2.subtract(thresh3, circlearea)
    result_f3 = cv2.erode(result_f3, np.ones((5, 5), np.uint8))
    result_f3 = cv2.dilate(result_f3, np.ones((5, 5), np.uint8))
    cut1 = result_f3 - result_f2
    precut = cut - cut1
    contours, hierarch = cv2.findContours(precut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if ( area < 10 ):
                cv2.drawContours(precut, [contours[i]], 0, 0, -1)
    contours, hierarch = cv2.findContours(result_f1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if (area < 1000 or area > 10000):
                cv2.drawContours(result_f1, [contours[i]], 0, 0, -1)
    contours1, hierarch1 = cv2.findContours(precut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarch = cv2.findContours(result_f1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        for i in range(len(contours)):
            k = 0
            for j in range(len(contours[i])):
                ax = contours[i][j][0][0]
                ay = contours[i][j][0][1]
                for m in range(len(contours1)):
                    for n in range(len(contours1[m])):
                        am = contours1[m][n][0][0]
                        an = contours1[m][n][0][1]
                        if (ax == am and ay == an):
                            k += 1
            if(k < 1):
                cv2.drawContours(result_f1, [contours[i]], 0, 0, -1)
    oriresult = result.copy()
    contours, hierarch = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        for i in range(len(contours)):
            rect = (cv2.minAreaRect(contours[i]))[0]
            a = rect[0]; b = rect[1]
            zxzb = (round(a), round(b))
            cv2.circle(result, zxzb, 1, (128, 128, 128), 3)
            if (result_f1[round(b), round(a)] > 10 and result_f1[round(b-20), round(a)] > 10 ):
                cv2.drawContours(result, [contours[i]], 0, 0, -1)
    result = oriresult - result
    contours, hierarch = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if (area < 100):
                cv2.drawContours(result, [contours[i]], 0, 0, -1)
    return result

if __name__ == '__main__':
    imgpath = r'xxxx'
    for file in os.listdir(imgpath):
        filepath = os.path.join(imgpath, file)
        img = cv2.imread(filepath)
        blue, green, red = cv2.split(img)
        MLdetect = Primary_ML_segmentation(img)
        result = Principle_Vessel_Segmentation_for_EX_and_SE(filepath)
        zxzb, div = GetCenterLocation(filepath, result)
        if (zxzb!=0):
           arr1 = np.array(zxzb, dtype=float)
           circlearea = FindCircleImg_for_EX(img, zxzb)
           EXdetect = Primary_EX_segmentation(img, circlearea, arr1)
           dst = Primary_MA_segmentation(green)
           result = Principle_Vessel_Segmentation_for_MA_and_HE(filepath)
           cut = cv2.bitwise_and(dst, result)
           MAdetect = cv2.subtract(dst, cut)
           MApredetect = MAdetect.copy()
           xgfg = Principle_Vessel_Segmentation_for_MA_and_HE(filepath)
           result = Ocular_boundary_Pextraction(filepath)
           HEdetect = Primary_HE_segmentation(green, xgfg, result, MApredetect)
           result = Principle_Vessel_Segmentation_for_EX_and_SE(filepath)
           zxzb, div = GetCenterLocation(filepath, result)
           arr1 = np.array(zxzb, dtype=float)
           circlearea = FindCircleImg_for_SE(img, zxzb)
           EXpredetect = Primary_EX_segmentation(img, circlearea, arr1)
           SEdetect = Primary_SE_segmentation(img, circlearea,  EXpredetect)
           EXdetect = EXdetect - SEdetect
           cv2.imshow('image', img)
           cv2.imshow('HEdetect', HEdetect)
           cv2.imshow('SEdetect', SEdetect)
           cv2.imshow('EXdetect', EXdetect)
           cv2.imshow('MAdetect', MAdetect)
           cv2.imshow('MLdetect', MLdetect)
           cv2.waitKey(0)
        else:
            print(file)