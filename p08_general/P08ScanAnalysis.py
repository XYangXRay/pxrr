# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 2021

@author: Florian
"""

from p08_general.P08ScanTools import Scan, ScanFitter
from scipy import optimize

import numpy

import matplotlib.pyplot as plt


class CalibrationTools(object):

    def __init__(self):
        pass


    @staticmethod
    def calibrateDetectorScan(filename, detector=None):

        scan = Scan()
        scan.load_scan(filename)

        if detector == None:
            detector = 'p100k'


        if detector.lower() == 'p100k':
            pixel_size = 0.172
        elif detector.lower() == 'eiger':
            pixel_size = 0.075

            x,y,z = numpy.where( scan.image_data['eiger']>(2**32-2) )

            scan.image_data['eiger'][x,y,z]=0

        elif detector.lower() == 'lambda':
            pixel_size = 0.055
            x,y,z = numpy.where( scan.image_data['lambda']>(6*10**4) )
            scan.image_data['lambda'][x,y,z]=0


        central_pixel = []
        slope = []
        
        for direction in [0, 1]:
            res =  CalibrationTools.analyzeDirection(scan, detector, direction)

            central_pixel.append( res[1] )
            slope.append( res[0] )


        max_slope = max(slope)
        mm_slope = max_slope*pixel_size
        sdd = mm_slope/numpy.tan(1*numpy.pi/180)

        if (slope[0] > slope[1] and scan.scan_motor_names[0].lower() == "tt") or (slope[0] < slope[1] and scan.scan_motor_names[0].lower() == "tth"):
            orientation = "vertical"
        else:
            orientation = "horizontal"


        

        return sdd, central_pixel, orientation

    @staticmethod
    def analyzeDirection(scan, detector, direction=0):
        img_idx  = range(len(scan.image_data[detector]))
        peak_pos = []
        peak_int = []
    
        for idx in img_idx:

            img = scan.image_data[detector][idx]
            pixels = range(len(img.sum(direction)))

            sigma = 1
            norm = numpy.sqrt(2*numpy.pi*sigma**2)

            max_pos = img.sum(direction).argmax()

            params = [sigma, max_pos, norm*img.sum(direction).max(), 0 ]

            fitfunc = lambda p, x: ScanFitter._gauss(x, p) # Target function
            errfunc = lambda p, x, y: numpy.abs(fitfunc(p, x) - y) # Distance to the target function

            p1, success = optimize.leastsq(errfunc, params[:], args=(pixels, img.sum(direction)))


            x_fit = numpy.linspace(min(pixels),max(pixels), len(pixels)*10)
            fit_res = ScanFitter._gauss(x_fit, p1)

            intensity = ScanFitter._gauss(x_fit, params)

            peak_pos.append(p1[1])
            peak_int.append(p1[2])


        peak_int_array = numpy.asarray(peak_int)
        peak_pos_array = numpy.asarray(peak_pos)
        img_idx_array = numpy.asarray(img_idx)
        scan_motors_array = numpy.asarray(scan.scan_motors[0])

        good_idx = numpy.where( peak_int_array > peak_int_array.max()*.8)[0]

        res = numpy.polyfit(scan_motors_array[good_idx], peak_pos_array[good_idx], 1) 
        
        return res




if __name__ == '__main__':
    #filename = '/asap3/petra3/gpfs/p08/2021/data/11012380/raw/detectorcalib_00375.fio'
    filename = '/asap3/petra3/gpfs/p08/2022/commissioning/c20220412_000_apr22/raw/align_10kev_lens_08408.fio'
    #filename = '/asap3/petra3/gpfs/p08/2022/commissioning/c20220412_000_apr22/raw/align_10kev_lens_08410.fio'
    #filename = '/asap3/petra3/gpfs/p08/2021/data/11012380/raw/detcalib_h_00399.fio'
    #filename = '/asap3/petra3/gpfs/p08/2021/data/11012380/raw/detcalib_v_00400.fio'

    sdd, central_pixel, orientation = CalibrationTools.calibrateDetectorScan(filename, detector = 'lambda')

    print ("sample detector distance: %6.2f mm" % sdd)
    print ("central pixel: [%6.2f, %6.2f]" % (central_pixel[0], central_pixel[1]) )
    print ("detector orientation: %s" % orientation)

