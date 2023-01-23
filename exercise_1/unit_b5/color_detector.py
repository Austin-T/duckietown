#!/usr/bin/env python3
import os
import cv2
import numpy as np
from time import sleep


def gst_pipeline_string():
    # Parameters from the camera_node
    # Refer here : https://github.com/duckietown/dt-duckiebot-interface/blob/daffy/packages/camera_driver/config/jetson_nano_camera_node/duckiebot.yaml
    res_w, res_h, fps = 640, 480, 30
    fov = 'full'
    # find best mode
    camera_mode = 3  # 
    # compile gst pipeline
    gst_pipeline = """ \
            nvarguscamerasrc \
            sensor-mode= exposuretimerange="100000 80000000" ! \
            video/x-raw(memory:NVMM), width=, height=, format=NV12, 
                framerate=/1 ! \
            nvjpegenc ! \
            appsink \
        """.format(
        camera_mode,
        res_w,
        res_h,
        fps
    )

    # ---
    print("Using GST pipeline: ``".format(gst_pipeline))
    return gst_pipeline


cap = cv2.VideoCapture()
cap.open(gst_pipeline_string(), cv2.CAP_GSTREAMER)

N_SPLITS = os.environ.get("N_SPLITS") or 10

hue_to_color = {
    0: "red",
    30: "orange",
    60: "yellow",
    120: "green",
    180: "turquoise",
    240: "blue",
    270: "purple",
    330: "pink"
}

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # proceed if frame was correctly captured
    if ret:
        # convert frame from BGR to HSV
        frame = np.float32(frame)
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # divide the frame horizontally into N_SPLITS rows
        sectors = np.split(frameHSV, N_SPLITS)
        
        # for each division of the image, find the most present color
        detected_color = []
        for sector in sectors:
            # reshape the sector into a list of HSV values
            hsv_list = sector.reshape(-1, 3)
            
            # count the frequency of each pixel value
            unique, counts = np.unique(hsv_list, axis=0, return_counts=True)
            
            # find the most common frequency
            hue, sat, val = unique[np.argmax(counts)]
    
            # find the nearest predefined color corresponding to the hue
            nearest_hue = min(hue_to_color, key=lambda x:abs(x-hue))
    
            # write detected color to output array
            detected_color.append(hue_to_color[nearest_hue])
    
        # print the output array
        print("Sector\tDetected Color")
        for i in range(0, N_SPLITS):
            print("{}\t{}".format(i, detected_color[i]))

    sleep(1)
