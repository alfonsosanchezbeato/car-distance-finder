# Copyright (C) 2018 Alfonso Sanchez-Beato

import cv2


class BoundingBox:
    def __init__(self, top_x, top_y, width, height):
        self.top_x = int(top_x)
        self.top_y = int(top_y)
        self.width = int(width)
        self.height = int(height)

    def scale(self, factor):
        self.top_x = factor*self.top_x
        self.top_y = factor*self.top_y
        self.width = factor*self.width
        self.height = factor*self.height


class TrackedVehicle:
    # Receives a frame and a bounding box with the current vehicle position
    def __init__(self, frame, roi):
        self.roi = roi
        # The way to do tracker initialization depends on the openCV version
        (major, minor) = cv2.__version__.split(".")[:2]
        if int(major) == 3 and int(minor) < 3:
            self.tracker = cv2.Tracker_create("KCF")
        else:
            # Best ones for this use case: Median Flow, Mosse, CSRT.
            # Slow, does not adapt bounding box
            # self.tracker = cv2.TrackerBoosting_create()
            # Slow, does not adapt bounding box
            # self.tracker = cv2.TrackerMIL_create()
            # Does not adapt bounding box
            # self.tracker = cv2.TrackerKCF_create()
            # Jumps between random parts of the image
            # self.tracker = cv2.TrackerTLD_create()
            # Fast, good tracking, adapts box, can lose tracking
            # self.tracker = cv2.TrackerMedianFlow_create()
            # Slow, jumps, large caffe model
            # self.tracker = cv2.TrackerGOTURN_create()
            # Fastest, does not adapt bounding box, can lose track
            self.tracker = cv2.TrackerMOSSE_create()
            # A bit slow, best tracking, adapts box
            # self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, (roi.top_x, roi.top_y, roi.width, roi.height))
        # XXX Constants for the moment
        # Focal length (pix)
        self._focal_len = 300
        # Lenght of back part of a vehicle (meters). Should change depending on
        # the vehicle type.
        self._vehicle_back_len = 2

    def update(self, frame):
        (success, box) = self.tracker.update(frame)
        self.roi = BoundingBox(*box)
        return success, self.roi

    def calculate_distance(self):
        return self._vehicle_back_len*self._focal_len/self.roi.width
