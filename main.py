# Copyright (C) 2018 Alfonso Sanchez-Beato
# Steps
# 1. Input as recorded video atm
# 2. Identify vehicles in the image
# 3. Identify back lights for each vehicle
# 3. Track them, calculate distance using distance between lights

import argparse
import cv2
import numpy as np
import sys

PROGRAM = 'detector'
VERSION = '0.1'
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
# We are interested in "bicycle", "bus", "car", and "motorbike".
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


class bounding_box:
    def __init__(self, top_x, top_y, bottom_x, bottom_y):
        self.top_x = 0
        self.top_y = 0
        self.bottom_x = 0
        self.bottom_y = 0


def load_model():
    # load our serialized model from disk
    print('Loading caffe model')
    return cv2.dnn.readNetFromCaffe('data/MobileNetSSD_deploy.prototxt',
                                    'data/MobileNetSSD_deploy.caffemodel')


def parseargs(argv=None):
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description='Calculate distance to objects')

    parser.add_argument('-v', '--version', action='version',
                        version='{} {}'.format(PROGRAM, VERSION))
    parser.add_argument('video', help='Path to input video')
    return parser.parse_args()


def detect_vehicle(net, min_confidence, frame):
    # Normalize
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
                                 (300, 300), 127.5)
    # Pass blob to CNN
    net.setInput(blob)
    detections = net.forward()
    vehicles = []
    for i in np.arange(0, detections.shape[2]):
        # Filter based on confidence for each detection
        confidence = detections[0, 0, i, 2]
        if confidence < min_confidence:
            continue

        # extract the index of the class label from the detections
        # then compute the (x, y)-coordinates of the bounding box for
        # the object
        idx = int(detections[0, 0, i, 1])
        print(detections[0, 0, i, 3:7])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        vehicles.append(bounding_box(startX, startY, endX, endY))

        # display the prediction
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        print("[INFO] {}".format(label))
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    return vehicles


# Merges currently tracked vehicles with currently detected. This can only add
# vehicles to the list - they get removed only when tracking fails.
def merge_vehicles(tracked_vehicles, vehicles):
    return


def frame_loop(video, net):
    # Process video, frame by frame
    print('Processing', video)
    vc = cv2.VideoCapture(video)
    tracked_vehicles = []
    while True:
        res, frame = vc.read()
        if res is False:
            break
        # XXX make confidence configurable?
        vehicles = detect_vehicle(net, 0.2, frame)
        cv2.imshow("Press 'q' to exit", frame)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        merge_vehicles(tracked_vehicles, vehicles)
        # Track currently detected vehicles


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parseargs(argv)
    net = load_model()
    frame_loop(args.video, net)


if __name__ == '__main__':
    sys.exit(main())
