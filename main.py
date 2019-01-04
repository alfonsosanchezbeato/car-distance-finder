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
from vehicle import TrackedVehicle, BoundingBox

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

        # Get the index of the class label from the detections and filter
        # non-vehicles
        idx = int(detections[0, 0, i, 1])
        if CLASSES[idx] != "bicycle" and CLASSES[idx] != "bus" and \
           CLASSES[idx] != "car" and CLASSES[idx] != "motorbike":
            continue

        # Get bounding box for the object
        print(detections[0, 0, i, 3:7])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        vehicles.append(BoundingBox(startX, startY,
                                    endX - startX, endY - startY))

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
def merge_vehicles(frame, vehicles, tracked_vehicles):
    # XXX just track the first set for the moment
    if tracked_vehicles:
        return
    for v in vehicles:
        tracked_vehicles.append(TrackedVehicle(frame, v))


def tracker_update(frame, vehicle):
    (success, box) = vehicle.update(frame)
    if success:
        box = vehicle.roi
        color = (0, 0, 255)
        cv2.rectangle(frame, (box.top_x, box.top_y),
                      (box.top_x + box.width, box.top_y + box.height),
                      color, 2)
        label = "Distance: {:.2f} m".format(vehicle.calculate_distance())
        y = box.top_y - 15 if box.top_y - 15 > 15 else box.top_y + 15
        cv2.putText(frame, label, (box.top_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return success


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
        if not tracked_vehicles:
            print('Detecting vehicles')
            vehicles = detect_vehicle(net, 0.2, frame)
        merge_vehicles(frame, vehicles, tracked_vehicles)
        # Track and show currently detected vehicles
        tracked_vehicles[:] = [v for v in tracked_vehicles
                               if tracker_update(frame, v)]
        cv2.imshow("Press 'q' to exit", frame)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break


def main(argv=None):
    print('CV version is', cv2.__version__)
    if argv is None:
        argv = sys.argv[1:]
    args = parseargs(argv)
    net = load_model()
    frame_loop(args.video, net)


if __name__ == '__main__':
    sys.exit(main())
