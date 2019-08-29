import cv2
import sys
import numpy as np 
import warnings
from pathlib import Path

from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from deep_sort.detection import Detection
from face_recognition import face_encodings
from bbox import AnnotationInstance
from bbox.instances import BBox
from bbox.contrib.detection.darknet import DarknetObjectDetector

        
model_path = sys.argv[1]

cap = cv2.VideoCapture(0)
warnings.filterwarnings('ignore')

cfg = list(Path(model_path).glob('*.cfg'))
weights = list(Path(model_path).glob('*.weights'))
data = list(Path(model_path).glob('*.data'))
names = list(Path(model_path).glob('names.txt'))

assert bool(len(cfg) > 0) is True, f'Unable to locate {Path(model_path) / "[MODEL].cfg"}'
assert bool(len(weights) > 0) is True, f'Unable to locate {Path(model_path) / "[MODEL].weights"}'
assert bool(len(data) > 0) is True, f'Unable to locate {Path(model_path) / "[MODEL].data"}'
assert bool(len(names) > 0) is True, f'Unable to locate {Path(model_path) / "names.txt"}: Check path in [MODEL].data'

labels = list(filter(lambda x: x != '', open(str(names[0]), 'r').read().split('\n')))

dn_path = '/home/anders/huddly/libs/darknet'
with DarknetObjectDetector(Path(dn_path).parents[0] / 'darknet',
                            cfg_path=cfg[0],
                            weights_path=weights[0],
                            data_file_path=data[0],
                            detection_score_threshold=0.4) as od:
    ret, first_frame = cap.read()
    height, width, ch = first_frame.shape
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", 0.2, None)
    tracker = Tracker(metric, n_init=30)
    
    i = 0
    first = True
    while True:
        result = []
        ret, frame = cap.read()
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 3:
                continue
            info = {'bbox': track.to_tlwh(), 'id': track.track_id, 'hits': track.hits}
            result.append(info)

        if (i % 2 == 0):
            i = 0
            pred = od.detect_image(frame)
            detections = []
            for inst in pred:
                if inst.label == 'person':
                    crop = frame[int(inst.ymin*height):int(inst.ymax*height), int(inst.xmin*width):int(inst.xmax*width)]
                    feat = face_encodings(crop)
                    detections.append(Detection((int(inst.xmin*width), int(inst.ymin*height), int((inst.xmax-inst.xmin)*width),
                                                int((inst.ymax-inst.ymin)*height)),
                                                inst.score,
                                                list(feat[0]) if len(feat) > 0 else [1]*128))
            first = False
            tracker.predict()
            tracker.update(detections)
        

        for inf in result:
            b = inf['bbox']
            t_id = inf['id']
            bb = BBox(xmin=np.clip(int(b[0]), 0, width), ymin=np.clip(int(b[1]), 0, height), xmax=int(b[0]+b[2]), ymax=int(b[1]+b[3]),
                    label='track: ' + str(t_id) + f" hits: {inf['hits']}",
                    coordinate_mode='ABSOLUTE')
            bb.overlaid_on_image(frame, (200, 200, 200), draw_label=True, draw_score=False)
            #cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]+b[0]), int(b[3]+b[1])), (255, 0, 0), 2)
            # cv2.putText(frame, str(t_id), (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        i += 1

        image = cv2.resize(frame, (1920, 1080))
        cv2.imshow('Pred', image)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
    cv2.destroyAllWindows() 