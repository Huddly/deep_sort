import cv2
import sys
import numpy as np 
import warnings
import os
from pathlib import Path

from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from deep_sort.detection import Detection
from face_recognition import face_encodings
from bbox import AnnotationInstance, AnnotationEntry, AnnotationContainer, DatasetSourceProvider
from bbox.instances import BBox
from bbox.contrib.detection.darknet import DarknetObjectDetector
    
model_path = sys.argv[1]
vid_path = sys.argv[2]
warnings.filterwarnings('ignore')

cfg = list(Path(model_path).glob('*.cfg'))
weights = list(Path(model_path).glob('*.weights'))
data = list(Path(model_path).glob('*.data'))
names = list(Path(model_path).glob('names.txt'))

assert len(cfg) > 0, f'Unable to locate {Path(model_path) / "[MODEL].cfg"}'
assert len(weights) > 0, f'Unable to locate {Path(model_path) / "[MODEL].weights"}'
assert len(data) > 0, f'Unable to locate {Path(model_path) / "[MODEL].data"}'
assert len(names) > 0, f'Unable to locate {Path(model_path) / "names.txt"}: Check path in [MODEL].data'

dssp = DatasetSourceProvider()
dssp.add_folder(folder_path=vid_path, dataset_name='playtime')
dssp_tracks = DatasetSourceProvider()
dssp_tracks.add_folder(folder_path=vid_path, dataset_name='tracks')

container = AnnotationContainer(dataset_source_provider=dssp, dataset_version='0.0.1')
container_tracks = AnnotationContainer(dataset_source_provider=dssp_tracks, dataset_version='0.0.1')

dn_path = '/home/anders/huddly/libs/darknet'
with DarknetObjectDetector(Path(dn_path).parents[0] / 'darknet',
                            cfg_path=cfg[0],
                            weights_path=weights[0],
                            data_file_path=data[0],
                            detection_score_threshold=0.3) as od:

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", 0.2, None)
    tracker = Tracker(metric, n_init=5)
    
    i = 0
    first = True
    video_path = Path(vid_path) / 'in'
    max_frame = len(list(Path(video_path).glob('*.jpg')))
    entries = []

    for img_nr in range(max_frame):
        result = []
        image = str(img_nr) + '.jpg'
        frame = cv2.imread(str(video_path / image))
        height, width, ch = frame.shape
        entry = AnnotationEntry(parent=container, image_name=image, image_size=(width, height),
                                dataset_name='playtime', dataset_subset='in')
        entry_tracks = AnnotationEntry(parent=container_tracks, image_name=image, image_size=(width, height),
                                dataset_name='tracks', dataset_subset='in')
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 2:
                continue
            info = {'bbox': track.to_tlwh(), 'id': track.track_id, 'hits': track.hits}
            result.append(info)

        if (i % 22 == 0) or first:
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
                    label=str(t_id),
                    coordinate_mode='ABSOLUTE')
            
            entry_tracks.add_instance(AnnotationInstance(parent=entry_tracks, bbox=bb))
            #bb.overlaid_on_image(frame, (200, 200, 200), draw_label=True, draw_score=False)
            #cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]+b[0]), int(b[3]+b[1])), (255, 0, 0), 2)
            # cv2.putText(frame, str(t_id), (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for inst in pred:
            inst.parent = entry
            inst.bbox.convert_coordinate_mode(target_mode='ABSOLUTE', image_size=entry.image_size)
            entry.add_instance(inst)
        i += 1
        #print(entry.get_image_full_path())
        container.add_entry(entry)
        container_tracks.add_entry(entry_tracks)
        #entry.show()
        for inst in entry:
            inst.bbox.overlaid_on_image(frame, (200, 200, 200), draw_label=True, draw_score=False)

        image = cv2.resize(frame, (1920, 1080))
        cv2.imshow('Pred', image)
        key = cv2.waitKey(20)

        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    
    container.summary()
    container_tracks.summary()
    container.write_json('/home/anders/huddly/projects/autozoom/scoring/scoring-output/' + vid_path.split('/')[-1] + '.json')
    container_tracks.write_json('/home/anders/huddly/projects/autozoom/scoring/scoring-output/' + vid_path.split('/')[-1] + '_tracks.json')
    print('wrote container to /home/anders/huddly/projects/autozoom/scoring/scoring-output/' + vid_path.split('/')[-1] + '.json')
