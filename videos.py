import cv2
import sys
import numpy as np 
import warnings
import os
from pathlib import Path
import argparse
from tqdm import tqdm

from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from deep_sort.detection import Detection
from face_recognition import face_encodings
from bbox import AnnotationInstance, AnnotationEntry, AnnotationContainer, DatasetSourceProvider
from bbox.instances import BBox
from bbox.contrib.detection.darknet import DarknetObjectDetector

width = 0
height = 0

warnings.filterwarnings('ignore')

def get_darknet_options(model_path):

    cfg = list(Path(model_path).glob('*.cfg'))
    weights = list(Path(model_path).glob('*.weights'))
    data = list(Path(model_path).glob('*.data'))
    names = list(Path(model_path).glob('names.txt'))

    assert len(cfg) > 0, f'Unable to locate {Path(model_path) / "[MODEL].cfg"}'
    assert len(weights) > 0, f'Unable to locate {Path(model_path) / "[MODEL].weights"}'
    assert len(data) > 0, f'Unable to locate {Path(model_path) / "[MODEL].data"}'
    assert len(names) > 0, f'Unable to locate {Path(model_path) / "names.txt"}: Check path in [MODEL].data'

    return cfg, weights, data, names


def make_containers(vid_path):

    dssp = DatasetSourceProvider()
    dssp.add_folder(folder_path=vid_path, dataset_name='playtime')
    dssp_tracks = DatasetSourceProvider()
    dssp_tracks.add_folder(folder_path=vid_path, dataset_name='tracks')

    container = AnnotationContainer(dataset_source_provider=dssp, dataset_version='0.0.1')
    container_tracks = AnnotationContainer(dataset_source_provider=dssp_tracks, dataset_version='0.0.1')

    return container, container_tracks

def add_tracks(entry, info):
    b = info['bbox']
    t_id = info['id']
    bb = BBox(xmin=np.clip(int(b[0]), 0, width), ymin=np.clip(int(b[1]), 0, height), xmax=int(b[0]+b[2]), ymax=int(b[1]+b[3]),
            label=str(t_id),
            coordinate_mode='ABSOLUTE')
    
    entry.add_instance(AnnotationInstance(parent=entry, bbox=bb))
    return bb, entry

def add_instance(entry, instance):
    instance.parent = entry
    instance.bbox.convert_coordinate_mode(target_mode='ABSOLUTE', image_size=entry.image_size)
    entry.add_instance(instance)
    
    return instance, entry

def get_tracker(n_init=4, max_cosine_distance=0.2):
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", 0.2, None)
    tracker = Tracker(metric, n_init=4)
    return tracker

def get_track_info(track):
    return {'bbox': track.to_tlwh(), 'id': track.track_id, 'hits': track.hits}

def draw(frame):
    image = cv2.resize(frame, (1920, 1080))
    cv2.imshow('Pred', image)
    return cv2.waitKey(1)

def process_image(tracker, img_nr, video_path, container, container_tracks):
    global height
    global width
    result = []
    image = str(img_nr) + '.jpg'
    frame = cv2.imread(str(video_path / image))
    height, width, _ = frame.shape
    entry = AnnotationEntry(parent=container, image_name=image, image_size=(width, height),
                            dataset_name='playtime', dataset_subset='in')
    entry_tracks = AnnotationEntry(parent=container_tracks, image_name=image, image_size=(width, height),
                            dataset_name='tracks', dataset_subset='in')

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 2:
            continue  
        result.append(get_track_info(track))
    
    return frame, entry, entry_tracks, result

def update_detection_and_tracker(frame, tracker, od):
    height, width, ch = frame.shape
    pred = od.detect_image(frame)
    detections = []
    for inst in pred:
        if inst.label == 'head':
            crop = frame[int(inst.ymin*height):int(inst.ymax*height), int(inst.xmin*width):int(inst.xmax*width)]
            feat = face_encodings(crop)
            detections.append(Detection((int(inst.xmin*width), int(inst.ymin*height), int((inst.xmax-inst.xmin)*width),
                                        int((inst.ymax-inst.ymin)*height)),
                                        inst.score,
                                        list(feat[0]) if len(feat) > 0 else [1]*128))
    tracker.predict()
    tracker.update(detections)

    return tracker, pred

def main(args):
    vid_path = args.VIDEO_PATH
    container, container_tracks = make_containers(vid_path)

    dn_path = args.DARKNET_PATH
    show = args.show

    cfg, weights, data, names = get_darknet_options(args.MODEL)
    with DarknetObjectDetector(
                            Path(dn_path).parents[0] / 'darknet',
                            cfg_path=cfg[0],
                            weights_path=weights[0],
                            data_file_path=data[0],
                            detection_score_threshold=0.4) as od:
        i = 0
        first = True
        video_path = Path(vid_path) / 'in'
        max_frame = len(list(Path(video_path).glob('*.jpg')))
        tracker = get_tracker(n_init=4)

        for img_nr in tqdm(range(max_frame)):
            frame, entry, entry_tracks, result = process_image(tracker, img_nr, video_path, container, container_tracks)

            if True:
                i = 0
                tracker, pred = update_detection_and_tracker(frame, tracker, od)
                first = False
            
            for inf in result:
                bb, entry_tracks = add_tracks(entry_tracks, inf)
                if show:
                    bb.overlaid_on_image(frame, (200, 200, 200), draw_label=True, draw_score=False)

            for inst in pred:
                instance, entry = add_instance(entry, inst)
                if show:
                   pass #instance.bbox.overlaid_on_image(frame, (200, 0, 200), draw_label=True, draw_score=False)

            container.add_entry(entry)
            container_tracks.add_entry(entry_tracks)

            if show: 
                key = draw(frame)
                if key == ord('q'):
                    break
            
            i += 1

    cv2.destroyAllWindows()
    container.summary()
    container_tracks.summary()
    container.write_json(os.path.join(args.OUTPUT, vid_path.split('/')[-1] + '.json'))
    container_tracks.to_file(os.path.join(args.OUTPUT, vid_path.split('/')[-1] + '_tracks.bbox'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run detection and tracking on scoring video')
    parser.add_argument('MODEL', help='path to darknet model')
    parser.add_argument('VIDEO_PATH', help='Path to "in" folder for video')
    parser.add_argument('DARKNET_PATH', help='Path to darknet folder')
    parser.add_argument('OUTPUT', help='output path')
    parser.add_argument('-s', '--show', help='display detections and tracks on video', action='store_true')
    args = parser.parse_args()
    main(args)
