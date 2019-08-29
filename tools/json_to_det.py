import sys
import json
import os

filename=sys.argv[1]

detections=[]

label="head" # person

with open(filename,encoding='utf-8-sig') as json_file:
    json_data = json.load(json_file)
    #count=0
    for images in json_data["entries"]:
        #print(count)
        width=images["image_size"]["width"]
        height=images["image_size"]["height"]
        id = os.path.splitext(os.path.basename(images["image_path"]))[0]
        #print("ID ",id)
        for inst in images["instances"]:
            box=inst["bbox"]
            if box["label"] == label:
                left=box["xmin"]*width
                top=box["ymin"]*height
                w=(box["xmax"]-box["xmin"])*width
                h=(box["ymax"]-box["ymin"])*height
                detections.append([int(id),"-1",left,top,w,h,box["score"],"-1","-1","-1"])
                #detections[id].append(detection(id,left,top,w,h,score))
                #print(count,",-1,",left,",",top,",",w,",",h,",",box["score"],",-1,-1,-1")
        #count+=1
                #print(box["score"])
for i in sorted(detections,key=lambda x: x[0]):
#for x in detections.sort(key=lambda x: x[1]):
#myString = ','.join(map(str, myList))
    print(','.join(map(str,i)))
#for i in
#    print(i)
