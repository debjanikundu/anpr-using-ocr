# import necessary libs
import uvicorn
import asyncio
import cv2
import easyocr
import numpy as np
from vidgear.gears.asyncio import WebGear
from vidgear.gears.asyncio.helper import reducer
import torch
import time
import re
import numpy as np
import easyocr
import os  

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


#function to run detection
def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates


# to plot the boxes and coordinates
def plot_boxes(results, frame, classes):

    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels

    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")

    #looping for detection
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]
            coords = [x1,y1,x2,y2]

            plate_num = recognize_plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)


            # if text_d == 'mask':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0), 4)
        return frame


#function to recognise number plate
def recognize_plate_easyocr(img,coords,reader,region_threshold):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    # nplate = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image

    ocr_result = reader.readtext(nplate)

    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)

    if len(text) ==1:
        text = text[0].upper()
    return text




### to filter out wrong detections 

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate


if __name__ == "__main__" :
    
    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    model =  torch.hub.load('ultralytics/yolov5', 'custom', path='models\\best.pt',force_reload=True) ## if you want to download the git repo and then run the detection

    classes = model.names ### class names in string format

    EASY_OCR = easyocr.Reader(['en']) ### initiating easyocr
    OCR_TH = 0.2

    options = {
        "custom_data_location" : "./",
        "frame_size_reduction" : 85,
        "jpeg_compression_quality" : 65,
        "jpeg_compression_fastdct" : True,
        "jpeg_compression_fastupsample" : False,
        "image_stabilization" : True,
        "crf" : 35,
        "iso" : 400,
        "exposure_compensation" : 55,
        "awb_mode" : "horizon",
        "video_source" : "assets\\video1.mp4"
    }


    # initialize WebGear app without any source
    web = WebGear(logging=True, stabilize=True, **options)

    # create your own custom frame producer
    async def my_frame_producer():

        # !!! define your own video source here !!!
        # Open any video stream such as live webcam 
        # video stream on first index(i.e. 0) device
        stream = cv2.VideoCapture(options["video_source"])
        
        # loop over frames
        while True:
            # read frame from provided source
            (grabbed, frame) = stream.read()
            # break if NoneType
            if not grabbed:
                break

            # do something with your OpenCV frame here
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detectx(frame, model = model)
            frame = plot_boxes(results, frame, classes = classes)
            
            # reducer frames size if you want more performance otherwise comment this line
            frame = await reducer(frame, percentage=30, interpolation=cv2.INTER_AREA)  # reduce frame by 30%
            # handle JPEG encoding
            encodedImage = cv2.imencode(".jpg", frame)[1].tobytes()
            # yield frame in byte format
            yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + encodedImage + b"\r\n")

            await asyncio.sleep(0.000001)

        # close stream
        stream.release()


    # add your custom frame producer to config
    web.config["generator"] = my_frame_producer

    # run this app on Uvicorn server at address http://localhost:8000/
    uvicorn.run(web(), host="localhost", port=8000)

    # close app safely
    web.shutdown()
