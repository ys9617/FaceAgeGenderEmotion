[image1]: ./img/result2.png "result1"
[image2]: ./img/result.gif "result"

# Age, Gender and Emotion recognition

Detect front-face position and recognizing age, gender and emotion.

![RESULT][image1]


## Overview

1. Capturing Video from Webcam
2. Detecting Front-face Position
3. Recognizing Age and Gender
4. Recognizing Emotion
5. Result
6. Reference


## Capturing Video from Webcam

```python
import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    flag, frame = cap.read()

    if not flag:
        break

    key_pressed = cv2.waitKey(60)

    '''
    main part
    '''

    if key_pressed == 27:
        break


cap.release()
cv2.destroyAllWindows()
```


## Detecting Front-face Position

For detecting front-face position, OpenVINO pre-trained object detection model 'face-detection-012' is used.
'face-detection-012' model is based on MobileNetV2 as a backbone with a multiple SSD head for indoor/outdoor scenes shot by a front-facing camera.  a backbone with a multiple SSD head for indoor/outdoor scenes shot by a front-facing camera.


```python
# load face detection model
fd_model_name = "face-detection-0102"
fd_model_bin = os.path.join(model_path, fd_model_name, "FP32", fd_model_name) + ".bin"
fd_model_xml = os.path.join(model_path, fd_model_name, "FP32", fd_model_name) + ".xml"

fd_net = ie.read_network(model=fd_model_xml, weights=fd_model_bin)
fd_exec_net = ie.load_network(fd_net, "CPU")

fd_input_blob = next(iter(fd_net.inputs))

...

    # detect front-face position
    preprocessed_frame = preprocess(frame, 384, 384)
    fd_exec_net.start_async(request_id=0, inputs={fd_input_blob: preprocessed_frame})

    if fd_exec_net.requests[0].wait(-1) == 0:
        output = fd_exec_net.requests[0].outputs['detection_out']

        # detecting maximum 5 faces
        for i in range(0,5):
            box = output[0][0][i]
            conf = box[2]
            
            if conf >= 0.9:
                xmin = int(box[3] * w)
                ymin = int(box[4] * h)
                xmax = int(box[5] * w)
                ymax = int(box[6] * h)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 3)
```


## Recognizing Age and Gender

For recognizing age and gender, OpenVINO pre-trained objec recognition model 'age-gender-recognition-retail-0013' is used. 'age-gender-recognition-retail-0013' is able to recognize age of people in [18, 75] years old range.


```python
# load age & gender recognition model
ag_model_name = "age-gender-recognition-retail-0013"
ag_model_bin = os.path.join(model_path, ag_model_name, "FP32", ag_model_name) + ".bin"
ag_model_xml = os.path.join(model_path, ag_model_name, "FP32", ag_model_name) + ".xml"

ag_net = ie.read_network(model=ag_model_xml, weights=ag_model_bin)
ag_exec_net = ie.load_network(ag_net, "CPU")

ag_input_blob = next(iter(ag_net.inputs))

...


            if conf >= 0.9:
                ...

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 3)

                roi = frame[ymin:ymax, xmin:xmax, :]

                # recognizing age and gender
                age, gender = ag_recognition(roi, ag_exec_net, ag_input_blob)

```


## Recognizing Emotion

For recognizing emotion,  OpenVINO pre-trained objec recognition model 'emotions-recognition-retail-0003' is used. 'emotions-recognition-retail-0003' is able to recognize 5 emotions ('neutral', 'happy', 'sad', 'surprise', 'anger').

```python
# load emotion recognition model
em_model_name = "emotions-recognition-retail-0003"
em_model_bin = os.path.join(model_path, em_model_name, "FP32", em_model_name) + ".bin"
em_model_xml = os.path.join(model_path, em_model_name, "FP32", em_model_name) + ".xml"

em_net = ie.read_network(model=em_model_xml, weights=em_model_bin)
em_exec_net = ie.load_network(em_net, "CPU")

em_input_blob = next(iter(em_net.inputs))

...

            if conf >= 0.9:
                ...

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 3)

                roi = frame[ymin:ymax, xmin:xmax, :]

                # recognizing age and gender
                age, gender = ag_recognition(roi, ag_exec_net, ag_input_blob)
                
                # recognizing emotion
                emotion = em_recognition(roi, em_exec_net, em_input_blob)
```


## Result


![RESULT][image2]



## Reference

* OpenVINO pre-trained front-face detection model [face-detection-012](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_0102_description_face_detection_0102.html)

* OpenVINO pre-trained age-gender recognition model [age-gender-recognition-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_age_gender_recognition_retail_0013_description_age_gender_recognition_retail_0013.html)

* OpenVINO pre-trained emotion recognition model [emotions-recognition-retail-0003](https://docs.openvinotoolkit.org/latest/_models_intel_emotions_recognition_retail_0003_description_emotions_recognition_retail_0003.html)




