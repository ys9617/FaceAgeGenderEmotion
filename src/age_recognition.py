import cv2
import numpy as np
import os

from openvino.inference_engine import IENetwork, IECore

model_path = "./models/intel"

def preprocess(input_img, width, height):
    preprocessed_img = np.copy(input_img)
    preprocessed_img = cv2.resize(preprocessed_img, (height, width))
    preprocessed_img = preprocessed_img.transpose((2,0,1))
    preprocessed_img = preprocessed_img.reshape(1, 3, height, width)

    return preprocessed_img
    
def ag_recognition(input_roi, exec_net, input_blob):
    # recognizing age and gender
    preprocessed_roi = preprocess(input_roi, 62, 62)

    output = exec_net.infer({input_blob: preprocessed_roi})

    gender_list = ['female', 'male']

    age = int(output['age_conv3'][0][0][0][0] * 100)
    gender = gender_list[np.argmax(output['prob'][0,:,0,0])]

    return age, gender

def em_recognition(input_roi, exec_net, input_blob):
    # recognizing emotion
    preprocessed_roi = preprocess(input_roi, 64, 64)

    output = exec_net.infer({input_blob: preprocessed_roi})

    emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']

    return emotions[np.argmax(output['prob_emotion'][0,:,0,0])]


# initialize inference engine
ie = IECore()

# load face detection model
fd_model_name = "face-detection-0102"
fd_model_bin = os.path.join(model_path, fd_model_name, "FP32", fd_model_name) + ".bin"
fd_model_xml = os.path.join(model_path, fd_model_name, "FP32", fd_model_name) + ".xml"

fd_net = ie.read_network(model=fd_model_xml, weights=fd_model_bin)
fd_exec_net = ie.load_network(fd_net, "CPU")

fd_input_blob = next(iter(fd_net.inputs))

# load age & gender recognition model
ag_model_name = "age-gender-recognition-retail-0013"
ag_model_bin = os.path.join(model_path, ag_model_name, "FP32", ag_model_name) + ".bin"
ag_model_xml = os.path.join(model_path, ag_model_name, "FP32", ag_model_name) + ".xml"

ag_net = ie.read_network(model=ag_model_xml, weights=ag_model_bin)
ag_exec_net = ie.load_network(ag_net, "CPU")

ag_input_blob = next(iter(ag_net.inputs))

# load emotion recognition model
em_model_name = "emotions-recognition-retail-0003"
em_model_bin = os.path.join(model_path, em_model_name, "FP32", em_model_name) + ".bin"
em_model_xml = os.path.join(model_path, em_model_name, "FP32", em_model_name) + ".xml"

em_net = ie.read_network(model=em_model_xml, weights=em_model_bin)
em_exec_net = ie.load_network(em_net, "CPU")

em_input_blob = next(iter(em_net.inputs))


# video capture
cap = cv2.VideoCapture(0)
w = int(cap.get(3))
h = int(cap.get(4))

while cap.isOpened():
    flag, frame = cap.read()

    if not flag:
        break

    key_pressed = cv2.waitKey(60)

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

                roi = frame[ymin:ymax, xmin:xmax, :]

                # recognizing age and gender
                age, gender = ag_recognition(roi, ag_exec_net, ag_input_blob)
                
                # recognizing emotion
                emotion = em_recognition(roi, em_exec_net, em_input_blob)

                cv2.putText(frame, "age:"+str(age), (xmax+5,ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "gender:"+str(gender), (xmax+5,ymin+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "emotion:"+str(emotion), (xmax+5,ymin+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


    cv2.imshow('frame', frame)

    if key_pressed == 27:
        break


cap.release()
cv2.destroyAllWindows()






























