import copy
import cv2
import numpy as np
from keras.models import load_model
import time

prediction = ''
gesture_names = {0: 'Fist', 1: 'L', 2: 'Okay', 3: 'Palm', 4: 'Peace'}
model = load_model('model/VGG_cross_validated.h5')

def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    return str(np.argmax(pred_array))

cap_region_x_begin = 0.5
cap_region_y_end = 0.8
threshold = 60
blurValue = 41
bgSubThreshold = 100
learningRate = 0

isBgCaptured = 0

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

cap = cv2.VideoCapture(0)
if cap.isOpened():
    bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
    print('Camara Turned On')
else:
    print('Error')
    
while(1):
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)

    frame = cv2.bilateralFilter(frame, 5, 50, 100)

    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),(frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    img = remove_background(frame)

    img = img[0:int(cap_region_y_end * frame.shape[0]),int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)

    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    target = np.stack((thresh,) * 3, axis=-1)
    target = cv2.resize(target, (224, 224))
    target = target.reshape(1, 224, 224, 3)
    prediction = predict_rgb_image_vgg(target)

    cv2.putText(frame,prediction, (120,120), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    cv2.imshow('frame',frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    elif k == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        time.sleep(2)
        isBgCaptured = 1
    
cv2.destroyAllWindows()
cap.release()
