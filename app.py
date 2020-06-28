import numpy as np
from flask import Flask, jsonify, render_template, request
import cv2
from keras.models import load_model

def x_cord_contour(contour):
    M=cv2.moments(contour)
    return int(M['m10']/M['m00'])

def make_square(not_square):
    BLACK=[0,0,0]
    img_dim=not_square.shape
    height=img_dim[0]
    width=img_dim[1]
    if width==height:
        return not_square
    else:
        width=2*width
        height=2*height
        double=cv2.resize(not_square,(width,height),interpolation=cv2.INTER_CUBIC)
        if height>width:
            pad=int((height-width)/2)
            square=cv2.copyMakeBorder(double,0,0,pad,pad,cv2.BORDER_CONSTANT, value=BLACK)
        else:
            pad=int((width-height)/2)
            square = cv2.copyMakeBorder(double,pad, pad,0,0, cv2.BORDER_CONSTANT, value=BLACK)

    return square

def resize_to_pixel(dimention,img):
    resized=cv2.resize(img,(dimention,dimention),interpolation=cv2.INTER_AREA)
    return resized

app= Flask(__name__)


@app.route('/')
def home():
    return  render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name=request.form['image']
    img=cv2.imread(name)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 50)
    contors, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contors = sorted(contors, key=x_cord_contour, reverse=False)
    number = 0
    classifier = load_model('digit_recognition_model.h5')
    for c in contors:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 5 and h > 25:
            roi = blurred[y - 15:y + h + 15, x - 10:x + w + 10]

            ret, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY_INV)
            roi = make_square(roi)
            roi = resize_to_pixel(28, roi)


            roi = roi / 255
            roi = roi.reshape(1, 28, 28, 1)


            res = classifier.predict_classes(roi, 1, verbose=0)[0]
            number = number * 10 + res
            #cv2.putText(img, str(res), (x, y + h + 60), cv2.FONT_HERSHEY_COMPLEX, 2, (250, 0, 255), 2)
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 240, 250), 2)

    return render_template('index.html', prediction_text=" Predicted Number is : {}".format(number))

if __name__=="__main__":
    app.run(debug=True)
