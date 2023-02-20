import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

global encodeListKnown

encodeListKnown = []

print("encoding.................")
path = r'/home/ubuntu/try/Images_Attendance/'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    encodeListKnown.append(encode)
print("image encoded")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    print("Predict was hit")
    if request.method == "GET":
        return jsonify({"method": "GET"})
    try:
        print("in try defffffffffffffff")
        video = request.files['video']
        video_file = video.read()
        np_video = np.frombuffer(video_file, np.uint8)
        with open('temp.mp4', 'wb') as f:
            f.write(np_video)
        print("temp video written")
        # Initialize the video capture process
        video = cv2.VideoCapture('temp.mp4')
        attendance = []

        while True:
            success, img = video.read()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(img)
            encodesCurFrame = face_recognition.face_encodings(img, facesCurFrame)
            
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                
                matchIndex = np.argmin(faceDis)
                
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    video.release()
                    print("face matched")
                    response = jsonify({'id': name})
                    response.headers.add("Access-Control-Allow-Origin", "*")
                    response.headers.add("Access-Control-Allow-Headers", "*")
                    response.headers.add("Access-Control-Allow-Methods", "*")
                    print("sending resposnse", response)
                    return response

        
    except Exception as e:
        return jsonify({"Error => ": str(e)})

@app.route("/add", methods=["POST"])
def add():
    try:
        image = request.files['image']
        id = request.form.get('id')
        image1 = image.read()
        tid=str(id)
        with open("/home/ubuntu/try/Images_Attendance/"+tid+".jpg", 'wb') as f:
            f.write(image1)
        path = r'/home/ubuntu/try/Images_Attendance/'
        images = []
        classNames = []
        myList = os.listdir(path)
        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0]) 

        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeListKnown.append(encode)
            print("image encoded")
        return jsonify({'status': 'success'})
       
    except Exception as e:
        return jsonify({"Error => ": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
