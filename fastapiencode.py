import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


path = r'F:\xampp\htdocs\video\Images_Attendance'
images = []
classNames = []
encodeListKnown = []


def encode_images():
    global encodeListKnown, images, classNames
    images = []
    classNames = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    encodeListKnown = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeListKnown.append(encode)


encode_images()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        video = request.files['video']
        video_file = video.read()
        np_video = np.frombuffer(video_file, np.uint8)
        with open('temp.mp4', 'wb') as f:
            f.write(np_video)

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
                    response = jsonify({'id': name})
                    response.headers.add("Access-Control-Allow-Origin", "*")
                    response.headers.add("Access-Control-Allow-Headers", "*")
                    response.headers.add("Access-Control-Allow-Methods", "*")
                    return response

    except Exception as e:
        return jsonify({"Error => ": str(e)})

@app.route("/add", methods=["POST"])
def add():
    try:
        image = request.files['image']
        id = request.form.get('id')
        image1 = image.read()
        tid = str(id)

        # Check if file with given filename already exists
        if os.path.exists(os.path.join(path, f"{tid}.jpg")):
            return jsonify({'error': f"Image with id '{tid}' already exists."})

        with open(os.path.join(path, f"{tid}.jpg"), 'wb') as f:
            f.write(image1)

        # Re-encode images to include newly added image
        encode_images()

        return jsonify({'status': 'success'})

    except Exception as e:
        return jsonify({"Error => ": str(e)})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    
