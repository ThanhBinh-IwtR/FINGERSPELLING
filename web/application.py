#import libraries

import time
from flask import Flask, render_template,Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# web_detect_dir = os.path.join(current_dir, 'web_detect')
sys.path.insert(0, "D:/AI/DPLS2L/S2L/Project")

import style_mediapipe
import Transformers_Landmark
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# ---------------------------------- init ---------------------------------- 
model = Transformers_Landmark.get_model()
model.load_weights('./Project/weight/model.h5')
start = Transformers_Landmark.predict(model,np.load('./Project/statics/start_up.npy'))
preprocess = Transformers_Landmark.PreprocessLayer()
all_res = []
n_frame = 0
imp_frame = 0
fps = 0
pc = 0

print('Completed model loading !')
# ---------------------------------- init ---------------------------------- 

ALLOWED_EXTENSIONS = {'mp4', 'avi'}

app = Flask(__name__, static_folder='assets')
UPLOAD_FOLDER = os.path.join(os.getcwd(), "web/assets/upload")
ALLOWED_EXTENSIONS = {'mp4', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

current_result = ""


def reset_all(res,sequence_arr):
    res += Transformers_Landmark.predict(model,sequence_arr)
    sequence=[]
    n_frame = 0
    return res,n_frame,sequence
def put_text(frame, fps , n_frame):
    cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1]-200,frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Pc: {int((n_frame/128)*100)}%", (frame.shape[1]-400,frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    return frame


def reset_all_import(imp_res,imp_sequence_arr):
    imp_res += Transformers_Landmark.predict(model,imp_sequence_arr)
    imp_sequence=[]
    imp_frame = 0
    return imp_res,imp_frame,imp_sequence

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_face_mesh = mp.solutions.face_mesh 
face_mesh = mp_face_mesh.FaceMesh() 

camera = cv2.VideoCapture(0)
def generate_frames():
    global all_res
    global n_frame
    global current_result
    global fps
    global pc

    res = ''
    num_frame_space = 10
    sequence =[]
    prev_time = 0
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                current_time = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (current_time - prev_time)
                prev_time = current_time
                n_frame +=1

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_holistic = holistic.process(frame_rgb)
                #Draw keypoint
                results_face_mesh = face_mesh.process(frame_rgb)
                style_mediapipe.draw_styled_landmarks(frame, results_holistic, results_face_mesh)
                #Preprocess data
                keypoints = style_mediapipe.extract_keypoints(results_holistic)
                sequence.append(keypoints)
                sequence_arr = np.array(sequence)
                if len(sequence) > num_frame_space and np.all(sequence_arr[n_frame-num_frame_space:n_frame,:84] == 0):
                    sequence_arr =  preprocess(sequence_arr).numpy()
                    res, n_frame,sequence  = reset_all(res,sequence_arr)
                    if  '2022 1020' in res:
                        res = ''
                    all_res.append(res)
                    res = ''
                    if (len(all_res) != 0) and (all_res[-1] !=  '. ') and (all_res[-1] != '') :
                        all_res.append('. ')

                    print('The result is : ',all_res)
                    print('RESET')
                    cv2.putText(frame, f"--", (1,20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                elif len(sequence) == 128:
                    res, n_frame,sequence  = reset_all(res,sequence_arr)
                    print('after 128: ',res)
                    all_res.append(res)
                    res = ''
                # frame = put_text(frame,fps,n_frame)
                cv2.waitKey(10)
                pc = int((n_frame/128)*100)
                success, buffer = cv2.imencode('.jpg', frame)
                if not success:
                    break
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def main():
    return render_template("home.html")

@app.route("/detect")
def detect():
    return render_template("detect.html", current_result=current_result)

@app.route('/get_latest_result')
def get_latest_result():
    global all_res
    result = ''.join(all_res)
    return jsonify({'result': result})

@app.route('/get_FPS_PC')
def get_FPS_PC():
    global fps
    global pc
    
    return jsonify({'fps': round(fps,0), 'pc':round(pc/100,1)})

@app.route('/clear_all_res', methods=['POST'])
def clear_all_res():
    global all_res
    all_res = []
    return jsonify({'result': 'success'})

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'result': 'failure', 'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'result': 'failure', 'error': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            res = generate_processed_frames('web/assets/upload/' + file.filename)
            return jsonify({'result': 'success', 'filename': filename,"data": res})
        return jsonify({'result': 'failure', 'error': 'File not allowed'})
    except Exception as e:
        return jsonify({'result': 'failure', 'error': str(e)})


def generate_processed_frames(filename,num_frame_space = 20):
    cap = cv2.VideoCapture(filename)
    global import_res
    global import_frame
    global current_result
    imp_res = ''
    all_res = []
    imp_frame = 0
    # num_frame_space = 30
    imp_sequence =[]
    if not cap.isOpened():
        print("Không thể mở video.")
        exit()
        # return 
    print('========================RUNING==================================')
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            imp_frame +=1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            #Draw keypoint
            results_face_mesh = face_mesh.process(frame_rgb)
            style_mediapipe.draw_styled_landmarks(frame, results, results_face_mesh)

            keypoints = style_mediapipe.extract_keypoints(results)
            imp_sequence.append(keypoints)
            imp_sequence_arr = np.array(imp_sequence)
            if len(imp_sequence) > num_frame_space and np.all(imp_sequence_arr[imp_frame-num_frame_space:imp_frame,:84] == 0):
                imp_sequence_arr =  preprocess(imp_sequence_arr).numpy()
                imp_res, imp_frame,imp_sequence  = reset_all_import(imp_res,imp_sequence_arr)
                if  '2022 1020' in imp_res:
                    imp_res = ''
                all_res.append(imp_res)
                imp_res = ''
                if (len(all_res) != 0) and (all_res[-1] !=  '. ') and (all_res[-1] != '') :
                    all_res.append('. ')
                current_result = ''.join(all_res)
                print('Ket qua la : ',all_res)
                print('RESET')
            elif len(imp_sequence) == 128:
                imp_res, imp_frame,imp_sequence  = reset_all_import(imp_res,imp_sequence_arr)
                print('after 128: ',all_res)
        if len(imp_sequence) < 128:
            imp_sequence_arr = preprocess(imp_sequence_arr).numpy()
            imp_res += Transformers_Landmark.predict(model,imp_sequence_arr)
        imp_res += '.'
        all_res.append(imp_res)
        cap.release()
        cv2.destroyAllWindows()
        print('===Done===')
        return all_res




@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    app.run( debug=False)