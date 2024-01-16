import cv2
import mediapipe as mp
import numpy as np
import time
from style_mediapipe import  draw_styled_landmarks,extract_keypoints
from Transformers_Landmark import get_model,predict, PreprocessLayer

def put_text(frame,fps):
    cv2.putText(frame, ' '.join(res), (3,30), 
                
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 200, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame
def reset_all(res):
    res += predict(model,sequence_arr)  + ' '
    if res == '2022 1020 ':
        res = 'None'
    sequence=[]
    print('====================>> ',res)
    n_frame = 0
    return res,n_frame,sequence
model = get_model()
model.load_weights('./Project/weight/model.h5')
start = predict(model,np.load('./Project/statics/start_up.npy'))
preprocess = PreprocessLayer()
# Khởi tạo Holistic
cap = cv2.VideoCapture('./Project/dinhcao5.mp4')
print('Completed model loading !')
n_test = 0
sequence =  []
res = ''
res_seq = []
num_frame_space = 20
n_frame = 0
prev_time = 0
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            continue
        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - prev_time)
        prev_time = current_time
        n_frame +=1
        n_test +=1
        # Chuyển đổi frame sang định dạng màu BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Dùng Holistic để xử lý frame
        results = holistic.process(frame_rgb)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence_arr = np.array(sequence)
        if len(sequence) > num_frame_space and np.all(sequence_arr[n_frame-num_frame_space:n_frame,:84] == 0):
            sequence_arr =  preprocess(sequence_arr).numpy()
            __, n_frame,sequence  = reset_all(res)
            res = ''
            cv2.putText(frame, f"Reset", (frame.shape[1] - 400, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.waitKey(1)
            print('Reset')
        if len(sequence) == 128:
            res, n_frame,sequence = reset_all(res)
        # Vẽ các landmark trên cơ thể và khuôn mặt
        # draw_styled_landmarks(frame,results)
        frame = put_text(frame,fps)
        if n_frame % 32 == 0:
            print('Percent',(n_frame/128)*100 ,'%')
        cv2.imshow('Holistic', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(f'n_frame:{n_frame} - n_test:{n_test}')
    cap.release()
    cv2.destroyAllWindows()
print(res)
    
