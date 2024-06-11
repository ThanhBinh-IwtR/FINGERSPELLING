import cv2
from style_mediapipe import  draw_styled_landmarks,extract_keypoints
import mediapipe as mp
import numpy as np
from Transformers_Landmark import get_model,predict, PreprocessLayer
model = get_model()
model.load_weights('./project/weight/model.h5')
# Khởi tạo Holistic
start = predict(model,np.load('./project/statics/start_up.npy'))
preprocess = PreprocessLayer()
sequence =  []
all_res = []
res = ''
n_frame = 0
print('Completed model loading !')
def reset_all(res,sequence_arr):
    res += predict(model,sequence_arr)
    sequence=[]
    n_frame = 0
    return res,n_frame,sequence
# Đường dẫn đến tệp video
def predict_video(file_path,num_frame_space = 30):
    cap = cv2.VideoCapture(file_path)
    sequence =  []
    all_res = []
    res = ''
    n_frame = 0
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
            n_frame +=1
            # Dùng Holistic để xử lý frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence_arr = np.array(sequence)
            if len(sequence) > num_frame_space and np.all(sequence_arr[n_frame-num_frame_space:n_frame,:84] == 0):
                sequence_arr = preprocess(sequence_arr).numpy()
                __, n_frame,sequence  = reset_all(res,sequence_arr)
                res += '. '
                print(res)
                all_res.append(res)
                res = ''
                print('RESET')
                # if all_res[-1][-2:] ==  '. ':
                #     all_res[-1][-2:] = ''
                yield ''.join(all_res)
            elif len(sequence) == 128:
                res, n_frame,sequence = reset_all(res,sequence_arr)
                print(res)
        if len(sequence) < 128:
            sequence_arr =  preprocess(sequence_arr).numpy()
            res += predict(model,sequence_arr)
        res += '.'
        all_res.append(res)
        cap.release()
        cv2.destroyAllWindows()
        print('===Xon=g')
        yield ''.join(all_res)
# for res in predict_video("D:/AI/DPLS2L/S2L/Project/18s.mp4"):
#     print('DAY LA YEILD KET QUA : ', res)   