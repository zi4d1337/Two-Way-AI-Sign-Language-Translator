import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import time

WINDOW_NAME = "Two-Handed Data Collection"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

DATA_DIR = './two_hand_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

while True:
    try:
        number_of_classes = int(input("Enter number of classes to collect: "))
        if number_of_classes > 0:
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

while True:
    try:
        dataset_size = int(input(f"Enter number of samples per class (e.g., 100 or 1000): "))
        if dataset_size > 0:
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

for i in range(number_of_classes):
    class_name = input(f'Enter name for class {i+1}: ').lower()
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print(f'\nCollecting data for class: {class_name}')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            continue
        
        frame = cv2.flip(frame, 1)
        H_frame, W_frame, _ = frame.shape
        if H_frame != FRAME_HEIGHT or W_frame != FRAME_WIDTH:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)

        cv2.putText(frame, f'Ready? Press "S" to start collecting for "{class_name}"', (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
            if win_w <= 0 or win_h <= 0: raise Exception("Window minimized")
            scale = min(win_w / FRAME_WIDTH, win_h / FRAME_HEIGHT)
            new_w, new_h = int(FRAME_WIDTH * scale), int(FRAME_HEIGHT * scale)
            x_offset, y_offset = (win_w - new_w) // 2, (win_h - new_h) // 2
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame
            cv2.imshow(WINDOW_NAME, canvas)
        except Exception:
            if 'frame' in locals():
                cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('s'):
            print("Starting countdown...")
            
            for i in range(5, 0, -1):
                ret, frame = cap.read()
                if not ret: continue
                
                frame = cv2.flip(frame, 1)
                H_frame, W_frame, _ = frame.shape
                if H_frame != FRAME_HEIGHT or W_frame != FRAME_WIDTH:
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)

                text = f"Starting in {i}..."
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                text_x = (FRAME_WIDTH - text_width) // 2
                text_y = (FRAME_HEIGHT + text_height) // 2
                
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)

                try:
                    _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
                    if win_w <= 0 or win_h <= 0: raise Exception("Window minimized")
                    scale = min(win_w / FRAME_WIDTH, win_h / FRAME_HEIGHT)
                    new_w, new_h = int(FRAME_WIDTH * scale), int(FRAME_HEIGHT * scale)
                    x_offset, y_offset = (win_w - new_w) // 2, (win_h - new_h) // 2
                    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame
                    cv2.imshow(WINDOW_NAME, canvas)
                except Exception:
                    if 'frame' in locals():
                        cv2.imshow(WINDOW_NAME, frame)
                
                cv2.waitKey(1000)
            
            break
        elif key == ord('q'):
            print("Quitting data collection.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print("Starting data collection...")
    sample_num = 0
    while sample_num < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        frame = cv2.flip(frame, 1)
        H_frame, W_frame, _ = frame.shape
        if H_frame != FRAME_HEIGHT or W_frame != FRAME_WIDTH:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        
        frame_with_landmarks = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        data_aux = np.zeros(42 * 2) 

        if results.multi_hand_landmarks:
            for i_hand, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i_hand].classification[0].label
                x_, y_ = [], []
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                min_x, min_y = min(x_), min(y_)
                normalized_landmarks = []
                for landmark in hand_landmarks.landmark:
                    normalized_landmarks.append(landmark.x - min_x)
                    normalized_landmarks.append(landmark.y - min_y)
                if handedness == 'Left':
                    data_aux[0:42] = normalized_landmarks
                elif handedness == 'Right':
                    data_aux[42:84] = normalized_landmarks
                
                mp_drawing.draw_landmarks(
                    frame_with_landmarks, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                )

            filename = os.path.join(class_path, f'{sample_num}.pickle')
            with open(filename, 'wb') as f:
                pickle.dump(data_aux, f)
                            
            cv2.putText(frame_with_landmarks, f'Collecting sample {sample_num+1}/{dataset_size}', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            sample_num += 1

        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
            if win_w <= 0 or win_h <= 0: raise Exception("Window minimized")
            scale = min(win_w / FRAME_WIDTH, win_h / FRAME_HEIGHT)
            new_w, new_h = int(FRAME_WIDTH * scale), int(FRAME_HEIGHT * scale)
            x_offset, y_offset = (win_w - new_w) // 2, (win_h - new_h) // 2
            resized_frame = cv2.resize(frame_with_landmarks, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame
            cv2.imshow(WINDOW_NAME, canvas)
        except Exception:
            if 'frame_with_landmarks' in locals():
                cv2.imshow(WINDOW_NAME, frame_with_landmarks)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("Data collection complete.")
cap.release()
cv2.destroyAllWindows()