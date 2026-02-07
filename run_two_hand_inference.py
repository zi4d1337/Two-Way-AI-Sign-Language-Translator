import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time

WINDOW_NAME = "Two-Handed Sign Recognition"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CONFIDENCE_THRESHOLD = 0.75 

def wrap_text(text, font, font_scale, thickness, max_width):
    """Wraps text to fit within a specified width."""
    lines = []
    words = text.split(' ')
    current_line = ""
    
    for word in words:
        test_line = current_line + word + " "
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        
        if text_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word + " "
            
    lines.append(current_line.strip())
    return lines

try:
    with open('two_hand_model.p', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file 'two_hand_model.p' not found.")
    print("Please run train_two_hand_model.py first.")
    exit()

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

# Text-to-Speech Engine
def get_tts_engine():
    """Initializes and returns a TTS engine with a slower rate."""
    try:
        engine = pyttsx3.init()
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate - 100)
        return engine
    except Exception as e:
        print(f"Error initializing TTS: {e}")
        return None

sentence = ""
last_prediction = None
last_word_add_time = time.time()
last_word_added = None 
PREDICTION_COOLDOWN_S = 1.5 


print("Starting inference... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break
    
    H_frame, W_frame, _ = frame.shape
    if H_frame != FRAME_HEIGHT or W_frame != FRAME_WIDTH:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    
    data_aux = np.zeros(42 * 2)
    
    predicted_sign = None
    confidence = 0.0 
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label

            x_ = []
            y_ = []
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
            
            min_x = min(x_)
            min_y = min(y_)
            
            normalized_landmarks = []
            for landmark in hand_landmarks.landmark:
                normalized_landmarks.append(landmark.x - min_x)
                normalized_landmarks.append(landmark.y - min_y)

            if handedness == 'Left':
                data_aux[0:42] = normalized_landmarks
            elif handedness == 'Right':
                data_aux[42:84] = normalized_landmarks
            
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )

        probabilities = model.predict_proba([data_aux])[0]
        confidence = np.max(probabilities)
        predicted_sign = model.classes_[np.argmax(probabilities)]
        
    show_prediction = False
    if confidence >= CONFIDENCE_THRESHOLD:
        show_prediction = True
        
    if show_prediction:
        text = f"{predicted_sign} ({confidence*100:.0f}%)"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
        
    current_time = time.time()
    
    if predicted_sign != last_prediction and (current_time - last_word_add_time > PREDICTION_COOLDOWN_S):
        
        if show_prediction and predicted_sign != last_word_added:
            sentence += predicted_sign + " "
            last_word_add_time = current_time
            last_word_added = predicted_sign
        
        last_prediction = predicted_sign

    cv2.rectangle(frame, (0, H - 70), (W, H), (0, 0, 0), -1)
    
    max_text_width = W - 40 
    wrapped_lines = wrap_text(sentence, cv2.FONT_HERSHEY_SIMPLEX, 1, 2, max_text_width)
    
    if wrapped_lines:
        last_line = wrapped_lines[-1]
        cv2.putText(frame, last_line, (20, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(frame, "'Q': Quit", (W - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, "'C': Clear", (W - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, "'S': Speak", (W - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
        if win_w <= 0 or win_h <= 0: raise Exception("Window minimized")
        scale = min(win_w / W, win_h / H)
        new_w, new_h = int(W * scale), int(H * scale)
        x_offset, y_offset = (win_w - new_w) // 2, (win_h - new_h) // 2
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame
        cv2.imshow(WINDOW_NAME, canvas)
    except Exception:
        if 'frame' in locals():
            cv2.imshow(WINDOW_NAME, frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        sentence = ""
        last_prediction = None 
        last_word_added = None 
    if key == ord('s'):
        sentence_to_speak = sentence.strip()
        if sentence_to_speak:
            print(f"Speaking: {sentence_to_speak}")
            tts_engine = get_tts_engine()
            if tts_engine:
                tts_engine.say(sentence_to_speak)
                tts_engine.runAndWait()
                del tts_engine

cap.release()
cv2.destroyAllWindows()