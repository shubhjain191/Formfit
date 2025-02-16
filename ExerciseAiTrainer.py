import cv2
import PoseModule2 as pm
import numpy as np
import streamlit as st
from AiTrainer_utils import *
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import time

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define relevant landmarks indices
relevant_landmarks_indices = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value
]

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    if np.any(np.array([a, b, c]) == 0):
        return -1.0  # Placeholder for missing landmarks
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to calculate Euclidean distance between two points
def calculate_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0  # Placeholder for missing landmarks
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

# Function to calculate Y-coordinate distance between two points
def calculate_y_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0  # Placeholder for missing landmarks
    return np.abs(a[1] - b[1])

def draw_styled_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.55, font_color=(255, 255, 255), font_thickness=2, bg_color=(0, 0, 0), padding=5):
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x, text_y = position
    box_coords = ((text_x - padding, text_y + padding), (text_x + text_size[0] + padding, text_y - text_size[1] - padding))
    cv2.rectangle(frame, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)


def count_repetition_push_up(detector, img, landmark_list, stage, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    right_shoulder = landmark_list[12][1:]
    right_wrist = landmark_list[16][1:]
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    left_shoulder = landmark_list[11][1:]
    exercise_instance.visualize_angle(img, right_arm_angle, right_shoulder)
    exercise_instance.visualize_angle(img, left_arm_angle, left_shoulder)

    if left_arm_angle < 220:
        stage = "down"
    if left_arm_angle > 240 and stage == "down":
        stage = "up"
        counter += 1
    
    return stage, counter



def count_repetition_squat(detector, img, landmark_list, stage, counter, exercise_instance):
    right_leg_angle = detector.find_angle(img, 24, 26, 28)
    left_leg_angle = detector.find_angle(img, 23, 25, 27)
    right_leg = landmark_list[26][1:]
    exercise_instance.visualize_angle(img, right_leg_angle, right_leg)

    if right_leg_angle > 160 and left_leg_angle < 220:
        stage = "down"
    if right_leg_angle < 140 and left_leg_angle > 210 and stage == "down":
        stage = "up"
        counter += 1
    
    return stage, counter

def count_repetition_bicep_curl(detector, img, landmark_list, stage_right, stage_left, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    exercise_instance.visualize_angle(img, right_arm_angle, landmark_list[14][1:])
    exercise_instance.visualize_angle(img, left_arm_angle, landmark_list[13][1:])

    if right_arm_angle > 160 and right_arm_angle < 200:
        stage_right = "down"
    if left_arm_angle < 200 and left_arm_angle > 140:
        stage_left = "down"
    
    if stage_right == "down" and (right_arm_angle > 310 or right_arm_angle < 60) and (left_arm_angle > 310 or left_arm_angle < 60) and stage_left == "down":
        stage_right = "up"
        stage_left = "up"
        counter += 1
    
    return stage_right, stage_left, counter

def count_repetition_shoulder_press(detector, img, landmark_list, stage, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    right_elbow = landmark_list[14][1:]
    exercise_instance.visualize_angle(img, right_arm_angle, right_elbow)

    if right_arm_angle > 280 and left_arm_angle < 80:
        stage = "down"
    if right_arm_angle < 240 and left_arm_angle > 120 and stage == "down":
        stage = "up"
        counter += 1
    
    return stage, counter



# Define the class that handles the analysis of the exercises
class Exercise:
    def __init__(self):
        try:
            self.lstm_model = load_model('final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5')
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            self.lstm_model = None
        
        try:
            self.scaler = joblib.load('thesis_bidirectionallstm_scaler.pkl')
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.scaler = None
        
        try:
            self.label_encoder = joblib.load('thesis_bidirectionallstm_label_encoder.pkl')
            self.exercise_classes = self.label_encoder.classes_
        except Exception as e:
            print(f"Error loading label encoder: {e}")
            self.label_encoder = None
            self.exercise_classes = []
            self.feedback_messages = []
            self.stop_requested = False

    def extract_features(self, landmarks):
        features = []
        if len(landmarks) == len(relevant_landmarks_indices) * 3:
            # Angles
            features.append(calculate_angle(landmarks[0:3], landmarks[6:9], landmarks[12:15]))  # LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
            features.append(calculate_angle(landmarks[3:6], landmarks[9:12], landmarks[15:18]))  # RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
            features.append(calculate_angle(landmarks[18:21], landmarks[24:27], landmarks[30:33]))  # LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
            features.append(calculate_angle(landmarks[21:24], landmarks[27:30], landmarks[33:36]))  # RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
            features.append(calculate_angle(landmarks[0:3], landmarks[18:21], landmarks[24:27]))  # LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE
            features.append(calculate_angle(landmarks[3:6], landmarks[21:24], landmarks[27:30]))  # RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE

            # New angles
            features.append(calculate_angle(landmarks[18:21], landmarks[0:3], landmarks[6:9]))  # LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW
            features.append(calculate_angle(landmarks[21:24], landmarks[3:6], landmarks[9:12]))  # RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW

            # Distances
            distances = [
                calculate_distance(landmarks[0:3], landmarks[3:6]),  # LEFT_SHOULDER, RIGHT_SHOULDER
                calculate_distance(landmarks[18:21], landmarks[21:24]),  # LEFT_HIP, RIGHT_HIP
                calculate_distance(landmarks[18:21], landmarks[24:27]),  # LEFT_HIP, LEFT_KNEE
                calculate_distance(landmarks[21:24], landmarks[27:30]),  # RIGHT_HIP, RIGHT_KNEE
                calculate_distance(landmarks[0:3], landmarks[18:21]),  # LEFT_SHOULDER, LEFT_HIP
                calculate_distance(landmarks[3:6], landmarks[21:24]),  # RIGHT_SHOULDER, RIGHT_HIP
                calculate_distance(landmarks[6:9], landmarks[24:27]),  # LEFT_ELBOW, LEFT_KNEE
                calculate_distance(landmarks[9:12], landmarks[27:30]),  # RIGHT_ELBOW, RIGHT_KNEE
                calculate_distance(landmarks[12:15], landmarks[0:3]),  # LEFT_WRIST, LEFT_SHOULDER
                calculate_distance(landmarks[15:18], landmarks[3:6]),  # RIGHT_WRIST, RIGHT_SHOULDER
                calculate_distance(landmarks[12:15], landmarks[18:21]),  # LEFT_WRIST, LEFT_HIP
                calculate_distance(landmarks[15:18], landmarks[21:24])   # RIGHT_WRIST, RIGHT_HIP
            ]

            # Y-coordinate distances
            y_distances = [
                calculate_y_distance(landmarks[6:9], landmarks[0:3]),  # LEFT_ELBOW, LEFT_SHOULDER
                calculate_y_distance(landmarks[9:12], landmarks[3:6])   # RIGHT_ELBOW, RIGHT_SHOULDER
            ]

            # Normalization factor based on shoulder-hip or hip-knee distance
            normalization_factor = -1
            distances_to_check = [
                calculate_distance(landmarks[0:3], landmarks[18:21]),  # LEFT_SHOULDER, LEFT_HIP
                calculate_distance(landmarks[3:6], landmarks[21:24]),  # RIGHT_SHOULDER, RIGHT_HIP
                calculate_distance(landmarks[18:21], landmarks[24:27]),  # LEFT_HIP, LEFT_KNEE
                calculate_distance(landmarks[21:24], landmarks[27:30])   # RIGHT_HIP, RIGHT_KNEE
            ]

            for distance in distances_to_check:
                if distance > 0:
                    normalization_factor = distance
                    break
            
            if normalization_factor == -1:
                normalization_factor = 0.5  # Fallback normalization factor
            
            # Normalize distances
            normalized_distances = [d / normalization_factor if d != -1.0 else d for d in distances]
            normalized_y_distances = [d / normalization_factor if d != -1.0 else d for d in y_distances]

            # Combine features
            features.extend(normalized_distances)
            features.extend(normalized_y_distances)

        else:
            print(f"Insufficient landmarks: expected {len(relevant_landmarks_indices)}, got {len(landmarks)//3}")
            features = [-1.0] * 22  # Placeholder for missing landmarks
        return features
    
    def preprocess_frame(self, frame, pose):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        landmarks = []
        if results.pose_landmarks:
            for idx in relevant_landmarks_indices:
                landmark = results.pose_landmarks.landmark[idx]
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def visualize_angle(self, img, angle, landmark):
        cv2.putText(img, str(int(angle)),
                    tuple(np.multiply(landmark, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Auto classify and count method with repetition counting logic
    def auto_classify_and_count(self):
        # Create persistent containers for the interface
        header = st.container()
        video_container = st.container()
        summary_container = st.container()

        with header:
            col1, col2 = st.columns([4, 1])
            with col2:
                stop_button = st.button('Stop Exercise', key='stop_button')

        stframe = video_container.empty()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error opening webcam.")
            return

        window_size = 30
        landmarks_window = []
        frame_count = 0
        current_prediction = "No prediction yet"
        counters = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}
        stages = {'push_up': None, 'squat': None, 'left_bicep_curl': None, 'right_bicep_curl': None, 'shoulder_press': None}

        print("Starting real-time classification...")

        detector = pm.posture_detector()
        pose = mp.solutions.pose.Pose()

        exercise_name_map = {
            'push_up': 'Push-up',
            'squat': 'Squat',
            'bicep_curl': 'Curl',
            'shoulder_press': 'Press'
        }

        while cap.isOpened():
            # Check if stop button was clicked
            if stop_button:
                break

            ret, frame = cap.read()
            if not ret:
                print("Error reading frame.")
                break

            landmarks = self.preprocess_frame(frame, pose)
            if len(landmarks) == len(relevant_landmarks_indices) * 3:
                features = self.extract_features(landmarks)
                if len(features) == 22:
                    landmarks_window.append(features)

            frame_count += 1

            if len(landmarks_window) == window_size:
                landmarks_window_np = np.array(landmarks_window).flatten().reshape(1, -1)
                scaled_landmarks_window = self.scaler.transform(landmarks_window_np)
                scaled_landmarks_window = scaled_landmarks_window.reshape(1, window_size, 22)

                prediction = self.lstm_model.predict(scaled_landmarks_window)

                if prediction.shape[1] != len(self.exercise_classes):
                    print(f"Unexpected prediction shape: {prediction.shape}")
                    return

                predicted_class = np.argmax(prediction, axis=1)[0]

                if predicted_class >= len(self.exercise_classes):
                    print(f"Invalid class index: {predicted_class}")
                    return

                current_prediction = self.exercise_classes[predicted_class]
                print(f"Current Prediction: {current_prediction}")

                landmarks_window = []
                frame_count = 0

            # Repetition counting logic based on current prediction
            detector.find_person(frame, draw=True)  # Ensuring landmarks are drawn on the frame
            landmark_list = detector.find_landmarks(frame, draw=True)  # Change draw=False to draw=True
            if len(landmark_list) > 0:
                exercise_type = None
                if current_prediction == 'push-up':
                    exercise_type = 'push_up'
                    stages['push_up'], counters['push_up'] = count_repetition_push_up(detector, frame, landmark_list, stages['push_up'], counters['push_up'], self)
                elif current_prediction == 'squat':
                    exercise_type = 'squat'
                    stages['squat'], counters['squat'] = count_repetition_squat(detector, frame, landmark_list, stages['squat'], counters['squat'], self)
                elif current_prediction == 'barbell biceps curl':
                    exercise_type = 'bicep_curl'
                    stages['right_bicep_curl'], stages['left_bicep_curl'], counters['bicep_curl'] = count_repetition_bicep_curl(detector, frame, landmark_list, stages['right_bicep_curl'], stages['left_bicep_curl'], counters['bicep_curl'], self)
                elif current_prediction == 'shoulder press':
                    exercise_type = 'shoulder_press'
                    stages['shoulder_press'], counters['shoulder_press'] = count_repetition_shoulder_press(detector, frame, landmark_list, stages['shoulder_press'], counters['shoulder_press'], self)

                 # Generate and display feedback if we have a valid exercise type
                if exercise_type:
                    self.feedback_messages = generate_feedback(detector, frame, landmark_list, exercise_type)
                    display_feedback(frame, self.feedback_messages)

            # Calculate the spacing for exercise repetitions display
            height, width, _ = frame.shape
            num_exercises = len(counters)
            vertical_spacing = height // (num_exercises + 1)

            # Draw black rectangles on the left and top side
            cv2.rectangle(frame, (0, 0), (0, height), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, 0), (width, 0), (0, 0, 0), -1)

            # Display the frame with predicted exercise and repetition count
            short_name = exercise_name_map.get(current_prediction, current_prediction)
            text_size, _ = cv2.getTextSize(f"Exercise: {short_name}", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
            draw_styled_text(frame, f"Exercise: {short_name}", ((width - 290) // 2 + 100, 20))

            for idx, (exercise, count) in enumerate(counters.items()):
                short_name = exercise_name_map.get(exercise, exercise)
                draw_styled_text(frame, f"{short_name}: {count}", (10, (idx + 1) * vertical_spacing))

            stframe.image(frame, channels='BGR', use_container_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Clear the video container
        video_container.empty()
        
        # Display final summary in the persistent summary container
        with summary_container:
            st.success("Exercise session completed!")
            st.write("### Exercise Summary")
            
            # Create a more visually appealing summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Repetitions")
                for exercise, count in counters.items():
                    if count > 0:
                        st.metric(
                            label=exercise_name_map.get(exercise, exercise),
                            value=f"{count} reps"
                        )
            
            with col2:
                st.write("#### Form Feedback")
                if self.feedback_messages:
                    for msg in self.feedback_messages:
                        st.info(msg)
                else:
                    st.success("Great form! Keep it up!")
            
            # Add a restart button
            if st.button("Start New Session", key="restart_button"):
                st.experimental_rerun()
                
    # Add feedback parameters and thresholds
EXERCISE_FEEDBACK = {
    'push_up': {
        'arm_angle': {
            'min': 70, 'max': 110,
            'good': "Great arm angle - perfect 90° at the bottom",
            'bad': "Keep your arms at 90 degrees at the bottom"
        },
        'back_straight': {
            'min': 160, 'max': 180,
            'good': "Excellent back position - staying straight and strong",
            'bad': "Maintain a straight back"
        },
        'elbow_position': {
            'threshold': 30,
            'good': "Perfect elbow position - nice and tight",
            'bad': "Keep elbows close to body"
        },
        'depth': {
            'threshold': 0.3,
            'good': "Great depth on those push-ups",
            'bad': "Lower your chest more"
        },
        'symmetry': {
            'threshold': 15,
            'good': "Excellent balanced form",
            'bad': "Balance your form on both sides"
        },
        'head_position': {
            'threshold': 10,
            'good': "Perfect head alignment",
            'bad': "Keep your head neutral, aligned with spine"
        },
        'hand_placement': {
            'min': 1.0, 'max': 1.5,
            'good': "Ideal hand placement",
            'bad': "Hands should be shoulder-width apart"
        },
        'hip_alignment': {
            'threshold': 5,
            'good': "Hips perfectly aligned",
            'bad': "Keep hips level with shoulders"
        },
        'core_engagement': {
            'threshold': 20,
            'good': "Strong core engagement",
            'bad': "Engage your core to prevent sagging"
        }
    },
    'squat': {
        'knee_angle': {
            'min': 60, 'max': 100,
            'good': "Perfect knee bend at 90°",
            'bad': "Bend knees to 90 degrees"
        },
        'hip_position': {
            'threshold': 0.4,
            'good': "Great squat depth",
            'bad': "Lower hips more"
        },
        'knee_alignment': {
            'threshold': 20,
            'good': "Excellent knee tracking",
            'bad': "Keep knees aligned with toes"
        },
        'back_angle': {
            'min': 150, 'max': 180,
            'good': "Perfect back position",
            'bad': "Keep your back straight"
        },
        'depth': {
            'threshold': 0.35,
            'good': "Hitting full depth - great work",
            'bad': "Go deeper into the squat"
        },
        'balance': {
            'threshold': 0.1,
            'good': "Perfectly balanced",
            'bad': "Distribute weight evenly"
        }
    },
    'bicep_curl': {
        'elbow_stability': {
            'threshold': 15,
            'good': "Stable elbows - great control",
            'bad': "Keep elbows still"
        },
        'curl_range': {
            'min': 30, 'max': 160,
            'good': "Full range of motion - excellent",
            'bad': "Complete full range of motion"
        },
        'shoulder_movement': {
            'threshold': 10,
            'good': "Perfect form - no swinging",
            'bad': "Minimize shoulder swinging"
        },
        'symmetry': {
            'threshold': 15,
            'good': "Great balanced curls",
            'bad': "Keep curls even on both sides"
        }
    },
    'shoulder_press': {
        'arm_alignment': {
            'threshold': 15,
            'good': "Perfect arm alignment",
            'bad': "Keep arms aligned with shoulders"
        },
        'press_height': {
            'threshold': 0.8,
            'good': "Full extension - great work",
            'bad': "Press all the way up"
        },
        'elbow_angle': {
            'min': 85, 'max': 95,
            'good': "Perfect starting position",
            'bad': "Start with 90° elbow angle"
        },
        'back_posture': {
            'min': 170, 'max': 180,
            'good': "Excellent posture",
            'bad': "Maintain straight back"
        }
    }
}

def generate_feedback(detector, img, landmark_list, exercise_type):
    feedback = {
        'positive': [],
        'negative': []
    }
    
    if exercise_type == 'push_up':
        # Check arm angle
        right_arm_angle = detector.find_angle(img, 12, 14, 16)
        left_arm_angle = detector.find_angle(img, 11, 13, 15)
        
        if (EXERCISE_FEEDBACK['push_up']['arm_angle']['min'] <= right_arm_angle <= EXERCISE_FEEDBACK['push_up']['arm_angle']['max']):
            feedback['positive'].append(EXERCISE_FEEDBACK['push_up']['arm_angle']['good'])
        else:
            feedback['negative'].append(EXERCISE_FEEDBACK['push_up']['arm_angle']['bad'])
            
        # Check back alignment
        shoulders = np.array([landmark_list[11][1:], landmark_list[12][1:]])
        hips = np.array([landmark_list[23][1:], landmark_list[24][1:]])
        back_angle = calculate_angle(shoulders[0], hips[0], hips[1])
        
        if back_angle >= EXERCISE_FEEDBACK['push_up']['back_straight']['min']:
            feedback['positive'].append(EXERCISE_FEEDBACK['push_up']['back_straight']['good'])
        else:
            feedback['negative'].append(EXERCISE_FEEDBACK['push_up']['back_straight']['bad'])
            
    elif exercise_type == 'squat':
        # Check knee angle
        right_knee_angle = detector.find_angle(img, 24, 26, 28)
        left_knee_angle = detector.find_angle(img, 23, 25, 27)
        
        if (EXERCISE_FEEDBACK['squat']['knee_angle']['min'] <= right_knee_angle <= EXERCISE_FEEDBACK['squat']['knee_angle']['max']):
            feedback['positive'].append(EXERCISE_FEEDBACK['squat']['knee_angle']['good'])
        else:
            feedback['negative'].append(EXERCISE_FEEDBACK['squat']['knee_angle']['bad'])
            
        # Check hip depth
        hip_y = (landmark_list[23][2] + landmark_list[24][2]) / 2
        knee_y = (landmark_list[25][2] + landmark_list[26][2]) / 2
        if abs(hip_y - knee_y) >= EXERCISE_FEEDBACK['squat']['hip_position']['threshold']:
            feedback['positive'].append(EXERCISE_FEEDBACK['squat']['hip_position']['good'])
        else:
            feedback['negative'].append(EXERCISE_FEEDBACK['squat']['hip_position']['bad'])
            
    elif exercise_type == 'bicep_curl':
        # Check elbow stability
        right_shoulder = np.array(landmark_list[12][1:])
        right_elbow = np.array(landmark_list[14][1:])
        movement = np.linalg.norm(right_shoulder - right_elbow)
        
        if movement <= EXERCISE_FEEDBACK['bicep_curl']['elbow_stability']['threshold']:
            feedback['positive'].append(EXERCISE_FEEDBACK['bicep_curl']['elbow_stability']['good'])
        else:
            feedback['negative'].append(EXERCISE_FEEDBACK['bicep_curl']['elbow_stability']['bad'])
            
        # Check range of motion
        right_arm_angle = detector.find_angle(img, 12, 14, 16)
        if (EXERCISE_FEEDBACK['bicep_curl']['curl_range']['min'] <= right_arm_angle <= EXERCISE_FEEDBACK['bicep_curl']['curl_range']['max']):
            feedback['positive'].append(EXERCISE_FEEDBACK['bicep_curl']['curl_range']['good'])
        else:
            feedback['negative'].append(EXERCISE_FEEDBACK['bicep_curl']['curl_range']['bad'])
            
    elif exercise_type == 'shoulder_press':
        # Check arm alignment
        right_shoulder = np.array(landmark_list[12][1:])
        right_wrist = np.array(landmark_list[16][1:])
        alignment = np.linalg.norm(right_shoulder - right_wrist)
        
        if alignment <= EXERCISE_FEEDBACK['shoulder_press']['arm_alignment']['threshold']:
            feedback['positive'].append(EXERCISE_FEEDBACK['shoulder_press']['arm_alignment']['good'])
        else:
            feedback['negative'].append(EXERCISE_FEEDBACK['shoulder_press']['arm_alignment']['bad'])
            
        # Check back posture
        back_angle = detector.find_angle(img, 11, 23, 25)
        if (EXERCISE_FEEDBACK['shoulder_press']['back_posture']['min'] <= back_angle <= EXERCISE_FEEDBACK['shoulder_press']['back_posture']['max']):
            feedback['positive'].append(EXERCISE_FEEDBACK['shoulder_press']['back_posture']['good'])
        else:
            feedback['negative'].append(EXERCISE_FEEDBACK['shoulder_press']['back_posture']['bad'])
    
    return feedback

def display_feedback(img, feedback_messages):
    # Display feedback messages in bottom right corner
    height, width = img.shape[:2]
    y_offset = height - 30
    
    # Combine positive and negative feedback
    all_messages = []
    if isinstance(feedback_messages, dict):
        all_messages.extend(feedback_messages.get('positive', []))
        all_messages.extend(feedback_messages.get('negative', []))
    
    # Show only last 3 messages
    for message in all_messages[-3:]:
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        draw_styled_text(img, message, (width - text_size[0] - 10, y_offset), 
                        font_scale=0.6, 
                        bg_color=(200, 50, 50))
        y_offset -= 30

# Update the exercise_method function to include feedback
def exercise_method(self, cap, is_video, count_repetition_function, multi_stage=False, counter=0, stage=None, stage_right=None, stage_left=None):
    # ... (previous code remains the same until inside the while loop)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = detector.find_person(frame)
        landmark_list = detector.find_landmarks(img, draw=False)

        if len(landmark_list) != 0:
            # Generate and display feedback
            exercise_type = None
            if count_repetition_function == count_repetition_push_up:
                exercise_type = 'push_up'
            elif count_repetition_function == count_repetition_squat:
                exercise_type = 'squat'
            elif count_repetition_function == count_repetition_bicep_curl:
                exercise_type = 'bicep_curl'
            elif count_repetition_function == count_repetition_shoulder_press:
                exercise_type = 'shoulder_press'

            feedback_messages = generate_feedback(detector, img, landmark_list, exercise_type)
            display_feedback(img, feedback_messages)

            if multi_stage:
                stage_right, stage_left, counter = count_repetition_function(detector, img, landmark_list, stage_right, stage_left, counter, self)
            else:
                stage, counter = count_repetition_function(detector, img, landmark_list, stage, counter, self)

            if self.are_hands_joined(landmark_list, stop=False, is_video=is_video):
                break

        self.repetitions_counter(img, counter)
        stframe.image(img, channels='BGR', use_container_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Check if hands are joined together in a 'prayer' gesture
    def are_hands_joined(self, landmark_list, stop, is_video=False):
        # Extract wrist coordinates
        left_wrist = landmark_list[15][1:]  # (x, y) for left wrist
        right_wrist = landmark_list[16][1:]  # (x, y) for right wrist

        # Calculate the Euclidean distance between the wrists
        distance = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
        # Consider hands joined if the distance is below a certain threshold, e.g., 50 pixels
        if distance < 30 and not is_video:
            print("JOINED HANDS")
            stop = True
            return stop
        
        return False

    # Visualize the angle between 3 point on screen
    def visualize_angle(self, img, angle, landmark):
            cv2.putText(img, str(angle),
                        tuple(np.multiply(landmark, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

    # Visualize repetitions of the exercise on screen
    def repetitions_counter(self, img, counter):
        cv2.rectangle(img, (0, 0), (225, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(img, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Define push-up method
    def push_up(self, cap, is_video=False, counter=0, stage=None):
        self.exercise_method(cap, is_video, count_repetition_push_up, counter=counter, stage=stage)

    # Define squat method
    def squat(self, cap, is_video=False, counter=0, stage=None):
        self.exercise_method(cap, is_video, count_repetition_squat, counter=counter, stage=stage)

    # Define bicep curl method
    def bicept_curl(self, cap, is_video=False, counter=0, stage_right=None, stage_left=None):
        self.exercise_method(cap, is_video, count_repetition_bicep_curl, multi_stage=True, counter=counter, stage_right=stage_right, stage_left=stage_left)

    # Define shoulder press method
    def shoulder_press(self, cap, is_video=False, counter=0, stage=None):
        self.exercise_method(cap, is_video, count_repetition_shoulder_press, counter=counter, stage=stage)

    # Generic exercise method
    # Generic exercise method
    # Generic exercise method
    # Previous code remains the same until exercise_method function

def exercise_method(self, cap, is_video, count_repetition_function, multi_stage=False, counter=0, stage=None, stage_right=None, stage_left=None):
    if is_video:
        stframe = st.empty()
        detector = pm.posture_detector()
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1 / original_fps
        frame_count = 0
        start_time = time.time()
        last_update_time = start_time
        update_interval = 0.1

        while cap.isOpened():
            current_time = time.time()
            elapsed_time = current_time - start_time
            target_frame = int(elapsed_time * original_fps)

            while frame_count < target_frame:
                ret, frame = cap.read()
                if not ret:
                    return

                frame_count += 1

                if frame_count == target_frame:
                    img = detector.find_person(frame)
                    landmark_list = detector.find_landmarks(img, draw=False)

                    if len(landmark_list) != 0:
                        # Determine exercise type
                        exercise_type = None
                        if count_repetition_function.__name__ == 'count_repetition_push_up':
                            exercise_type = 'push_up'
                        elif count_repetition_function.__name__ == 'count_repetition_squat':
                            exercise_type = 'squat'
                        elif count_repetition_function.__name__ == 'count_repetition_bicep_curl':
                            exercise_type = 'bicep_curl'
                        elif count_repetition_function.__name__ == 'count_repetition_shoulder_press':
                            exercise_type = 'shoulder_press'

                        # Generate and display feedback
                        if exercise_type:
                            feedback_messages = generate_feedback(detector, img, landmark_list, exercise_type)
                            display_feedback(img, feedback_messages)

                        if multi_stage:
                            stage_right, stage_left, counter = count_repetition_function(detector, img, landmark_list, stage_right, stage_left, counter, self)
                        else:
                            stage, counter = count_repetition_function(detector, img, landmark_list, stage, counter, self)

                    self.repetitions_counter(img, counter)

            if current_time - last_update_time >= update_interval:
                stframe.image(img, channels='BGR', use_container_width=True)
                last_update_time = current_time

            time.sleep(0.001)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    else:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        detector = pm.posture_detector()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img = detector.find_person(frame)
            landmark_list = detector.find_landmarks(img, draw=False)

            if len(landmark_list) != 0:
                # Determine exercise type
                exercise_type = None
                if count_repetition_function.__name__ == 'count_repetition_push_up':
                    exercise_type = 'push_up'
                elif count_repetition_function.__name__ == 'count_repetition_squat':
                    exercise_type = 'squat'
                elif count_repetition_function.__name__ == 'count_repetition_bicep_curl':
                    exercise_type = 'bicep_curl'
                elif count_repetition_function.__name__ == 'count_repetition_shoulder_press':
                    exercise_type = 'shoulder_press'

                # Generate and display feedback
                if exercise_type:
                    feedback_messages = generate_feedback(detector, img, landmark_list, exercise_type)
                    display_feedback(img, feedback_messages)

                if multi_stage:
                    stage_right, stage_left, counter = count_repetition_function(detector, img, landmark_list, stage_right, stage_left, counter, self)
                else:
                    stage, counter = count_repetition_function(detector, img, landmark_list, stage, counter, self)

                if self.are_hands_joined(landmark_list, stop=False):
                    break

            self.repetitions_counter(img, counter)
            stframe.image(img, channels='BGR', use_container_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
