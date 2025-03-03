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
import random
import pandas as pd

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

# Utility Functions
def calculate_angle(a, b, c):
    if np.any(np.array([a, b, c]) == 0):
        return -1.0
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def calculate_y_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0
    return np.abs(a[1] - b[1])

# Feedback Message Pools
POSITIVE_MESSAGES = {
    "arm": ["Great arm angle!", "Perfect depth!", "Excellent form!"],
    "back": ["Excellent back position!", "Your back is straight!", "Perfect alignment!"],
    "general": ["Great job!", "Keep pushing!", "You're doing great!"]
}

NEGATIVE_MESSAGES = {
    "arm_too_high": ["Try to lower a bit more.", "Bend your elbows more.", "Get closer to the ground."],
    "arm_too_low": ["Don't go too low; aim for 90°.", "You're going too deep.", "Ease up a bit."],
    "back": ["Keep your back straight.", "Don't let your hips sag.", "Chest up, back straight."]
}

IMPROVEMENT_TIPS = {
    "back": "Engage your core to keep your back straight.",
    "knees": "Keep your knees aligned with your toes."
}

MOTIVATIONAL_MESSAGES = ["You're getting stronger!", "Almost there, keep it up!", "Great effort!"]

# Repetition Counting Functions
def count_repetition_push_up(detector, img, landmark_list, stage, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    right_shoulder = landmark_list[12][1:]
    left_shoulder = landmark_list[11][1:]
    exercise_instance.visualize_angle(img, right_arm_angle, right_shoulder)
    exercise_instance.visualize_angle(img, left_arm_angle, left_shoulder)

    if left_arm_angle < 220:
        stage = "down"
    if left_arm_angle > 240 and stage == "down":
        stage = "up"
        counter += 1
        current_time = time.time()
        exercise_instance.log_rep_data('push_up', current_time, {'right_arm': right_arm_angle, 'left_arm': left_arm_angle})
        print(f"Push-up rep {counter} logged at {current_time}")
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
        current_time = time.time()
        exercise_instance.log_rep_data('squat', current_time, {'right_leg': right_leg_angle, 'left_leg': left_leg_angle})
        print(f"Squat rep {counter} logged at {current_time}")
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
        current_time = time.time()
        exercise_instance.log_rep_data('bicep_curl', current_time, {'right_arm': right_arm_angle, 'left_arm': left_arm_angle})
        print(f"Bicep curl rep {counter} logged at {current_time}")
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
        current_time = time.time()
        exercise_instance.log_rep_data('shoulder_press', current_time, {'right_arm': right_arm_angle, 'left_arm': left_arm_angle})
        print(f"Shoulder press rep {counter} logged at {current_time}")
    return stage, counter

# Feedback Generation Function
def generate_feedback(detector, img, landmark_list, exercise_type, stage=None):
    form_feedback = []
    if exercise_type == 'push_up':
        right_arm_angle = detector.find_angle(img, 12, 14, 16)
        left_arm_angle = detector.find_angle(img, 11, 13, 15)
        avg_arm_angle = (right_arm_angle + left_arm_angle) / 2
        shoulders = np.array([landmark_list[11][1:], landmark_list[12][1:]]).mean(axis=0)
        hips = np.array([landmark_list[23][1:], landmark_list[24][1:]]).mean(axis=0)
        knees = np.array([landmark_list[25][1:], landmark_list[26][1:]]).mean(axis=0)
        back_angle = calculate_angle(shoulders, hips, knees)

        if stage == "down":
            if 70 <= avg_arm_angle <= 110:
                form_feedback.append((random.choice(POSITIVE_MESSAGES["arm"]), True))
            elif 110 < avg_arm_angle <= 120:
                form_feedback.append((random.choice(NEGATIVE_MESSAGES["arm_too_high"]), False))
            elif avg_arm_angle > 120:
                form_feedback.append(("Lower your body more; chest almost to the ground.", False))
            elif 60 <= avg_arm_angle < 70:
                form_feedback.append((random.choice(NEGATIVE_MESSAGES["arm_too_low"]), False))
            elif avg_arm_angle < 60:
                form_feedback.append(("Don't go too low; it might strain your shoulders.", False))
            if 160 <= back_angle <= 180:
                form_feedback.append((random.choice(POSITIVE_MESSAGES["back"]), True))
            elif 150 <= back_angle < 160:
                form_feedback.append(("Try to straighten your back a little more.", False))
            elif back_angle < 150:
                form_feedback.append((random.choice(NEGATIVE_MESSAGES["back"]), False))
                if random.random() < 0.2:
                    form_feedback.append((IMPROVEMENT_TIPS["back"], False))
        else:
            form_feedback.append((random.choice(POSITIVE_MESSAGES["general"]), True))

    elif exercise_type == 'squat':
        right_knee_angle = detector.find_angle(img, 24, 26, 28)
        left_knee_angle = detector.find_angle(img, 23, 25, 27)
        avg_knee_angle = (right_knee_angle + left_knee_angle) / 2
        shoulders = np.array([landmark_list[11][1:], landmark_list[12][1:]]).mean(axis=0)
        hips = np.array([landmark_list[23][1:], landmark_list[24][1:]]).mean(axis=0)
        knees = np.array([landmark_list[25][1:], landmark_list[26][1:]]).mean(axis=0)
        back_angle = calculate_angle(shoulders, hips, knees)

        if stage == "down":
            if 60 <= avg_knee_angle <= 100:
                form_feedback.append(("Perfect squat depth!", True))
            elif 100 < avg_knee_angle <= 110:
                form_feedback.append(("Try to squat a bit lower.", False))
            elif avg_knee_angle > 110:
                form_feedback.append(("Squat lower; aim for thighs parallel to the ground.", False))
            elif 50 <= avg_knee_angle < 60:
                form_feedback.append(("You're squatting a bit too low.", False))
            elif avg_knee_angle < 50:
                form_feedback.append(("Don't squat too low; it might strain your knees.", False))
            if 150 <= back_angle <= 180:
                form_feedback.append(("Perfect back alignment!", True))
            elif 140 <= back_angle < 150:
                form_feedback.append(("Try to keep your back straighter.", False))
            elif back_angle < 140:
                form_feedback.append(("Keep your chest up and back straight.", False))
        else:
            form_feedback.append((random.choice(POSITIVE_MESSAGES["general"]), True))

    elif exercise_type == 'bicep_curl':
        right_shoulder_x = landmark_list[12][1]
        right_elbow_x = landmark_list[14][1]
        if abs(right_elbow_x - right_shoulder_x) > 0.1:
            form_feedback.append(("Keep your right elbow closer to your body.", False))
        left_shoulder_x = landmark_list[11][1]
        left_elbow_x = landmark_list[13][1]
        if abs(left_elbow_x - left_shoulder_x) > 0.1:
            form_feedback.append(("Keep your left elbow closer to your body.", False))
        right_arm_angle = detector.find_angle(img, 12, 14, 16)
        left_arm_angle = detector.find_angle(img, 11, 13, 15)
        if right_arm_angle < 30 and left_arm_angle < 30:
            form_feedback.append(("Good curl, now extend.", True))
        elif right_arm_angle > 160 and left_arm_angle > 160:
            form_feedback.append(("Good, now curl up.", True))
        else:
            form_feedback.append(("Keep your movement smooth.", True))

    elif exercise_type == 'shoulder_press':
        back_angle = detector.find_angle(img, 11, 23, 25)
        if 170 <= back_angle <= 180:
            form_feedback.append(("Excellent upright posture!", True))
        else:
            form_feedback.append(("Keep your back straight.", False))
        right_arm_angle = detector.find_angle(img, 12, 14, 16)
        left_arm_angle = detector.find_angle(img, 11, 13, 15)
        if stage == "up":
            if abs(right_arm_angle - 180) < 10 and abs(left_arm_angle - 180) < 10:
                form_feedback.append(("Arms fully extended, great!", True))
            else:
                form_feedback.append(("Extend your arms fully.", False))
        elif stage == "down":
            form_feedback.append(("Lower your arms to start position.", False))

    if random.random() < 0.1:
        form_feedback.append((random.choice(MOTIVATIONAL_MESSAGES), True))

    return form_feedback[:3]

# Exercise Class
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
        self.form_feedback = []
        self.fatigue_feedback = []
        self.stop_requested = False
        self.rep_timestamps = []
        self.angle_history = {exercise: [] for exercise in ['push_up', 'squat', 'bicep_curl', 'shoulder_press']}
        self.fatigue_level = 0.0  # Current smoothed fatigue level
        self.fatigue_warnings = []
        self.prev_form_feedback = []
        self.prev_fatigue_score = 0.0  # To smooth fatigue transitions

    def log_rep_data(self, exercise_type, timestamp, angles):
        self.rep_timestamps.append(timestamp)
        self.angle_history[exercise_type].append(angles)

    def detect_fatigue(self, exercise_type, window_size=5, min_reps=3):
        if len(self.rep_timestamps) < min_reps or not self.angle_history[exercise_type]:
            return 0.0, None

        # Use a sliding window of the last 'window_size' reps
        window_timestamps = self.rep_timestamps[-window_size:] if len(self.rep_timestamps) >= window_size else self.rep_timestamps
        window_angles = self.angle_history[exercise_type][-window_size:] if len(self.angle_history[exercise_type]) >= window_size else self.angle_history[exercise_type]

        # Repetition speed (time between reps)
        rep_times = np.diff(window_timestamps)
        avg_rep_speed = np.mean(rep_times) if len(rep_times) > 0 else 0.0
        baseline_speed = np.mean(np.diff(self.rep_timestamps[:min_reps])) if len(self.rep_timestamps) >= min_reps else avg_rep_speed
        speed_score = min(1.0, max(0.0, (avg_rep_speed - baseline_speed) / baseline_speed)) if baseline_speed > 0 else 0.0

        # Angle variability (consistency of movement)
        key_angle = 'right_arm' if exercise_type in ['push_up', 'bicep_curl', 'shoulder_press'] else 'right_leg'
        angles = [rep[key_angle] for rep in window_angles]
        variability = np.std(angles)
        variability_score = min(1.0, variability / 10.0)  # Increased threshold for more sensitivity

        # Range of Motion (ROM) degradation
        target_rom = {
            'push_up': 90.0,  # Ideal lowest angle
            'squat': 90.0,    # Ideal lowest knee angle
            'bicep_curl': 150.0,  # Ideal max extension
            'shoulder_press': 180.0  # Ideal max extension
        }
        if exercise_type in ['push_up', 'squat']:
            rom = min(angles)
            rom_score = max(0.0, (target_rom[exercise_type] - rom) / target_rom[exercise_type])
        else:
            rom = max(angles)
            rom_score = max(0.0, (rom - target_rom[exercise_type]) / target_rom[exercise_type])
        rom_score = min(1.0, rom_score)

        # Weighted fatigue score (emphasize speed and ROM)
        fatigue_score = (0.4 * speed_score + 0.3 * variability_score + 0.3 * rom_score)
        
        # Smooth the fatigue score with previous value
        fatigue_score = 0.7 * fatigue_score + 0.3 * self.prev_fatigue_score
        self.prev_fatigue_score = fatigue_score

        # Detailed feedback based on dominant factor
        dominant_factor = max([('speed', speed_score), ('variability', variability_score), ('ROM', rom_score)], key=lambda x: x[1])[0]
        if fatigue_score < 0.2:
            warning = None
        elif fatigue_score < 0.4:
            if dominant_factor == 'speed':
                warning = "Mild slowdown detected—maintain your pace."
            elif dominant_factor == 'variability':
                warning = "Mild inconsistency—keep movements steady."
            else:
                warning = "Mild ROM reduction—focus on full range."
        elif fatigue_score < 0.6:
            if dominant_factor == 'speed':
                warning = "Moderate slowdown—take a short rest soon."
            elif dominant_factor == 'variability':
                warning = "Moderate inconsistency—stabilize your form."
            else:
                warning = "Moderate ROM loss—extend fully."
        else:
            if dominant_factor == 'speed':
                warning = "High fatigue: Slow pace detected—rest for 30s."
            elif dominant_factor == 'variability':
                warning = "High fatigue: Unsteady form—rest now."
            else:
                warning = "High fatigue: Reduced ROM—take a break."

        print(f"{exercise_type} - Speed: {speed_score:.2f}, Variability: {variability_score:.2f}, ROM: {rom_score:.2f}, Fatigue: {fatigue_score:.2f}, Dominant: {dominant_factor}")
        
        return fatigue_score, warning

    def export_fatigue_data(self, filename="fatigue_data.csv"):
        data = {'timestamp': self.rep_timestamps, 'exercise': [], 'angles': []}
        for ex, angles in self.angle_history.items():
            data['exercise'].extend([ex] * len(angles))
            data['angles'].extend(angles)
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Fatigue data exported to {filename}")

    def extract_features(self, landmarks):
        features = []
        if len(landmarks) == len(relevant_landmarks_indices) * 3:
            features.append(calculate_angle(landmarks[0:3], landmarks[6:9], landmarks[12:15]))
            features.append(calculate_angle(landmarks[3:6], landmarks[9:12], landmarks[15:18]))
            features.append(calculate_angle(landmarks[18:21], landmarks[24:27], landmarks[30:33]))
            features.append(calculate_angle(landmarks[21:24], landmarks[27:30], landmarks[33:36]))
            features.append(calculate_angle(landmarks[0:3], landmarks[18:21], landmarks[24:27]))
            features.append(calculate_angle(landmarks[3:6], landmarks[21:24], landmarks[27:30]))
            features.append(calculate_angle(landmarks[18:21], landmarks[0:3], landmarks[6:9]))
            features.append(calculate_angle(landmarks[21:24], landmarks[3:6], landmarks[9:12]))

            distances = [
                calculate_distance(landmarks[0:3], landmarks[3:6]),
                calculate_distance(landmarks[18:21], landmarks[21:24]),
                calculate_distance(landmarks[18:21], landmarks[24:27]),
                calculate_distance(landmarks[21:24], landmarks[27:30]),
                calculate_distance(landmarks[0:3], landmarks[18:21]),
                calculate_distance(landmarks[3:6], landmarks[21:24]),
                calculate_distance(landmarks[6:9], landmarks[24:27]),
                calculate_distance(landmarks[9:12], landmarks[27:30]),
                calculate_distance(landmarks[12:15], landmarks[0:3]),
                calculate_distance(landmarks[15:18], landmarks[3:6]),
                calculate_distance(landmarks[12:15], landmarks[18:21]),
                calculate_distance(landmarks[15:18], landmarks[21:24])
            ]
            y_distances = [
                calculate_y_distance(landmarks[6:9], landmarks[0:3]),
                calculate_y_distance(landmarks[9:12], landmarks[3:6])
            ]
            normalization_factor = max([d for d in distances if d > 0], default=0.5)
            normalized_distances = [d / normalization_factor if d != -1.0 else d for d in distances]
            normalized_y_distances = [d / normalization_factor if d != -1.0 else d for d in y_distances]
            features.extend(normalized_distances)
            features.extend(normalized_y_distances)
        else:
            features = [-1.0] * 22
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
        cv2.putText(img, str(int(angle)), tuple(np.multiply(landmark, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    def auto_classify_and_count(self):
        header = st.container()
        main_container = st.container()
        summary_container = st.container()

        with header:
            col1, col2 = st.columns([4, 1])
            with col2:
                stop_button = st.button('Stop Exercise', key='stop_button')

        with main_container:
            col_video, col_feedback = st.columns([3, 1])
            with col_video:
                stframe = st.empty()
            with col_feedback:
                form_feedback_placeholder = st.empty()
                fatigue_feedback_placeholder = st.empty()

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

        detector = pm.posture_detector()
        pose = mp.solutions.pose.Pose()

        exercise_name_map = {
            'push_up': 'Push-up',
            'squat': 'Squat',
            'bicep_curl': 'Curl',
            'shoulder_press': 'Press'
        }

        while cap.isOpened():
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
                predicted_class = np.argmax(prediction, axis=1)[0]
                current_prediction = self.exercise_classes[predicted_class]
                landmarks_window = []
                frame_count = 0

            detector.find_person(frame, draw=True)
            landmark_list = detector.find_landmarks(frame, draw=True)
            if len(landmark_list) > 0:
                exercise_type = None
                stage = None
                if current_prediction == 'push-up':
                    exercise_type = 'push_up'
                    stages['push_up'], counters['push_up'] = count_repetition_push_up(detector, frame, landmark_list, stages['push_up'], counters['push_up'], self)
                    stage = stages['push_up']
                elif current_prediction == 'squat':
                    exercise_type = 'squat'
                    stages['squat'], counters['squat'] = count_repetition_squat(detector, frame, landmark_list, stages['squat'], counters['squat'], self)
                    stage = stages['squat']
                elif current_prediction == 'barbell biceps curl':
                    exercise_type = 'bicep_curl'
                    stages['right_bicep_curl'], stages['left_bicep_curl'], counters['bicep_curl'] = count_repetition_bicep_curl(detector, frame, landmark_list, stages['right_bicep_curl'], stages['left_bicep_curl'], counters['bicep_curl'], self)
                elif current_prediction == 'shoulder press':
                    exercise_type = 'shoulder_press'
                    stages['shoulder_press'], counters['shoulder_press'] = count_repetition_shoulder_press(detector, frame, landmark_list, stages['shoulder_press'], counters['shoulder_press'], self)
                    stage = stages['shoulder_press']

                if exercise_type:
                    new_form_feedback = generate_feedback(detector, frame, landmark_list, exercise_type, stage=stage)
                    fatigue_score, fatigue_warning = self.detect_fatigue(exercise_type)
                    self.fatigue_level = max(self.fatigue_level, fatigue_score)
                    new_fatigue_feedback = []
                    if fatigue_warning:
                        new_fatigue_feedback.append((fatigue_warning, False))
                        self.fatigue_warnings.append(fatigue_warning)

                    # Update form feedback if changed
                    if new_form_feedback != self.prev_form_feedback:
                        self.form_feedback = new_form_feedback
                        self.prev_form_feedback = new_form_feedback
                        with form_feedback_placeholder.container():
                            st.subheader("Form Feedback")
                            if self.form_feedback:
                                for msg, is_positive in self.form_feedback:
                                    if is_positive:
                                        st.success(msg)
                                    else:
                                        st.error(msg)
                            else:
                                st.info("No form feedback yet.")

                    # Update fatigue feedback
                    self.fatigue_feedback = new_fatigue_feedback
                    with fatigue_feedback_placeholder.container():
                        st.subheader("Fatigue Feedback")
                        st.metric("Fatigue Level", f"{self.fatigue_level:.0%}")
                        if self.fatigue_feedback:
                            for msg, _ in self.fatigue_feedback:
                                st.warning(msg)
                        else:
                            st.success("No fatigue detected—keep going!")

            height, width, _ = frame.shape
            num_exercises = len(counters)
            vertical_spacing = height // (num_exercises + 1)

            cv2.rectangle(frame, (0, 0), (0, height), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, 0), (width, 0), (0, 0, 0), -1)

            short_name = exercise_name_map.get(current_prediction, current_prediction)
            cv2.putText(frame, f"Exercise: {short_name}", ((width - 290) // 2 + 100, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw rep counters with highlighted exercise names and counters
            for idx, (exercise, count) in enumerate(counters.items()):
                short_name = exercise_name_map.get(exercise, exercise)
                y_pos = (idx + 1) * vertical_spacing
                
                # Calculate text sizes
                name_size = cv2.getTextSize(short_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                count_text = f": {count}"
                count_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Add background for both name and counter
                overlay = frame.copy()
                cv2.rectangle(overlay, 
                            (10, y_pos - 20),  # Adjust y position for background
                            (10 + name_size[0] + count_size[0], y_pos + 5),  # Size based on total width
                            (0, 0, 0), 
                            -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Draw exercise name and counter in yellow
                cv2.putText(frame, short_name, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, count_text, (15 + name_size[0], y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            stframe.image(frame, channels='BGR', use_container_width=False, width=850)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        main_container.empty()

        with summary_container:
            st.success("Exercise session completed!")
            st.write("### Exercise Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Repetitions")
                for exercise, count in counters.items():
                    if count > 0:
                        st.metric(label=exercise_name_map.get(exercise, exercise), value=f"{count} reps")
            with col2:
                st.write("#### Form Feedback")
                if self.form_feedback:
                    for msg, is_positive in self.form_feedback:
                        if is_positive:
                            st.success(msg)
                        else:
                            st.error(msg)
                else:
                    st.success("Great form! Keep it up!")
            st.write("### Fatigue Analysis")
            st.metric("Peak Fatigue Level", f"{self.fatigue_level:.0%}")
            if self.fatigue_warnings:
                st.warning("Fatigue Warnings Issued:")
                for w in set(self.fatigue_warnings):
                    st.write(f"- {w}")
            self.export_fatigue_data()
            if st.button("Start New Session", key="restart_button"):
                st.experimental_rerun()

    def are_hands_joined(self, landmark_list, stop, is_video=False):
        left_wrist = landmark_list[15][1:]
        right_wrist = landmark_list[16][1:]
        distance = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
        if distance < 30 and not is_video:
            stop = True
        return stop

    def repetitions_counter(self, img, counter):
        cv2.rectangle(img, (0, 0), (225, 73), (245, 117, 16), -1)
        cv2.putText(img, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    def push_up(self, cap, is_video=False, counter=0, stage=None):
        self.exercise_method(cap, is_video, count_repetition_push_up, counter=counter, stage=stage)

    def squat(self, cap, is_video=False, counter=0, stage=None):
        self.exercise_method(cap, is_video, count_repetition_squat, counter=counter, stage=stage)

    def bicept_curl(self, cap, is_video=False, counter=0, stage_right=None, stage_left=None):
        self.exercise_method(cap, is_video, count_repetition_bicep_curl, multi_stage=True, counter=counter, stage_right=stage_right, stage_left=stage_left)

    def shoulder_press(self, cap, is_video=False, counter=0, stage=None):
        self.exercise_method(cap, is_video, count_repetition_shoulder_press, counter=counter, stage=stage)

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
                            exercise_type = None
                            if count_repetition_function.__name__ == 'count_repetition_push_up':
                                exercise_type = 'push_up'
                            elif count_repetition_function.__name__ == 'count_repetition_squat':
                                exercise_type = 'squat'
                            elif count_repetition_function.__name__ == 'count_repetition_bicep_curl':
                                exercise_type = 'bicep_curl'
                            elif count_repetition_function.__name__ == 'count_repetition_shoulder_press':
                                exercise_type = 'shoulder_press'

                            if exercise_type:
                                stage_to_pass = stage if exercise_type in ['push_up', 'squat', 'shoulder_press'] else None
                                self.form_feedback = generate_feedback(detector, img, landmark_list, exercise_type, stage=stage_to_pass)

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
                    exercise_type = None
                    if count_repetition_function.__name__ == 'count_repetition_push_up':
                        exercise_type = 'push_up'
                    elif count_repetition_function.__name__ == 'count_repetition_squat':
                        exercise_type = 'squat'
                    elif count_repetition_function.__name__ == 'count_repetition_bicep_curl':
                        exercise_type = 'bicep_curl'
                    elif count_repetition_function.__name__ == 'count_repetition_shoulder_press':
                        exercise_type = 'shoulder_press'

                    if exercise_type:
                        stage_to_pass = stage if exercise_type in ['push_up', 'squat', 'shoulder_press'] else None
                        self.form_feedback = generate_feedback(detector, img, landmark_list, exercise_type, stage=stage_to_pass)

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

if __name__ == "__main__":
    exercise = Exercise()
    exercise.auto_classify_and_count()
