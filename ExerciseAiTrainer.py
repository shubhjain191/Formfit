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
    """Calculate the angle between three points."""
    if np.any(np.array([a, b, c]) == 0):
        return -1.0  # Placeholder for missing landmarks
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    # Calculate angle using arctan2 for better quadrant handling
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    # Ensure angle is within 0-180 degrees range
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_distance(a, b):
    """Calculate Euclidean distance between two points."""
    if np.any(np.array([a, b]) == 0):
        return -1.0  # Placeholder for missing landmarks
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def calculate_y_distance(a, b):
    """Calculate Y-coordinate distance between two points."""
    if np.any(np.array([a, b]) == 0):
        return -1.0  # Placeholder for missing landmarks
    return np.abs(a[1] - b[1])

def draw_styled_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.55, font_color=(255, 255, 255), font_thickness=2, bg_color=(0, 0, 0), padding=5):
    """Draw text with a colored background rectangle."""
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x, text_y = position
    box_coords = ((text_x - padding, text_y + padding), (text_x + text_size[0] + padding, text_y - text_size[1] - padding))
    cv2.rectangle(frame, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

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
    """Count push-up repetitions and update stage."""
    # Calculate angles for both arms to ensure symmetrical form
    right_arm_angle = detector.find_angle(img, 12, 14, 16)  # Right shoulder, elbow, wrist
    left_arm_angle = detector.find_angle(img, 11, 13, 15)   # Left shoulder, elbow, wrist
    
    # Get shoulder positions for angle visualization
    right_shoulder = landmark_list[12][1:]
    left_shoulder = landmark_list[11][1:]
    
    # Display angles on the frame
    exercise_instance.visualize_angle(img, right_arm_angle, right_shoulder)
    exercise_instance.visualize_angle(img, left_arm_angle, left_shoulder)

    # Count rep when arms bend below threshold (down) and then extend back up
    if left_arm_angle < 220:  # Arms bent - down position
        stage = "down"
    if left_arm_angle > 240 and stage == "down":  # Arms extended - up position
        stage = "up"
        counter += 1
    return stage, counter

def count_repetition_squat(detector, img, landmark_list, stage, counter, exercise_instance):
    """Count squat repetitions and update stage."""
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
    """Count bicep curl repetitions and update stages for both arms."""
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
    """Count shoulder press repetitions and update stage."""
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

# Feedback Generation Function
def generate_feedback(detector, img, landmark_list, exercise_type, stage=None):
    """Generate real-time feedback based on exercise type and stage."""
    feedback = []

    if exercise_type == 'push_up':
        right_arm_angle = detector.find_angle(img, 12, 14, 16)
        left_arm_angle = detector.find_angle(img, 11, 13, 15)
        avg_arm_angle = (right_arm_angle + left_arm_angle) / 2
        shoulders = np.array([landmark_list[11][1:], landmark_list[12][1:]]).mean(axis=0)
        hips = np.array([landmark_list[23][1:], landmark_list[24][1:]]).mean(axis=0)
        knees = np.array([landmark_list[25][1:], landmark_list[26][1:]]).mean(axis=0)
        back_angle = calculate_angle(shoulders, hips, knees)

        if stage == "down":
            # Arm angle feedback
            if 70 <= avg_arm_angle <= 110:
                feedback.append((random.choice(POSITIVE_MESSAGES["arm"]), True))
            elif 110 < avg_arm_angle <= 120:
                feedback.append((random.choice(NEGATIVE_MESSAGES["arm_too_high"]), False))
            elif avg_arm_angle > 120:
                feedback.append(("Lower your body more; chest almost to the ground.", False))
            elif 60 <= avg_arm_angle < 70:
                feedback.append((random.choice(NEGATIVE_MESSAGES["arm_too_low"]), False))
            elif avg_arm_angle < 60:
                feedback.append(("Don't go too low; it might strain your shoulders.", False))
            # Back alignment feedback
            if 160 <= back_angle <= 180:
                feedback.append((random.choice(POSITIVE_MESSAGES["back"]), True))
            elif 150 <= back_angle < 160:
                feedback.append(("Try to straighten your back a little more.", False))
            elif back_angle < 150:
                feedback.append((random.choice(NEGATIVE_MESSAGES["back"]), False))
                if random.random() < 0.2:
                    feedback.append((IMPROVEMENT_TIPS["back"], False))
        else:  # "up" stage
            feedback.append((random.choice(POSITIVE_MESSAGES["general"]), True))

    elif exercise_type == 'squat':
        right_knee_angle = detector.find_angle(img, 24, 26, 28)
        left_knee_angle = detector.find_angle(img, 23, 25, 27)
        avg_knee_angle = (right_knee_angle + left_knee_angle) / 2
        shoulders = np.array([landmark_list[11][1:], landmark_list[12][1:]]).mean(axis=0)
        hips = np.array([landmark_list[23][1:], landmark_list[24][1:]]).mean(axis=0)
        knees = np.array([landmark_list[25][1:], landmark_list[26][1:]]).mean(axis=0)
        back_angle = calculate_angle(shoulders, hips, knees)

        if stage == "down":
            # Knee angle feedback
            if 60 <= avg_knee_angle <= 100:
                feedback.append(("Perfect squat depth!", True))
            elif 100 < avg_knee_angle <= 110:
                feedback.append(("Try to squat a bit lower.", False))
            elif avg_knee_angle > 110:
                feedback.append(("Squat lower; aim for thighs parallel to the ground.", False))
            elif 50 <= avg_knee_angle < 60:
                feedback.append(("You're squatting a bit too low.", False))
            elif avg_knee_angle < 50:
                feedback.append(("Don't squat too low; it might strain your knees.", False))
            # Back angle feedback
            if 150 <= back_angle <= 180:
                feedback.append(("Perfect back alignment!", True))
            elif 140 <= back_angle < 150:
                feedback.append(("Try to keep your back straighter.", False))
            elif back_angle < 140:
                feedback.append(("Keep your chest up and back straight.", False))
        else:  # "up" stage
            feedback.append((random.choice(POSITIVE_MESSAGES["general"]), True))

    elif exercise_type == 'bicep_curl':
        # Elbow stability feedback
        right_shoulder_x = landmark_list[12][1]
        right_elbow_x = landmark_list[14][1]
        if abs(right_elbow_x - right_shoulder_x) > 0.1:
            feedback.append(("Keep your right elbow closer to your body.", False))
        left_shoulder_x = landmark_list[11][1]
        left_elbow_x = landmark_list[13][1]
        if abs(left_elbow_x - left_shoulder_x) > 0.1:
            feedback.append(("Keep your left elbow closer to your body.", False))
        # Range of motion feedback
        right_arm_angle = detector.find_angle(img, 12, 14, 16)
        left_arm_angle = detector.find_angle(img, 11, 13, 15)
        if right_arm_angle < 30 and left_arm_angle < 30:
            feedback.append(("Good curl, now extend.", True))
        elif right_arm_angle > 160 and left_arm_angle > 160:
            feedback.append(("Good, now curl up.", True))
        else:
            feedback.append(("Keep your movement smooth.", True))

    elif exercise_type == 'shoulder_press':
        # Back posture feedback
        back_angle = detector.find_angle(img, 11, 23, 25)
        if 170 <= back_angle <= 180:
            feedback.append(("Excellent upright posture!", True))
        else:
            feedback.append(("Keep your back straight.", False))
        # Arm extension feedback
        right_arm_angle = detector.find_angle(img, 12, 14, 16)
        left_arm_angle = detector.find_angle(img, 11, 13, 15)
        if stage == "up":
            if abs(right_arm_angle - 180) < 10 and abs(left_arm_angle - 180) < 10:
                feedback.append(("Arms fully extended, great!", True))
            else:
                feedback.append(("Extend your arms fully.", False))
        elif stage == "down":
            feedback.append(("Lower your arms to start position.", False))

    # Add occasional motivational message (10% chance)
    if random.random() < 0.1:
        feedback.append((random.choice(MOTIVATIONAL_MESSAGES), True))

    return feedback[:3]  # Limit to 3 messages per frame

# Feedback Display Function
def display_feedback(img, feedback_messages):
    """Display feedback messages on the frame with color coding."""
    height, width = img.shape[:2]
    y_offset = height - 30
    for message, is_positive in feedback_messages:
        bg_color = (0, 255, 0) if is_positive else (0, 0, 255)  # Green for positive, red for negative
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        draw_styled_text(img, message, (width - text_size[0] - 10, y_offset), font_scale=0.6, bg_color=bg_color)
        y_offset -= 30

# Exercise Class
class Exercise:
    def __init__(self):
        # Load pre-trained models and encoders
        try:
            # LSTM model for exercise classification
            self.lstm_model = load_model('final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5')
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            self.lstm_model = None
            
        try:
            # Feature scaler for input normalization
            self.scaler = joblib.load('thesis_bidirectionallstm_scaler.pkl')
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.scaler = None
            
        try:
            # Label encoder for exercise class names
            self.label_encoder = joblib.load('thesis_bidirectionallstm_label_encoder.pkl')
            self.exercise_classes = self.label_encoder.classes_
        except Exception as e:
            print(f"Error loading label encoder: {e}")
            self.label_encoder = None
            self.exercise_classes = []
            
        self.feedback_messages = []
        self.stop_requested = False

    def extract_features(self, landmarks):
        """Extract features from landmarks for classification."""
        features = []
        if len(landmarks) == len(relevant_landmarks_indices) * 3:
            # Calculate joint angles for key body parts
            features.append(calculate_angle(landmarks[0:3], landmarks[6:9], landmarks[12:15]))  # LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
            features.append(calculate_angle(landmarks[3:6], landmarks[9:12], landmarks[15:18]))  # RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
            features.append(calculate_angle(landmarks[18:21], landmarks[24:27], landmarks[30:33]))  # LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
            features.append(calculate_angle(landmarks[21:24], landmarks[27:30], landmarks[33:36]))  # RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
            features.append(calculate_angle(landmarks[0:3], landmarks[18:21], landmarks[24:27]))  # LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE
            features.append(calculate_angle(landmarks[3:6], landmarks[21:24], landmarks[27:30]))  # RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE
            features.append(calculate_angle(landmarks[18:21], landmarks[0:3], landmarks[6:9]))  # LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW
            features.append(calculate_angle(landmarks[21:24], landmarks[3:6], landmarks[9:12]))  # RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW

            # Calculate distances between key points
            distances = [
                calculate_distance(landmarks[0:3], landmarks[3:6]),  # LEFT_SHOULDER to RIGHT_SHOULDER
                calculate_distance(landmarks[18:21], landmarks[21:24]),  # LEFT_HIP to RIGHT_HIP
                calculate_distance(landmarks[18:21], landmarks[24:27]),  # LEFT_HIP to LEFT_KNEE
                calculate_distance(landmarks[21:24], landmarks[27:30]),  # RIGHT_HIP to RIGHT_KNEE
                calculate_distance(landmarks[0:3], landmarks[18:21]),  # LEFT_SHOULDER to LEFT_HIP
                calculate_distance(landmarks[3:6], landmarks[21:24]),  # RIGHT_SHOULDER to RIGHT_HIP
                calculate_distance(landmarks[6:9], landmarks[24:27]),  # LEFT_ELBOW to LEFT_KNEE
                calculate_distance(landmarks[9:12], landmarks[27:30]),  # RIGHT_ELBOW to RIGHT_KNEE
                calculate_distance(landmarks[12:15], landmarks[0:3]),  # LEFT_WRIST to LEFT_SHOULDER
                calculate_distance(landmarks[15:18], landmarks[3:6]),  # RIGHT_WRIST to RIGHT_SHOULDER
                calculate_distance(landmarks[12:15], landmarks[18:21]),  # LEFT_WRIST to LEFT_HIP
                calculate_distance(landmarks[15:18], landmarks[21:24])   # RIGHT_WRIST to RIGHT_HIP
            ]
            
            # Calculate vertical distances for specific joints
            y_distances = [
                calculate_y_distance(landmarks[6:9], landmarks[0:3]),  # LEFT_ELBOW to LEFT_SHOULDER
                calculate_y_distance(landmarks[9:12], landmarks[3:6])   # RIGHT_ELBOW to RIGHT_SHOULDER
            ]

            # Normalize distances using body proportions
            normalization_factor = -1
            distances_to_check = [
                calculate_distance(landmarks[0:3], landmarks[18:21]),  # Various body segment lengths
                calculate_distance(landmarks[3:6], landmarks[21:24]),
                calculate_distance(landmarks[18:21], landmarks[24:27]),
                calculate_distance(landmarks[21:24], landmarks[27:30])
            ]
            
            # Find valid normalization factor
            for distance in distances_to_check:
                if distance > 0:
                    normalization_factor = distance
                    break
            if normalization_factor == -1:
                normalization_factor = 0.5  # Fallback value

            # Apply normalization to make features scale-invariant
            normalized_distances = [d / normalization_factor if d != -1.0 else d for d in distances]
            normalized_y_distances = [d / normalization_factor if d != -1.0 else d for d in y_distances]

            features.extend(normalized_distances)
            features.extend(normalized_y_distances)
        else:
            features = [-1.0] * 22  # Default features if landmarks are missing
        return features

    def preprocess_frame(self, frame, pose):
        """Preprocess video frame to extract landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        landmarks = []
        if results.pose_landmarks:
            for idx in relevant_landmarks_indices:
                landmark = results.pose_landmarks.landmark[idx]
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks

    def visualize_angle(self, img, angle, landmark):
        """Visualize angle on the frame."""
        cv2.putText(img, str(int(angle)), tuple(np.multiply(landmark, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    def auto_classify_and_count(self):
        """Main function to classify exercises and count repetitions."""
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
                    # Feedback is per frame, no stage needed
                elif current_prediction == 'shoulder press':
                    exercise_type = 'shoulder_press'
                    stages['shoulder_press'], counters['shoulder_press'] = count_repetition_shoulder_press(detector, frame, landmark_list, stages['shoulder_press'], counters['shoulder_press'], self)
                    stage = stages['shoulder_press']

                if exercise_type:
                    self.feedback_messages = generate_feedback(detector, frame, landmark_list, exercise_type, stage=stage)
                    display_feedback(frame, self.feedback_messages)

            height, width, _ = frame.shape
            num_exercises = len(counters)
            vertical_spacing = height // (num_exercises + 1)

            cv2.rectangle(frame, (0, 0), (0, height), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, 0), (width, 0), (0, 0, 0), -1)

            short_name = exercise_name_map.get(current_prediction, current_prediction)
            draw_styled_text(frame, f"Exercise: {short_name}", ((width - 290) // 2 + 100, 20))

            for idx, (exercise, count) in enumerate(counters.items()):
                short_name = exercise_name_map.get(exercise, exercise)
                draw_styled_text(frame, f"{short_name}: {count}", (10, (idx + 1) * vertical_spacing))

            stframe.image(frame, channels='BGR', use_container_width=False, width=820)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        video_container.empty()

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
                if self.feedback_messages:
                    for msg, is_positive in self.feedback_messages:
                        if is_positive:
                            st.success(msg)
                        else:
                            st.error(msg)
                else:
                    st.success("Great form! Keep it up!")
            if st.button("Start New Session", key="restart_button"):
                st.experimental_rerun()

    def are_hands_joined(self, landmark_list, stop, is_video=False):
        """Check if hands are joined to stop exercise."""
        left_wrist = landmark_list[15][1:]
        right_wrist = landmark_list[16][1:]
        distance = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
        if distance < 30 and not is_video:
            stop = True
        return stop

    def repetitions_counter(self, img, counter):
        """Display repetition counter on frame."""
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
        """Generic method to handle exercise tracking."""
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
                                feedback_messages = generate_feedback(detector, img, landmark_list, exercise_type, stage=stage_to_pass)
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
                        feedback_messages = generate_feedback(detector, img, landmark_list, exercise_type, stage=stage_to_pass)
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
