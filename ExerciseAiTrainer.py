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
from collections import defaultdict
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

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

# Feedback message categories
POSITIVE_MESSAGES = {
    "push_up_arm": ["Perfect arm bend—textbook form!", "Great elbow depth!", "Arms are spot on!", "Smooth arm movement!", "Nailed the push-up angle!"],
    "push_up_back": ["Back’s like a steel rod!", "Core locked in tight!", "Perfect plank alignment!", "Rock-solid posture!", "Back’s a straight line!"],
    "squat_legs": ["Legs are powerhouse strong!", "Knee bend is perfect!", "Solid squat depth!", "Great leg control!", "Thighs are dialed in!"],
    "squat_back": ["Back’s upright and steady!", "Posture’s a 10!", "Chest up, back strong!", "Spine’s perfectly aligned!", "Great back stability!"],
    "bicep_curl": ["Biceps are popping!", "Smooth curl, full range!", "Perfect arm isolation!", "Controlled and strong!", "Curls are flawless!"],
    "shoulder_press": ["Shoulders are crushing it!", "Full press, full power!", "Great overhead form!", "Arms locked out perfectly!", "Press is on point!"]
}

NEGATIVE_MESSAGES = {
    "push_up_arm": ["Bend elbows more for depth.", "Lower your chest closer.", "Arms need more range.", "Don’t lock out early.", "Push deeper!"],
    "push_up_back": ["Straighten that back!", "Hips are sagging—lift up.", "Avoid dipping your core.", "Back’s bending—tighten up!", "Keep it flat!"],
    "squat_legs": ["Squat lower—hit parallel!", "Knees too far forward.", "Hips need to drop more.", "Legs not deep enough.", "Push knees out!"],
    "squat_back": ["Keep back straight—don’t lean!", "Chest up, don’t slump!", "Back’s tilting—fix it!", "Don’t round your spine!", "Stay upright!"],
    "bicep_curl": ["Elbows drifting—keep them in!", "Don’t swing—control it!", "Curl higher for range.", "Too fast—slow it down!", "Arms need more lift!"],
    "shoulder_press": ["Extend arms fully up!", "Don’t lean back—stay tall!", "Lower arms more on reset.", "Press straighter!", "Arms not locked out!"]
}

IMPROVEMENT_TIPS = {
    "push_up_arm": ["Aim for a 90° elbow bend.", "Lower slow and controlled.", "Keep elbows close to body.", "Pause at the bottom."],
    "push_up_back": ["Engage core for a Fill out the rest flat back.", "Imagine a plank hold.", "Squeeze glutes to lift hips.", "Focus on shoulder-hip alignment."],
    "push_up_symmetry": ["Even out arm angles.", "Push with both sides equally.", "Check your mirror image.", "Balance your effort."],
    "squat_legs": ["Push knees out slightly.", "Drive up through heels.", "Sink hips to knee level.", "Control the descent."],
    "squat_back": ["Pull shoulders back tight.", "Look forward, not down.", "Brace your core hard.", "Keep chest lifted."],
    "squat_symmetry": ["Match knee bends.", "Shift weight evenly.", "Check leg alignment.", "Stay centered."],
    "bicep_curl": ["Lock elbows by your sides.", "Lower weight slowly.", "Full curl to chest.", "Avoid momentum swings."],
    "bicep_symmetry": ["Sync both arms’ motion.", "Lift weights evenly.", "Focus on weaker side.", "Mirror your curls."],
    "shoulder_press": ["Press straight overhead.", "Keep back flat, no arch.", "Lower to chin level.", "Engage shoulders fully."],
    "shoulder_symmetry": ["Even out arm extension.", "Press both sides together.", "Match your arm heights.", "Balance the load."]
}

MOTIVATIONAL_MESSAGES = [
    "You’re unstoppable—keep it up!", 
    "Powering through like a champ!", 
    "Every rep’s building strength!", 
    "You’ve got this—pure grit!", 
    "Beast mode activated!"
]

def count_repetition_push_up(detector, img, landmark_list, stage, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    right_shoulder = landmark_list[12][1:]
    left_shoulder = landmark_list[11][1:]
    exercise_instance.visualize_angle(img, right_arm_angle, right_shoulder)
    exercise_instance.visualize_angle(img, left_arm_angle, left_shoulder)
    if left_arm_angle < 240:
        stage = "down"
    if left_arm_angle > 250 and stage == "down":
        stage = "up"
        counter += 1
        current_time = time.time()
        exercise_instance.log_rep_data('push_up', current_time, {'right_arm': right_arm_angle, 'left_arm': left_arm_angle})
        print(f"Push-up rep {counter} logged at {current_time}")
        exercise_instance.last_feedback_time = current_time - exercise_instance.feedback_interval
        exercise_instance.last_activity_time = current_time
    return stage, counter

def count_repetition_squat(detector, img, landmark_list, stage, counter, exercise_instance):
    right_leg_angle = detector.find_angle(img, 24, 26, 28)
    left_leg_angle = detector.find_angle(img, 23, 25, 27)
    right_leg = landmark_list[26][1:]
    exercise_instance.visualize_angle(img, right_leg_angle, right_leg)
    if right_leg_angle > 150 and left_leg_angle < 200:
        stage = "down"
    if right_leg_angle < 150 and left_leg_angle > 180 and stage == "down":
        stage = "up"
        counter += 1
        current_time = time.time()
        exercise_instance.log_rep_data('squat', current_time, {'right_leg': right_leg_angle, 'left_leg': left_leg_angle})
        print(f"Squat rep {counter} logged at {current_time}")
        exercise_instance.last_feedback_time = current_time - exercise_instance.feedback_interval
        exercise_instance.last_activity_time = current_time
    return stage, counter

def count_repetition_bicep_curl(detector, img, landmark_list, stage_right, stage_left, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    exercise_instance.visualize_angle(img, right_arm_angle, landmark_list[14][1:])
    exercise_instance.visualize_angle(img, left_arm_angle, landmark_list[13][1:])
    if right_arm_angle > 140 and right_arm_angle < 180:
        stage_right = "down"
    if left_arm_angle > 140 and left_arm_angle < 180:
        stage_left = "down"
    if stage_right == "down" and stage_left == "down" and (
        (right_arm_angle < 80 or right_arm_angle > 300) and (left_arm_angle < 80 or left_arm_angle > 300)
    ):
        stage_right = "up"
        stage_left = "up"
        counter += 1
        current_time = time.time()
        exercise_instance.log_rep_data('bicep_curl', current_time, {'right_arm': right_arm_angle, 'left_arm': left_arm_angle})
        print(f"Bicep curl rep {counter} logged at {current_time}")
        exercise_instance.last_feedback_time = current_time - exercise_instance.feedback_interval
        exercise_instance.last_activity_time = current_time
    return stage_right, stage_left, counter

def count_repetition_shoulder_press(detector, img, landmark_list, stage, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    right_elbow = landmark_list[14][1:]
    exercise_instance.visualize_angle(img, right_arm_angle, right_elbow)
    if right_arm_angle > 250 and left_arm_angle < 100:
        stage = "down"
    if right_arm_angle < 210 and left_arm_angle > 150 and stage == "down":
        stage = "up"
        counter += 1
        current_time = time.time()
        exercise_instance.log_rep_data('shoulder_press', current_time, {'right_arm': right_arm_angle, 'left_arm': left_arm_angle})
        print(f"Shoulder press rep {counter} logged at {current_time}")
        exercise_instance.last_feedback_time = current_time - exercise_instance.feedback_interval
        exercise_instance.last_activity_time = current_time
    return stage, counter

def generate_feedback(detector, img, landmark_list, exercise_type, stage, feedback_history, prev_angles, exercise_instance):
    form_feedback = []
    issues = defaultdict(int)
    current_angles = {}
    user_reps = exercise_instance.counters.get(exercise_type, 0)

    feedback_triggered = stage == "up"

    # Default positive feedback key mapping for each exercise
    default_positive_key = {
        'push_up': 'push_up_arm',
        'squat': 'squat_legs',
        'bicep_curl': 'bicep_curl',
        'shoulder_press': 'shoulder_press'
    }

    if exercise_type == 'push_up':
        right_arm_angle = detector.find_angle(img, 12, 14, 16)
        left_arm_angle = detector.find_angle(img, 11, 13, 15)
        current_angles = {'right_arm': right_arm_angle, 'left_arm': left_arm_angle}
        avg_arm_angle = (right_arm_angle + left_arm_angle) / 2
        symmetry_diff = abs(right_arm_angle - left_arm_angle)
        shoulders = np.array([landmark_list[11][1:], landmark_list[12][1:]]).mean(axis=0)
        hips = np.array([landmark_list[23][1:], landmark_list[24][1:]]).mean(axis=0)
        knees = np.array([landmark_list[25][1:], landmark_list[26][1:]]).mean(axis=0)
        back_angle = calculate_angle(shoulders, hips, knees)

        if stage == "down":
            if avg_arm_angle > 110:
                issues["arm_depth"] += 1
                form_feedback.append((random.choice(NEGATIVE_MESSAGES["push_up_arm"]), False))
            elif avg_arm_angle < 70:
                issues["arm_too_low"] += 1
                form_feedback.append(("Don’t go too low—ease up!", False))
            elif symmetry_diff > 20:
                issues["symmetry"] += 1
                form_feedback.append(("Arms uneven—sync them up!", False))
            elif back_angle < 160:
                issues["back"] += 1
                form_feedback.append((random.choice(NEGATIVE_MESSAGES["push_up_back"]), False))
            else:
                form_feedback.append((random.choice(POSITIVE_MESSAGES["push_up_arm"]), True))

    elif exercise_type == 'squat':
        right_knee_angle = detector.find_angle(img, 24, 26, 28)
        left_knee_angle = detector.find_angle(img, 23, 25, 27)
        current_angles = {'right_leg': right_knee_angle, 'left_leg': left_knee_angle}
        avg_knee_angle = (right_knee_angle + left_knee_angle) / 2
        symmetry_diff = abs(right_knee_angle - left_knee_angle)
        shoulders = np.array([landmark_list[11][1:], landmark_list[12][1:]]).mean(axis=0)
        hips = np.array([landmark_list[23][1:], landmark_list[24][1:]]).mean(axis=0)
        knees = np.array([landmark_list[25][1:], landmark_list[26][1:]]).mean(axis=0)
        back_angle = calculate_angle(shoulders, hips, knees)

        if stage == "down":
            if avg_knee_angle > 105:
                issues["leg_depth"] += 1
                form_feedback.append((random.choice(NEGATIVE_MESSAGES["squat_legs"]), False))
            elif avg_knee_angle < 65:
                issues["leg_too_low"] += 1
                form_feedback.append(("Too deep—back off a bit!", False))
            elif symmetry_diff > 20:
                issues["symmetry"] += 1
                form_feedback.append(("Legs uneven—align them!", False))
            elif back_angle < 150:
                issues["back"] += 1
                form_feedback.append((random.choice(NEGATIVE_MESSAGES["squat_back"]), False))
            else:
                form_feedback.append((random.choice(POSITIVE_MESSAGES["squat_legs"]), True))

    elif exercise_type == 'bicep_curl':
        right_shoulder_x = landmark_list[12][1]
        right_elbow_x = landmark_list[14][1]
        left_shoulder_x = landmark_list[11][1]
        left_elbow_x = landmark_list[13][1]
        right_arm_angle = detector.find_angle(img, 12, 14, 16)
        left_arm_angle = detector.find_angle(img, 11, 13, 15)
        current_angles = {'right_arm': right_arm_angle, 'left_arm': left_arm_angle}
        symmetry_diff = abs(right_arm_angle - left_arm_angle)

        if abs(right_elbow_x - right_shoulder_x) > 0.15 or abs(left_elbow_x - left_shoulder_x) > 0.15:
            issues["elbow_position"] += 1
            form_feedback.append((random.choice(NEGATIVE_MESSAGES["bicep_curl"]), False))
        elif symmetry_diff > 25:
            issues["symmetry"] += 1
            form_feedback.append(("Arms out of sync—match them!", False))
        elif right_arm_angle > 180 or left_arm_angle > 180:
            form_feedback.append(("Lower fully—great effort!", True))
        elif right_arm_angle < 90 and left_arm_angle < 90:
            form_feedback.append((random.choice(POSITIVE_MESSAGES["bicep_curl"]), True))
        else:
            form_feedback.append(("Solid curl—keep it up!", True))

    elif exercise_type == 'shoulder_press':
        back_angle = detector.find_angle(img, 11, 23, 25)
        right_arm_angle = detector.find_angle(img, 12, 14, 16)
        left_arm_angle = detector.find_angle(img, 11, 13, 15)
        current_angles = {'right_arm': right_arm_angle, 'left_arm': left_arm_angle}
        symmetry_diff = abs(right_arm_angle - left_arm_angle)

        if back_angle < 165:
            issues["back"] += 1
            form_feedback.append((random.choice(NEGATIVE_MESSAGES["shoulder_press"]), False))
        elif symmetry_diff > 20:
            issues["symmetry"] += 1
            form_feedback.append(("Arms uneven—press evenly!", False))
        elif stage == "up" and (abs(right_arm_angle - 180) > 30 or abs(left_arm_angle - 180) > 30):
            issues["arm_extension"] += 1
            form_feedback.append(("Extend fully—reach up!", False))
        else:
            form_feedback.append((random.choice(POSITIVE_MESSAGES["shoulder_press"]), True))

    if feedback_triggered:
        if form_feedback:
            form_feedback.append((random.choice(MOTIVATIONAL_MESSAGES), True))
        else:
            # Use the default positive key for the exercise type
            positive_key = default_positive_key.get(exercise_type, exercise_type)
            form_feedback = [(random.choice(POSITIVE_MESSAGES[positive_key]), True), (random.choice(MOTIVATIONAL_MESSAGES), True)]

    if issues and random.random() < 0.5:
        for issue in issues:
            feedback_history[exercise_type][issue] += issues[issue]
            if feedback_history[exercise_type][issue] > 5:
                tip_key = f"{exercise_type}_{issue.split('_')[0]}" if issue != "symmetry" else f"{exercise_type}_symmetry"
                form_feedback.append((random.choice(IMPROVEMENT_TIPS.get(tip_key, ["Work on your form!"])), False))
                break

    if len(form_feedback) > 2:
        form_feedback = form_feedback[:2]

    significant_change = False
    if prev_angles:
        for key in current_angles:
            if abs(current_angles[key] - prev_angles.get(key, current_angles[key])) > 20:
                significant_change = True
                break

    return form_feedback, current_angles, significant_change

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
        self.fatigue_level = 0.0
        self.fatigue_trend = 0.0
        self.fatigue_warnings = []
        self.prev_form_feedback = []
        self.prev_fatigue_score = 0.0
        self.feedback_history = {
            'push_up': defaultdict(int),
            'squat': defaultdict(int),
            'bicep_curl': defaultdict(int),
            'shoulder_press': defaultdict(int)
        }
        self.last_feedback_time = time.time()
        self.feedback_interval = 2
        self.prev_angles = {}
        self.counters = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}
        self.sets = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}  # New sets tracker
        self.baseline_speeds = {}
        self.last_activity_time = time.time()
        self.was_resting = {exercise: False for exercise in ['push_up', 'squat', 'bicep_curl', 'shoulder_press']}  # Track rest state

    def log_rep_data(self, exercise_type, timestamp, angles):
        self.rep_timestamps.append(timestamp)
        self.angle_history[exercise_type].append(angles)

    def detect_fatigue(self, exercise_type, window_size=5, min_reps=3, weight_factor=1.0):
        current_time = time.time()
        
        # Check for rest and decay fatigue
        rest_threshold = 20  # Seconds to consider a rest period
        rest_time = current_time - self.last_activity_time
        if rest_time > rest_threshold and self.fatigue_level > 0:
            decay_rate = 0.05
            self.fatigue_level *= math.exp(-decay_rate * rest_time)
            self.fatigue_level = max(0.1, self.fatigue_level)
            self.prev_fatigue_score = self.fatigue_level
            if self.counters[exercise_type] > 0 and not self.was_resting[exercise_type]:
                self.sets[exercise_type] += 1  # Increment set count
                self.counters[exercise_type] = 0  # Reset rep counter
                print(f"Set {self.sets[exercise_type]} completed for {exercise_type}. Reps reset.")
            self.was_resting[exercise_type] = True
            print(f"Rest detected for {rest_time:.1f}s. Fatigue decayed to {self.fatigue_level:.2f}")
            return self.fatigue_level, "Resting—fatigue easing up!"

        # Reset resting flag when activity resumes
        if rest_time <= rest_threshold and self.was_resting[exercise_type]:
            self.was_resting[exercise_type] = False

        # Normal fatigue calculation if active
        if len(self.rep_timestamps) < min_reps or not self.angle_history[exercise_type]:
            return self.fatigue_level, None

        total_reps = len(self.rep_timestamps)
        effective_window = min(window_size, total_reps)
        window_timestamps = self.rep_timestamps[-effective_window:]
        window_angles = self.angle_history[exercise_type][-effective_window:]

        # Adaptive baseline speed
        baseline_reps = min(5, total_reps)
        if exercise_type not in self.baseline_speeds or total_reps == baseline_reps:
            baseline_speed = np.mean(np.diff(self.rep_timestamps[:baseline_reps])) if total_reps >= min_reps else 0.0
            self.baseline_speeds[exercise_type] = baseline_speed
        else:
            baseline_speed = self.baseline_speeds[exercise_type]
        rep_times = np.diff(window_timestamps)
        avg_rep_speed = np.mean(rep_times) if len(rep_times) > 0 else baseline_speed
        speed_score = min(1.0, max(0.0, (avg_rep_speed - baseline_speed) / baseline_speed)) if baseline_speed > 0 else 0.0

        # Variability score
        key_angle = 'right_arm' if exercise_type in ['push_up', 'bicep_curl', 'shoulder_press'] else 'right_leg'
        angles = [rep[key_angle] for rep in window_angles]
        variability = np.std(angles)
        variability_score = min(1.0, variability / 15.0)

        # Flexible ROM targets
        rom_targets = {
            'push_up': (70, 110),
            'squat': (65, 105),
            'bicep_curl': (80, 180),
            'shoulder_press': (150, 210)
        }
        min_rom, max_rom = rom_targets[exercise_type]
        if exercise_type in ['push_up', 'squat']:
            rom = min(angles)
            rom_score = max(0.0, (min_rom - rom) / (min_rom - 50)) if rom < min_rom else 0.0
        else:
            rom = max(angles)
            rom_score = max(0.0, (rom - max_rom) / (250 - max_rom)) if rom > max_rom else 0.0
        rom_score = min(1.0, rom_score)

        # Fatigue score with weight factor
        fatigue_score = (0.4 * speed_score + 0.3 * variability_score + 0.3 * rom_score) * weight_factor
        fatigue_score = min(1.0, max(0.0, 0.7 * fatigue_score + 0.3 * self.prev_fatigue_score))
        
        # Update fatigue level and trend
        self.fatigue_level = max(self.fatigue_level, fatigue_score)
        fatigue_trend = fatigue_score - self.prev_fatigue_score
        self.fatigue_trend = 0.7 * fatigue_trend + 0.3 * self.fatigue_trend if self.prev_fatigue_score > 0 else fatigue_trend
        self.prev_fatigue_score = fatigue_score

        # Warning logic
        dominant_factor = max([('speed', speed_score), ('variability', variability_score), ('ROM', rom_score)], key=lambda x: x[1])[0]
        if fatigue_score < 0.2:
            warning = "Feeling fresh—great start!" if total_reps <= min_reps else None
        elif fatigue_score < 0.4:
            if dominant_factor == 'speed':
                warning = "Pace slowing a bit—stay steady!"
            elif dominant_factor == 'variability':
                warning = "Slight wobble—keep it smooth!"
            else:
                warning = "Range dipping—push a little more!"
        elif fatigue_score < 0.6:
            if dominant_factor == 'speed':
                warning = f"Slowing down on {exercise_type}—rest soon?"
            elif dominant_factor == 'variability':
                warning = "Form’s getting shaky—focus up!"
            else:
                warning = "Range dropping—extend fully!"
        else:
            if dominant_factor == 'speed':
                warning = "High fatigue: Pace way off—rest now!"
            elif dominant_factor == 'variability':
                warning = "High fatigue: Form’s off—take a break!"
            else:
                warning = "High fatigue: Range too low—pause!"

        if self.fatigue_trend < -0.1 and fatigue_score < 0.4:
            warning = f"Recovering well on {exercise_type}—nice bounce back!"

        print(f"{exercise_type} - Speed: {speed_score:.2f}, Variability: {variability_score:.2f}, ROM: {rom_score:.2f}, Fatigue: {self.fatigue_level:.2f}, Trend: {self.fatigue_trend:.2f}")
        return self.fatigue_level, warning

    def export_fatigue_data(self, filename="fatigue_data.csv"):
        data = {'timestamp': self.rep_timestamps, 'exercise': [], 'angles': [], 'sets': []}
        for ex, angles in self.angle_history.items():
            data['exercise'].extend([ex] * len(angles))
            data['angles'].extend(angles)
            data['sets'].extend([self.sets[ex]] * len(angles))
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
        stages = {'push_up': None, 'squat': None, 'left_bicep_curl': None, 'right_bicep_curl': None, 'shoulder_press': None}
        detector = pm.posture_detector()
        pose = mp.solutions.pose.Pose()
        exercise_name_map = {
            'push_up': 'Push-up',
            'squat': 'Squat',
            'bicep_curl': 'Curl',
            'shoulder_press': 'Press'
        }
        weight_factor = st.sidebar.slider("Weight Intensity (1.0 = Light, 1.5 = Heavy)", 1.0, 1.5, 1.0, 0.1)
        while cap.isOpened():
            if stop_button:
                break
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame.")
                break
            current_time = time.time()
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
                    stages['push_up'], self.counters['push_up'] = count_repetition_push_up(detector, frame, landmark_list, stages['push_up'], self.counters['push_up'], self)
                    stage = stages['push_up']
                elif current_prediction == 'squat':
                    exercise_type = 'squat'
                    stages['squat'], self.counters['squat'] = count_repetition_squat(detector, frame, landmark_list, stages['squat'], self.counters['squat'], self)
                    stage = stages['squat']
                elif current_prediction == 'barbell biceps curl':
                    exercise_type = 'bicep_curl'
                    stages['right_bicep_curl'], stages['left_bicep_curl'], self.counters['bicep_curl'] = count_repetition_bicep_curl(detector, frame, landmark_list, stages['right_bicep_curl'], stages['left_bicep_curl'], self.counters['bicep_curl'], self)
                elif current_prediction == 'shoulder press':
                    exercise_type = 'shoulder_press'
                    stages['shoulder_press'], self.counters['shoulder_press'] = count_repetition_shoulder_press(detector, frame, landmark_list, stages['shoulder_press'], self.counters['shoulder_press'], self)
                    stage = stages['shoulder_press']
                if exercise_type:
                    if current_time - self.last_feedback_time >= self.feedback_interval:
                        new_form_feedback, current_angles, significant_change = generate_feedback(
                            detector, frame, landmark_list, exercise_type, stage, self.feedback_history, self.prev_angles, self
                        )
                        fatigue_score, fatigue_warning = self.detect_fatigue(exercise_type, weight_factor=weight_factor)
                        self.fatigue_level = fatigue_score
                        new_fatigue_feedback = []
                        if fatigue_warning:
                            new_fatigue_feedback.append((fatigue_warning, False))
                            self.fatigue_warnings.append(fatigue_warning)
                        self.form_feedback = new_form_feedback
                        self.prev_form_feedback = new_form_feedback
                        self.last_feedback_time = current_time
                        self.prev_angles = current_angles
                        self.fatigue_feedback = new_fatigue_feedback
                    with form_feedback_placeholder.container():
                        st.subheader("Form Feedback")
                        if self.form_feedback:
                            for msg, is_positive in self.form_feedback[:2]:
                                if is_positive:
                                    st.success(msg)
                                else:
                                    st.error(msg)
                        else:
                            st.info("No form feedback yet.")
                    with fatigue_feedback_placeholder.container():
                        st.subheader("Fatigue Feedback")
                        st.metric("Fatigue Level", f"{self.fatigue_level:.0%}")
                        st.metric("Fatigue Trend", f"{self.fatigue_trend:+.0%}", delta_color="inverse")
                        if self.fatigue_feedback:
                            for msg, _ in self.fatigue_feedback:
                                st.warning(msg)
                        else:
                            st.success("No fatigue detected—keep going!")
            height, width, _ = frame.shape
            num_exercises = len(self.counters)
            vertical_spacing = height // (num_exercises + 1)
            cv2.rectangle(frame, (0, 0), (0, height), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, 0), (width, 0), (0, 0, 0), -1)
            short_name = exercise_name_map.get(current_prediction, current_prediction)
            cv2.putText(frame, f"Exercise: {short_name}", ((width - 290) // 2 + 100, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            for idx, (exercise, count) in enumerate(self.counters.items()):
                short_name = exercise_name_map.get(exercise, exercise)
                y_pos = (idx + 1) * vertical_spacing
                set_text = f"Set: {self.sets[exercise]} "
                rep_text = f"Reps: {count}"
                set_size = cv2.getTextSize(set_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                rep_size = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                overlay = frame.copy()
                cv2.rectangle(overlay, 
                            (10, y_pos - 20),
                            (10 + set_size[0] + rep_size[0], y_pos + 5),
                            (0, 0, 0), 
                            -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.putText(frame, set_text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, rep_text, (15 + set_size[0], y_pos),
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
                st.write("#### Sets and Repetitions")
                for exercise in self.counters.keys():
                    if self.sets[exercise] > 0 or self.counters[exercise] > 0:
                        st.metric(label=exercise_name_map.get(exercise, exercise), 
                                  value=f"{self.sets[exercise]} sets, {self.counters[exercise]} reps")
            with col2:
                st.write("#### Latest Form Feedback")
                if self.form_feedback:
                    msg, is_positive = self.form_feedback[0]
                    if is_positive:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.success("Great form! Keep it up!")
            st.write("### Fatigue Analysis")
            st.metric("Peak Fatigue Level", f"{self.fatigue_level:.0%}")
            st.metric("Final Fatigue Trend", f"{self.fatigue_trend:+.0%}")
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
                            if exercise_type and current_time - self.last_feedback_time >= self.feedback_interval:
                                stage_to_pass = stage if exercise_type in ['push_up', 'squat', 'shoulder_press'] else None
                                new_feedback, current_angles, significant_change = generate_feedback(
                                    detector, img, landmark_list, exercise_type, stage_to_pass, self.feedback_history, self.prev_angles, self
                                )
                                if significant_change or new_feedback != self.prev_form_feedback:
                                    self.form_feedback = new_feedback
                                    self.prev_form_feedback = new_feedback
                                    self.last_feedback_time = current_time
                                    self.prev_angles = current_angles
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
                current_time = time.time()
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
                    if exercise_type and current_time - self.last_feedback_time >= self.feedback_interval:
                        stage_to_pass = stage if exercise_type in ['push_up', 'squat', 'shoulder_press'] else None
                        new_feedback, current_angles, significant_change = generate_feedback(
                            detector, img, landmark_list, exercise_type, stage_to_pass, self.feedback_history, self.prev_angles, self
                        )
                        if significant_change or new_feedback != self.prev_form_feedback:
                            self.form_feedback = new_feedback
                            self.prev_form_feedback = new_feedback
                            self.last_feedback_time = current_time
                            self.prev_angles = current_angles
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
