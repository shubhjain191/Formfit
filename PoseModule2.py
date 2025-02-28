import mediapipe as mp
import math
import cv2
import time

class posture_detector():
    def __init__(self, mode=False, up_body=1, smooth=True, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.up_body = up_body
        self.smooth = smooth
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.up_body, self.smooth,
                                      min_detection_confidence=self.detection_con, 
                                      min_tracking_confidence=self.track_con)
        
        # Define custom drawing styles
        self.landmark_style = mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 255, 255), thickness=2, circle_radius=4)  # Cyan joints
        self.connection_style = mp.solutions.drawing_utils.DrawingSpec(
            color=(255, 0, 255), thickness=2)  # Magenta connections

    def find_person(self, img, draw=True):
        """Detect pose in the image and optionally draw landmarks."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img, 
                self.results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.landmark_style,  # Custom cyan joints
                self.connection_style  # Custom magenta connections
            )
        return img

    def find_landmarks(self, img, draw=True):
        """Extract landmark coordinates and optionally draw them."""
        self.landmark_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (0, 255, 255), cv2.FILLED)  # Cyan joints
        return self.landmark_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        """Calculate and optionally visualize the angle between three points."""
        # Get the landmarks
        x1, y1 = self.landmark_list[p1][1:]
        x1, y1 = max(0, x1), max(0, y1)  # Ensure coordinates are non-negative
        x2, y2 = self.landmark_list[p2][1:]
        x2, y2 = max(0, x2), max(0, y2)
        x3, y3 = self.landmark_list[p3][1:]
        x3, y3 = max(0, x3), max(0, y3)

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            # Use magenta for lines
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 0, 255), 3)
            # Use cyan for joints
            cv2.circle(img, (x1, y1), 4, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 4, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 4, (0, 255, 255), cv2.FILLED)
            # White text for angle
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        return angle

    def find_coordinate(self):
        pass

def main():
    cap = cv2.VideoCapture(0)
    detector = posture_detector()
    pTime = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = detector.find_person(frame, draw=True)  # Draw with custom styles
        landmark_list = detector.find_landmarks(img, draw=True)

        if len(landmark_list) != 0:
            # Example: Draw elbow angle (right arm)
            angle = detector.find_angle(img, 12, 14, 16, draw=True)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
