import cv2
import mediapipe as mp
import pyautogui
import time
import webbrowser

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

GAME_URL = "https://poki.com/en/g/subway-surfers"   # or another Subway Surfers web link

def open_subway_surfers():
    webbrowser.open(GAME_URL)
    time.sleep(8)   # wait for game to load; adjust if needed

def is_hands_joined(landmarks, w, h, thresh=60):
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

    x1, y1 = int(left_wrist.x * w), int(left_wrist.y * h)
    x2, y2 = int(right_wrist.x * w), int(right_wrist.y * h)

    dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return dist < thresh

def main():
    open_subway_surfers()

    cap = cv2.VideoCapture(0)
    last_action_time = 0
    cooldown = 0.35   # seconds

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose.process(rgb)
            rgb.flags.writeable = True

            gesture = None

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark

                nose = lm[mp_pose.PoseLandmark.NOSE]
                ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
                rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]

                nose_y = nose.y
                shoulders_y = (ls.y + rs.y) / 2
                hips_y = (lh.y + rh.y) / 2

                # Jump (up arrow)
                if nose_y < shoulders_y - 0.05:
                    gesture = "jump"
                # Slide (down arrow)
                elif nose_y > hips_y - 0.05:
                    gesture = "slide"
                else:
                    center_x = 0.5
                    shoulders_x = (ls.x + rs.x) / 2
                    if shoulders_x < center_x - 0.06:
                        gesture = "left"
                    elif shoulders_x > center_x + 0.06:
                        gesture = "right"

                if is_hands_joined(lm, w, h):
                    gesture = "start"

                mp_drawing.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            now = time.time()
            if gesture and now - last_action_time > cooldown:
                if gesture == "left":
                    pyautogui.press("left")
                elif gesture == "right":
                    pyautogui.press("right")
                elif gesture == "jump":
                    pyautogui.press("up")
                elif gesture == "slide":
                    pyautogui.press("down")
                elif gesture == "start":
                    pyautogui.press("space")  # start / continue

                last_action_time = now

            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Subway Surfers Gesture Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
