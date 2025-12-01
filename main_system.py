import cv2
import speech_recognition as sr
import os
import csv
from datetime import datetime
import time
import numpy as np
import pickle
import random
from scipy.spatial import distance as dist
from arduino_manager import ArduinoDoorLock

# ==========================================
# CONFIGURATION
# ==========================================
USE_SIMULATION = True
CHECK_LIVENESS = True
USE_VOICE_BIOMETRICS = False  # Set True if you want to store voice samples

KNOWN_FACES_DIR = "known_faces"
VOICE_SAMPLES_DIR = "voice_samples"  # For voice biometrics (optional)
LOG_FILE = "access_log.csv"
SIMULATION_IMAGE = "test_feed.jpg"

# Liveness Detection Settings
LIVENESS_TIMEOUT = 10  # seconds
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold

# Voice Challenge Words Bank
WORD_BANK = [
    "access", "blue", "campus", "door", "enter", "faculty", 
    "green", "hello", "instructor", "justice", "key", "light",
    "morning", "notebook", "open", "professor", "quiet", "room",
    "student", "teacher", "university", "verify", "welcome", "year",
    "zero", "alpha", "bravo", "charlie", "delta", "echo"
]
# ==========================================

class VoiceAuthenticator:
    """Handles voice challenge generation and verification."""
    
    def __init__(self, word_bank=WORD_BANK):
        self.word_bank = word_bank
        self.recognizer = sr.Recognizer()
        
    def generate_challenge(self, num_words=3):
        """Generate random words for voice challenge."""
        return random.sample(self.word_bank, num_words)
    
    def verify_challenge(self, challenge_words, timeout=5):
        """
        Verify user speaks the challenge words.
        Returns: (success, spoken_text, match_percentage)
        """
        if USE_SIMULATION:
            # Simulation mode
            print(f"\n[VOICE] Challenge Words: {' '.join(challenge_words).upper()}")
            print(f"[VOICE] Type these words:")
            spoken = input(">> ").lower().strip()
            
            # Check match
            spoken_words = spoken.split()
            matches = sum(1 for word in challenge_words if word in spoken_words)
            match_percentage = (matches / len(challenge_words)) * 100
            
            success = match_percentage >= 66.67  # At least 2 out of 3
            return success, spoken, match_percentage
        
        else:
            # Live mode - use microphone
            with sr.Microphone() as source:
                print(f"\n[VOICE] Say these words: {' '.join(challenge_words).upper()}")
                print("[VOICE] Listening...")
                
                try:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
                    spoken_text = self.recognizer.recognize_google(audio).lower()
                    
                    print(f"[VOICE] You said: '{spoken_text}'")
                    
                    # Check how many words match
                    spoken_words = spoken_text.split()
                    matches = sum(1 for word in challenge_words if word in spoken_words)
                    match_percentage = (matches / len(challenge_words)) * 100
                    
                    success = match_percentage >= 66.67  # At least 2 out of 3
                    return success, spoken_text, match_percentage
                    
                except sr.WaitTimeoutError:
                    print("[VOICE] Timeout - no speech detected")
                    return False, "", 0
                except sr.UnknownValueError:
                    print("[VOICE] Could not understand audio")
                    return False, "", 0
                except Exception as e:
                    print(f"[VOICE] Error: {e}")
                    return False, "", 0


class LivenessDetector:
    """Enhanced liveness detection with multiple random tests."""
    
    def __init__(self, ear_threshold=0.25):
        self.EAR_THRESHOLD = ear_threshold
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Available liveness tests
        self.available_tests = ['blink', 'look_left', 'look_right']
        
        # Test tracking
        self.blink_counter = 0
        self.frames_below_threshold = 0
    
    def generate_random_test(self):
        """Randomly select a liveness test."""
        return random.choice(self.available_tests)
    
    def calculate_ear(self, eye):
        """Calculate Eye Aspect Ratio."""
        x, y, w, h = eye
        if w == 0:
            return 0
        return float(h) / float(w)
    
    def detect_eyes(self, face_roi):
        """Detect eyes within face region."""
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )
        return eyes
    
    def check_blink(self, frame, face_location):
        """Check for blink. Returns: (blink_detected, total_blinks)"""
        x, y, w, h = face_location
        face_roi = frame[y:y+h, x:x+w]
        
        eyes = self.detect_eyes(face_roi)
        
        if len(eyes) < 2:
            return False, self.blink_counter
        
        ear_values = [self.calculate_ear(eye) for eye in eyes[:2]]
        avg_ear = np.mean(ear_values)
        
        if avg_ear < self.EAR_THRESHOLD:
            self.frames_below_threshold += 1
        else:
            if self.frames_below_threshold >= 2:
                self.blink_counter += 1
                self.frames_below_threshold = 0
                return True, self.blink_counter
            self.frames_below_threshold = 0
        
        return False, self.blink_counter
    
    def check_head_turn(self, frame, face_location, direction):
        """
        Check if head is turned left or right.
        Returns: (turned, confidence)
        """
        x, y, w, h = face_location
        face_roi = frame[y:y+h, x:x+w]
        
        eyes = self.detect_eyes(face_roi)
        
        if len(eyes) < 2:
            return False, 0
        
        # Sort eyes by x-coordinate
        eyes_sorted = sorted(eyes, key=lambda e: e[0])
        left_eye, right_eye = eyes_sorted[0], eyes_sorted[1]
        
        # Calculate horizontal distance between eyes
        eye_distance = right_eye[0] - (left_eye[0] + left_eye[2])
        
        # Get face width
        face_width = w
        
        # Calculate ratio (smaller ratio = head turned)
        ratio = eye_distance / face_width if face_width > 0 else 0
        
        # Thresholds for head turn detection
        if direction == 'left':
            # When looking left, right eye becomes less visible
            # Right eye should be closer to right edge
            turned = (right_eye[0] + right_eye[2]) > (face_width * 0.7)
            confidence = 0.8 if turned else 0.3
        elif direction == 'right':
            # When looking right, left eye becomes less visible
            # Left eye should be closer to left edge
            turned = left_eye[0] < (face_width * 0.3)
            confidence = 0.8 if turned else 0.3
        else:
            return False, 0
        
        return turned, confidence
    
    def reset(self):
        """Reset counters."""
        self.blink_counter = 0
        self.frames_below_threshold = 0


class SmartAccessSystem:
    def __init__(self):
        print("--- SMART ACCESS SYSTEM v2.0 ---")
        print("    Enhanced Security Features")
        print("="*40)
        
        self.arduino = ArduinoDoorLock()
        
        # Face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("[ERROR] Could not load face cascade!")
            exit(1)
        
        # Face recognizer
        try:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            print("[INIT] ✓ Face Recognition Module")
        except:
            print("[ERROR] OpenCV face module not available!")
            exit(1)

        # Enhanced modules
        self.liveness_detector = LivenessDetector(ear_threshold=EAR_THRESHOLD)
        self.voice_authenticator = VoiceAuthenticator()
        
        print("[INIT] ✓ Liveness Detection (Random Tests)")
        print("[INIT] ✓ Voice Challenge System")

        self.known_face_names = {}
        self.load_known_faces()
        
        if not USE_SIMULATION:
            self.video_capture = cv2.VideoCapture(0)
        else:
            print("[INIT] ✓ Simulation Mode Active")
        
        print("="*40 + "\n")

    def detect_faces(self, image):
        """Detect faces using Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        return faces, gray

    def load_known_faces(self):
        """Load and train face recognizer."""
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)
            print(f"[INFO] Created directory: {KNOWN_FACES_DIR}")
            return

        print("[DATA] Loading face database...")
        
        faces = []
        labels = []
        label_map = {}
        current_id = 0
        
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(KNOWN_FACES_DIR, filename)
                try:
                    img = cv2.imread(path)
                    if img is None: 
                        continue
                    
                    detected_faces, gray = self.detect_faces(img)
                    
                    if len(detected_faces) == 0:
                        print(f"   ⚠ No face: {filename}")
                        continue
                    
                    face = detected_faces[0] if len(detected_faces) == 1 else max(detected_faces, key=lambda f: f[2]*f[3])
                    
                    x, y, w, h = face
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (200, 200))
                    
                    faces.append(face_roi)
                    labels.append(current_id)
                    
                    name = os.path.splitext(filename)[0]
                    label_map[current_id] = name
                    current_id += 1
                    
                    print(f"   ✓ Loaded: {name}")
                        
                except Exception as e:
                    print(f"   ✗ Error: {filename}")
        
        if len(faces) == 0:
            print("[WARNING] No faces loaded!")
            return
        
        print("[TRAIN] Training recognition model...")
        try:
            self.face_recognizer.train(faces, np.array(labels))
            self.known_face_names = label_map
            
            self.face_recognizer.save('face_model.yml')
            with open('label_map.pkl', 'wb') as f:
                pickle.dump(label_map, f)
            
            print(f"[DATA] ✓ {len(label_map)} users registered")
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")

    def run_liveness_test(self, frame, face_location):
        """
        Run random liveness test.
        Returns: (success, test_type)
        """
        # Select random test
        test_type = self.liveness_detector.generate_random_test()
        
        if USE_SIMULATION:
            # Simulation mode
            print("\n" + "="*50)
            print(f"[LIVENESS] Test Type: {test_type.upper().replace('_', ' ')}")
            print("="*50)
            
            if test_type == 'blink':
                print("[LIVENESS] Please blink twice")
            elif test_type == 'look_left':
                print("[LIVENESS] Please look to your LEFT")
            elif test_type == 'look_right':
                print("[LIVENESS] Please look to your RIGHT")
            
            response = input("Simulate successful test? (y/n): ").lower()
            return response == 'y', test_type
        
        # Live mode
        print("\n" + "="*50)
        
        if test_type == 'blink':
            return self._test_blink(), test_type
        elif test_type == 'look_left':
            return self._test_head_turn('left'), test_type
        elif test_type == 'look_right':
            return self._test_head_turn('right'), test_type
        
        return False, test_type
    
    def _test_blink(self):
        """Test: Blink twice."""
        print("[LIVENESS] Please BLINK twice")
        print(f"[LIVENESS] Timeout: {LIVENESS_TIMEOUT}s")
        print("="*50)
        
        self.liveness_detector.reset()
        required_blinks = 2
        start_time = time.time()
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, test_frame = cap.read()
            if not ret:
                break
            
            elapsed = time.time() - start_time
            if elapsed > LIVENESS_TIMEOUT:
                print("[LIVENESS] ✗ TIMEOUT")
                cap.release()
                cv2.destroyAllWindows()
                return False
            
            faces, _ = self.liveness_detector.face_cascade.detectMultiScale(
                cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
            )
            
            if len(faces) > 0:
                face = faces[0]
                x, y, w, h = face
                
                blink_occurred, total_blinks = self.liveness_detector.check_blink(
                    test_frame, face
                )
                
                if blink_occurred:
                    print(f"[LIVENESS] ✓ Blink {total_blinks}/{required_blinks}")
                    
                    if total_blinks >= required_blinks:
                        print("[LIVENESS] ✓✓ TEST PASSED!")
                        cap.release()
                        cv2.destroyAllWindows()
                        return True
                
                # Draw feedback
                cv2.rectangle(test_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(test_frame, f"Blink {total_blinks}/{required_blinks}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(test_frame, f"Time: {int(LIVENESS_TIMEOUT - elapsed)}s",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Liveness Test - Blink', test_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    def _test_head_turn(self, direction):
        """Test: Turn head left or right."""
        direction_text = "LEFT" if direction == 'left' else "RIGHT"
        print(f"[LIVENESS] Please look to your {direction_text}")
        print(f"[LIVENESS] Timeout: {LIVENESS_TIMEOUT}s")
        print("="*50)
        
        start_time = time.time()
        success_frames = 0
        required_frames = 10  # Must maintain turn for 10 frames
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, test_frame = cap.read()
            if not ret:
                break
            
            elapsed = time.time() - start_time
            if elapsed > LIVENESS_TIMEOUT:
                print("[LIVENESS] ✗ TIMEOUT")
                cap.release()
                cv2.destroyAllWindows()
                return False
            
            faces, _ = self.liveness_detector.face_cascade.detectMultiScale(
                cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
            )
            
            if len(faces) > 0:
                face = faces[0]
                x, y, w, h = face
                
                turned, confidence = self.liveness_detector.check_head_turn(
                    test_frame, face, direction
                )
                
                if turned:
                    success_frames += 1
                    if success_frames >= required_frames:
                        print(f"[LIVENESS] ✓✓ TEST PASSED!")
                        cap.release()
                        cv2.destroyAllWindows()
                        return True
                else:
                    success_frames = 0
                
                # Draw feedback
                color = (0, 255, 0) if success_frames > 0 else (0, 165, 255)
                cv2.rectangle(test_frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw arrow
                center_x = x + w // 2
                center_y = y + h // 2
                if direction == 'left':
                    cv2.arrowedLine(test_frame, (center_x, center_y), 
                                   (center_x - 100, center_y), (0, 255, 0), 3)
                else:
                    cv2.arrowedLine(test_frame, (center_x, center_y),
                                   (center_x + 100, center_y), (0, 255, 0), 3)
                
                cv2.putText(test_frame, f"Look {direction_text}: {success_frames}/{required_frames}",
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(test_frame, f"Time: {int(LIVENESS_TIMEOUT - elapsed)}s",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(f'Liveness Test - Look {direction_text}', test_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        return False

    def get_input_feed(self):
        if USE_SIMULATION:
            if os.path.exists(SIMULATION_IMAGE):
                frame = cv2.imread(SIMULATION_IMAGE)
                if frame is None: 
                    return False, None
                cv2.putText(frame, "SIMULATION MODE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return True, frame
            else:
                print(f"[ERROR] Not found: {SIMULATION_IMAGE}")
                return False, None
        else:
            return self.video_capture.read()

    def run(self):
        print("\n[SYSTEM] Starting Enhanced Access System...")
        if USE_SIMULATION:
            print("[CONTROLS] 'c'=close & continue | 'q'=quit\n")
        else:
            print("[CONTROLS] 'q'=quit\n")
        
        if len(self.known_face_names) == 0:
            print("[ERROR] No users registered!")
            return
        
        print(f"[INFO] Registered: {', '.join(self.known_face_names.values())}\n")
        
        while True:
            ret, frame = self.get_input_feed()
            if not ret or frame is None: 
                break

            display_frame = frame.copy()
            detected_faces, gray = self.detect_faces(frame)

            for (x, y, w, h) in detected_faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))
                
                label, confidence = self.face_recognizer.predict(face_roi)
                
                if confidence < 70:
                    name = self.known_face_names.get(label, "Unknown")
                    match_confidence = (100 - confidence) / 100
                else:
                    name = "Unknown"
                    match_confidence = 0
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                label_text = f"{name}"
                if name != "Unknown":
                    label_text += f" ({match_confidence:.0%})"
                
                cv2.rectangle(display_frame, (x, y+h-35), (x+w, y+h), color, cv2.FILLED)
                cv2.putText(display_frame, label_text, (x+6, y+h-6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                if name != "Unknown":
                    print(f"\n{'='*50}")
                    print(f"[FACE] ✓ Identified: {name} ({match_confidence:.0%})")
                    print(f"{'='*50}")
                    
                    cv2.imshow('Smart Access System v2.0', display_frame)
                    cv2.waitKey(1)
                    
                    # LIVENESS TEST (Random)
                    if CHECK_LIVENESS:
                        liveness_passed, test_type = self.run_liveness_test(frame, (x, y, w, h))
                        
                        if not liveness_passed:
                            print(f"[SECURITY] ✗ Liveness failed ({test_type})")
                            print(f"[ACCESS] ✗✗ DENIED - Possible spoofing")
                            self._log_access(name, "DENIED - Liveness failed", match_confidence)
                            print(f"{'='*50}\n")
                            continue
                        else:
                            print(f"[SECURITY] ✓ Liveness verified ({test_type})")
                    
                    # VOICE CHALLENGE (Random Words)
                    challenge_words = self.voice_authenticator.generate_challenge(3)
                    success, spoken, match_pct = self.voice_authenticator.verify_challenge(challenge_words)
                    
                    if success:
                        print(f"[VOICE] ✓ Challenge passed ({match_pct:.0f}% match)")
                        print(f"[ACCESS] ✓✓✓ GRANTED to {name}")
                        print(f"{'='*50}\n")
                        
                        self.arduino.unlock()
                        self._log_access(name, "GRANTED", match_confidence)
                        time.sleep(2)
                    else:
                        print(f"[VOICE] ✗ Challenge failed ({match_pct:.0f}% match)")
                        print(f"[ACCESS] ✗✗ DENIED")
                        self._log_access(name, f"DENIED - Voice challenge failed", match_confidence)
                        print(f"{'='*50}\n")

            cv2.imshow('Smart Access System v2.0', display_frame)
            
            if USE_SIMULATION:
                print("\n[CONTROL] 'c'=close window | 'q'=quit")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    cv2.destroyAllWindows()
                    response = input("Run another simulation? (y/n): ").lower()
                    if response != 'y':
                        break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if not USE_SIMULATION and hasattr(self, 'video_capture'):
            self.video_capture.release()
        cv2.destroyAllWindows()
        print("\n[SYSTEM] Stopped")
    
    def _log_access(self, name, status, confidence):
        """Log access attempt."""
        try:
            file_exists = os.path.exists(LOG_FILE)
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Timestamp', 'Name', 'Status', 'Confidence'])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    name, status, f"{confidence:.2f}"
                ])
        except Exception as e:
            print(f"[LOG ERROR] {e}")


if __name__ == "__main__":
    try:
        system = SmartAccessSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted")
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()