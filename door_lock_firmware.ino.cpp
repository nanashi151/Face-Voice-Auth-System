/*
 * Smart Classroom Access System - Door Lock Controller
 * Arduino Uno + Servo Motor
 * 
 * Hardware:
 * - Arduino Uno
 * - Servo Motor (Pin 9)
 * - Built-in LED (Pin 13)
 * 
 * Commands from Python:
 * '1' = Unlock door
 * '0' = Lock door
 * 'T' = Test connection (blink LED)
 */

#include <Servo.h>

// ==========================================
// CONFIGURATION
// ==========================================
const int SERVO_PIN = 9;           // Servo motor signal wire
const int LED_PIN = 13;            // Built-in LED

// Servo Angles
const int LOCKED_ANGLE = 0;        // Door locked position
const int UNLOCKED_ANGLE = 90;     // Door unlocked position

// Safety Settings
const unsigned long AUTO_LOCK_TIME = 10000;  // Auto-lock after 10 seconds

// State Tracking
Servo doorLockServo;
bool isUnlocked = false;
unsigned long unlockStartTime = 0;

// ==========================================
// SETUP
// ==========================================
void setup() {
  // Initialize Serial (9600 baud - must match Python)
  Serial.begin(9600);
  
  // Setup Servo
  doorLockServo.attach(SERVO_PIN);
  
  // Setup LED
  pinMode(LED_PIN, OUTPUT);
  
  // Force LOCKED state on startup
  lockDoor();
  
  // Brief LED flash to show ready
  digitalWrite(LED_PIN, HIGH);
  delay(300);
  digitalWrite(LED_PIN, LOW);
}

// ==========================================
// MAIN LOOP
// ==========================================
void loop() {
  // Check for commands from Python
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    switch (command) {
      case '1':  // UNLOCK
        unlockDoor();
        break;
        
      case '0':  // LOCK
        lockDoor();
        break;
        
      case 'T':  // TEST (blink LED 3 times)
        testConnection();
        break;
    }
  }
  
  // Auto-lock safety check
  if (isUnlocked) {
    unsigned long elapsed = millis() - unlockStartTime;
    if (elapsed >= AUTO_LOCK_TIME) {
      lockDoor();  // Auto-lock after timeout
    }
  }
  
  delay(10);  // Small delay to prevent CPU overload
}

// ==========================================
// DOOR CONTROL FUNCTIONS
// ==========================================
void lockDoor() {
  // Smooth movement to locked position
  smoothServoMove(doorLockServo.read(), LOCKED_ANGLE);
  
  // Update state
  isUnlocked = false;
  digitalWrite(LED_PIN, LOW);  // LED OFF = Locked
}

void unlockDoor() {
  // Smooth movement to unlocked position
  smoothServoMove(doorLockServo.read(), UNLOCKED_ANGLE);
  
  // Update state
  isUnlocked = true;
  unlockStartTime = millis();  // Start auto-lock timer
  digitalWrite(LED_PIN, HIGH);  // LED ON = Unlocked
}

void smoothServoMove(int fromAngle, int toAngle) {
  // Smooth servo movement to reduce mechanical stress
  int step = (fromAngle < toAngle) ? 1 : -1;
  
  for (int angle = fromAngle; angle != toAngle; angle += step) {
    doorLockServo.write(angle);
    delay(15);  // Speed control (lower = faster, higher = smoother)
  }
  
  doorLockServo.write(toAngle);  // Ensure exact final position
}

void testConnection() {
  // Blink LED 3 times to confirm connection
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }
}