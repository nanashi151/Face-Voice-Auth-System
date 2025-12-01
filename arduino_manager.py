import serial
import time
import serial.tools.list_ports

class ArduinoDoorLock:
    def __init__(self, port=None, baud_rate=9600):
        self.connection = None
        self.baud_rate = baud_rate
        self.simulation_mode = False
        
        # Try to find the port automatically
        self.port = port if port else self.find_arduino_port()
        self.connect()

    def find_arduino_port(self):
        """Auto-detects Arduino port to avoid manual configuration."""
        ports = list(serial.tools.list_ports.comports())
        
        # List all available ports for debugging
        if ports:
            print(f"[ARDUINO] Scanning {len(ports)} port(s)...")
            for p in ports:
                print(f"[ARDUINO]   - {p.device}: {p.description}")
        
        # Look for Arduino
        for p in ports:
            # Common descriptions for Arduino and clones
            desc_lower = p.description.lower()
            if any(keyword in desc_lower for keyword in ["arduino", "ch340", "ch341", "cp210", "ftdi", "usb serial"]):
                print(f"[ARDUINO] ✓ Found Arduino-compatible device on {p.device}")
                return p.device
        
        return None

    def connect(self):
        """Establishes the connection."""
        if self.port:
            try:
                self.connection = serial.Serial(
                    self.port, 
                    self.baud_rate, 
                    timeout=1,
                    write_timeout=1  # Prevent hanging on write
                )
                time.sleep(2)  # Wait for Arduino reboot
                print(f"[ARDUINO] ✓ Connected successfully on {self.port}")
                print(f"[ARDUINO] Baud rate: {self.baud_rate}")
                self.simulation_mode = False
                
                # Optional: Send test signal
                try:
                    self.connection.write(b'T')  # Test command
                    print(f"[ARDUINO] Hardware communication verified")
                except:
                    print(f"[ARDUINO] ⚠ Connected but communication test failed")
                    
            except serial.SerialException as e:
                print(f"[ARDUINO] ✗ Connection Error: {e}")
                print(f"[ARDUINO] Possible causes:")
                print(f"          - Port {self.port} in use by another program")
                print(f"          - Incorrect baud rate (expected {self.baud_rate})")
                print(f"          - Permission denied (try running as admin/sudo)")
                self.connection = None
                self.simulation_mode = True
            except Exception as e:
                print(f"[ARDUINO] ✗ Unexpected Error: {e}")
                self.connection = None
                self.simulation_mode = True
        else:
            print("[ARDUINO] ⚠ Hardware not found. Running in SIMULATION MODE.")
            self.simulation_mode = True

    def unlock(self, duration=5):
        """
        Sends signal to unlock door.
        
        Args:
            duration (int): How long to keep door unlocked (seconds)
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.connection and self.connection.is_open:
            try:
                print(f"[ARDUINO] >>> Sending UNLOCK signal...")
                self.connection.write(b'1')  # Send '1' to unlock
                self.connection.flush()  # Ensure data is sent
                
                # Visual feedback
                for i in range(duration):
                    print(f"[ARDUINO] Door unlocked... ({duration - i}s remaining)")
                    time.sleep(1)
                
                print(f"[ARDUINO] >>> Sending LOCK signal...")
                self.connection.write(b'0')  # Send '0' to lock
                self.connection.flush()
                print(f"[ARDUINO] ✓ Door locked")
                return True
                
            except serial.SerialException as e:
                print(f"[ARDUINO] ✗ Communication error: {e}")
                return False
            except Exception as e:
                print(f"[ARDUINO] ✗ Unexpected error: {e}")
                return False
        else:
            # SIMULATION MODE (when Arduino not connected)
            return self._simulate_unlock(duration)

    def _simulate_unlock(self, duration=10):
        """Simulates door unlock for testing without hardware."""
        print("\n" + "="*50)
        print("║  [HARDWARE SIMULATION MODE]")
        print("║")
        print("║  >>> DOOR UNLOCKING... <<<")
        print("║  Servo motor rotating 90° clockwise...")
        print("="*50)
        
        # Simulate unlock duration with countdown
        for i in range(duration):
            remaining = duration - i
            print(f"║  Door remains UNLOCKED... ({remaining}s remaining)")
            time.sleep(1)
        
        print("║")
        print("║  >>> DOOR LOCKING... <<<")
        print("║  Servo motor rotating 90° counter-clockwise...")
        print("║")
        print("║  ✓ DOOR SECURED")
        print("="*50 + "\n")
        return True

    def is_connected(self):
        """Check if Arduino is connected and responsive."""
        return self.connection is not None and self.connection.is_open

    def get_status(self):
        """Returns current connection status."""
        if self.is_connected():
            return {
                'connected': True,
                'port': self.port,
                'baud_rate': self.baud_rate,
                'simulation': False
            }
        else:
            return {
                'connected': False,
                'port': None,
                'baud_rate': self.baud_rate,
                'simulation': self.simulation_mode
            }

    def reconnect(self):
        """Attempt to reconnect to Arduino."""
        print("[ARDUINO] Attempting to reconnect...")
        self.close()
        time.sleep(1)
        self.port = self.find_arduino_port()
        self.connect()
        return self.is_connected()

    def close(self):
        """Safely close the connection."""
        if self.connection:
            try:
                if self.connection.is_open:
                    # Send lock command before closing
                    self.connection.write(b'0')
                    time.sleep(0.5)
                    self.connection.close()
                    print("[ARDUINO] Connection closed safely")
            except Exception as e:
                print(f"[ARDUINO] Error during close: {e}")
            finally:
                self.connection = None

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()


# ========================================
# TESTING CODE (Run this file directly to test Arduino connection)
# ========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ARDUINO DOOR LOCK - CONNECTION TEST")
    print("="*60 + "\n")
    
    # Initialize
    arduino = ArduinoDoorLock()
    
    # Show status
    status = arduino.get_status()
    print(f"\n[STATUS] Connected: {status['connected']}")
    print(f"[STATUS] Port: {status['port']}")
    print(f"[STATUS] Simulation: {status['simulation']}")
    
    # Test unlock
    print("\n[TEST] Testing unlock sequence...")
    input("Press ENTER to start test (or Ctrl+C to cancel)...")
    
    success = arduino.unlock(duration=3)
    
    if success:
        print("\n[TEST] ✓ Test completed successfully!")
    else:
        print("\n[TEST] ✗ Test failed!")
    
    # Cleanup
    arduino.close()
    print("\n[TEST] Test finished.\n")