"""
CVSU Smart Access System - Admin Panel
Standalone administration tool for managing users and viewing logs
"""

import cv2
import os
import sqlite3
import csv
from datetime import datetime
import time
import getpass
import hashlib

# ==========================================
# CONFIGURATION
# ==========================================
KNOWN_FACES_DIR = "known_faces"
LOG_FILE = "access_log.csv"
DATABASE_FILE = "admin_database.db"

# Admin credentials (change these!)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_HASH = "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918"  # "admin"
# To generate new hash: hashlib.sha256("your_password".encode()).hexdigest()

# ==========================================
# DATABASE MANAGER
# ==========================================
class AdminDatabase:
    """Manages user database and audit logs."""
    
    def __init__(self, db_path=DATABASE_FILE):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                employee_id TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'faculty', 'staff')),
                face_image_file TEXT,
                created_at TEXT NOT NULL,
                created_by TEXT,
                last_modified TEXT,
                active INTEGER DEFAULT 1
            )
        """)
        
        # Admin action logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_user TEXT NOT NULL,
                action TEXT NOT NULL,
                target_user TEXT,
                timestamp TEXT NOT NULL,
                details TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        print("[DATABASE] ✓ Admin database initialized")
    
    def add_user(self, name, employee_id, role, face_file, admin_user):
        """Add new user to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO users (name, employee_id, role, face_image_file, created_at, created_by)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, employee_id, role, face_file, 
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"), admin_user))
            
            user_id = cursor.lastrowid
            
            # Log the action
            cursor.execute("""
                INSERT INTO admin_logs (admin_user, action, target_user, timestamp, details)
                VALUES (?, ?, ?, ?, ?)
            """, (admin_user, "ADD_USER", employee_id, 
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  f"Added {name} as {role}"))
            
            conn.commit()
            conn.close()
            
            return user_id
        except sqlite3.IntegrityError:
            print(f"[ERROR] Employee ID {employee_id} already exists")
            return None
        except Exception as e:
            print(f"[ERROR] Database error: {e}")
            return None
    
    def get_all_users(self, include_inactive=False):
        """Retrieve all users."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if include_inactive:
            cursor.execute("""
                SELECT id, name, employee_id, role, active, created_at
                FROM users ORDER BY name
            """)
        else:
            cursor.execute("""
                SELECT id, name, employee_id, role, active, created_at
                FROM users WHERE active = 1 ORDER BY name
            """)
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def deactivate_user(self, employee_id, admin_user):
        """Soft delete user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users 
            SET active = 0, last_modified = ?
            WHERE employee_id = ?
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), employee_id))
        
        # Log action
        cursor.execute("""
            INSERT INTO admin_logs (admin_user, action, target_user, timestamp, details)
            VALUES (?, ?, ?, ?, ?)
        """, (admin_user, "DEACTIVATE_USER", employee_id,
              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              f"Deactivated user {employee_id}"))
        
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return affected > 0
    
    def reactivate_user(self, employee_id, admin_user):
        """Reactivate deactivated user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users 
            SET active = 1, last_modified = ?
            WHERE employee_id = ?
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), employee_id))
        
        # Log action
        cursor.execute("""
            INSERT INTO admin_logs (admin_user, action, target_user, timestamp, details)
            VALUES (?, ?, ?, ?, ?)
        """, (admin_user, "REACTIVATE_USER", employee_id,
              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              f"Reactivated user {employee_id}"))
        
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return affected > 0
    
    def get_admin_logs(self, limit=50):
        """Get recent admin actions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, admin_user, action, target_user, details
            FROM admin_logs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def search_user(self, search_term):
        """Search users by name or employee ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, employee_id, role, active
            FROM users
            WHERE name LIKE ? OR employee_id LIKE ?
        """, (f"%{search_term}%", f"%{search_term}%"))
        
        results = cursor.fetchall()
        conn.close()
        return results


# ==========================================
# AUTHENTICATION
# ==========================================
def hash_password(password):
    """Hash password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_admin():
    """Verify admin credentials."""
    print("\n" + "="*60)
    print("  ADMIN AUTHENTICATION REQUIRED")
    print("="*60)
    
    attempts = 0
    max_attempts = 3
    
    while attempts < max_attempts:
        username = input("Username: ").strip()
        password = getpass.getpass("Password: ")
        
        if username == ADMIN_USERNAME and hash_password(password) == ADMIN_PASSWORD_HASH:
            print("[AUTH] ✓ Authentication successful\n")
            return True
        
        attempts += 1
        remaining = max_attempts - attempts
        if remaining > 0:
            print(f"[AUTH] ✗ Incorrect credentials ({remaining} attempts remaining)\n")
        else:
            print("[AUTH] ✗ Too many failed attempts. Access denied.")
    
    return False


# ==========================================
# USER MANAGEMENT FUNCTIONS
# ==========================================
def add_new_user(db, admin_user):
    """Interactive user registration."""
    print("\n" + "="*60)
    print("  ADD NEW USER")
    print("="*60)
    
    # Get user details
    name = input("Full Name: ").strip()
    if not name:
        print("[ERROR] Name is required")
        return False
    
    employee_id = input("Employee/Student ID: ").strip()
    if not employee_id:
        print("[ERROR] Employee ID is required")
        return False
    
    print("\nRole options: admin, faculty, staff")
    role = input("Role: ").strip().lower()
    if role not in ['admin', 'faculty', 'staff']:
        print("[ERROR] Invalid role")
        return False
    
    # Capture face photo
    print(f"\n[CAPTURE] Preparing to capture face for {name}")
    print("[INFO] Position face clearly in frame")
    print("[INFO] Press SPACE to capture, ESC to cancel\n")
    
    input("Press ENTER to open camera...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Cannot access camera")
        return False
    
    captured_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from camera")
            break
        
        # Display instructions
        cv2.putText(frame, "Press SPACE to capture", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press ESC to cancel", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw center guide
        height, width = frame.shape[:2]
        cv2.circle(frame, (width//2, height//2), 150, (0, 255, 0), 2)
        
        cv2.imshow('Capture Face Photo', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # SPACE
            captured_frame = frame.copy()
            print("[CAPTURE] ✓ Photo captured")
            break
        elif key == 27:  # ESC
            print("[CAPTURE] Cancelled by user")
            cap.release()
            cv2.destroyAllWindows()
            return False
    
    cap.release()
    cv2.destroyAllWindows()
    
    if captured_frame is None:
        print("[ERROR] No photo captured")
        return False
    
    # Save photo
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
    
    filename = f"{name.replace(' ', '_')}_{employee_id}.jpg"
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    
    cv2.imwrite(filepath, captured_frame)
    print(f"[SAVE] ✓ Photo saved to {filepath}")
    
    # Add to database
    user_id = db.add_user(name, employee_id, role, filename, admin_user)
    
    if user_id:
        print(f"\n[SUCCESS] ✓✓ {name} (ID: {employee_id}) registered successfully!")
        print(f"[INFO] User database ID: {user_id}")
        print(f"[INFO] Remember to retrain the face recognition model")
        return True
    else:
        # Cleanup photo if database failed
        os.remove(filepath)
        print("[ERROR] Failed to add user to database")
        return False


def view_all_users(db):
    """Display all registered users."""
    print("\n" + "="*60)
    print("  REGISTERED USERS")
    print("="*60)
    
    users = db.get_all_users(include_inactive=True)
    
    if not users:
        print("\n[INFO] No users found in database\n")
        return
    
    print(f"\n{'ID':<5} {'Name':<25} {'Emp ID':<15} {'Role':<10} {'Status':<10} {'Created':<20}")
    print("-" * 95)
    
    for user in users:
        user_id, name, emp_id, role, active, created = user
        status = "Active" if active else "Inactive"
        print(f"{user_id:<5} {name:<25} {emp_id:<15} {role:<10} {status:<10} {created:<20}")
    
    print(f"\nTotal: {len(users)} users")
    print("="*60 + "\n")


def deactivate_user_interactive(db, admin_user):
    """Deactivate a user."""
    print("\n" + "="*60)
    print("  DEACTIVATE USER")
    print("="*60)
    
    employee_id = input("\nEnter Employee ID to deactivate: ").strip()
    
    if not employee_id:
        print("[ERROR] Employee ID required")
        return
    
    # Search for user
    users = db.search_user(employee_id)
    
    if not users:
        print(f"[ERROR] User {employee_id} not found")
        return
    
    user = users[0]
    print(f"\nFound: {user[1]} (ID: {user[2]}) - Role: {user[3]}")
    
    if user[4] == 0:
        print("[WARNING] User is already inactive")
        return
    
    confirm = input("\nConfirm deactivation? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        if db.deactivate_user(employee_id, admin_user):
            print(f"[SUCCESS] ✓ User {employee_id} deactivated")
            
            # Optional: Delete face photo
            delete_photo = input("Delete face photo from known_faces? (yes/no): ").strip().lower()
            if delete_photo == 'yes':
                # Find and delete photo
                for filename in os.listdir(KNOWN_FACES_DIR):
                    if employee_id in filename:
                        filepath = os.path.join(KNOWN_FACES_DIR, filename)
                        os.remove(filepath)
                        print(f"[DELETE] ✓ Photo deleted: {filename}")
        else:
            print("[ERROR] Failed to deactivate user")
    else:
        print("[INFO] Deactivation cancelled")


def view_access_logs():
    """View recent access logs from CSV."""
    print("\n" + "="*60)
    print("  RECENT ACCESS LOGS")
    print("="*60)
    
    if not os.path.exists(LOG_FILE):
        print("\n[INFO] No access logs found\n")
        return
    
    try:
        with open(LOG_FILE, 'r') as f:
            reader = csv.reader(f)
            logs = list(reader)
        
        if len(logs) <= 1:  # Only header or empty
            print("\n[INFO] No access events logged yet\n")
            return
        
        # Display last 20 entries
        print(f"\n{'Timestamp':<20} {'Name':<25} {'Status':<30} {'Confidence':<12}")
        print("-" * 95)
        
        for log in logs[-20:]:
            if len(log) >= 4:
                timestamp, name, status, confidence = log[:4]
                print(f"{timestamp:<20} {name:<25} {status:<30} {confidence:<12}")
        
        print(f"\nTotal entries: {len(logs) - 1}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"[ERROR] Failed to read logs: {e}")


def view_admin_logs(db):
    """View admin action logs."""
    print("\n" + "="*60)
    print("  ADMIN ACTION LOGS")
    print("="*60)
    
    logs = db.get_admin_logs(limit=30)
    
    if not logs:
        print("\n[INFO] No admin actions logged\n")
        return
    
    print(f"\n{'Timestamp':<20} {'Admin':<15} {'Action':<20} {'Target':<15} {'Details':<30}")
    print("-" * 110)
    
    for log in logs:
        timestamp, admin, action, target, details = log
        target = target or "N/A"
        details = details or ""
        print(f"{timestamp:<20} {admin:<15} {action:<20} {target:<15} {details[:30]:<30}")
    
    print(f"\nTotal actions: {len(logs)}")
    print("="*60 + "\n")


def search_users_interactive(db):
    """Search for users."""
    print("\n" + "="*60)
    print("  SEARCH USERS")
    print("="*60)
    
    search_term = input("\nEnter name or employee ID: ").strip()
    
    if not search_term:
        print("[ERROR] Search term required")
        return
    
    results = db.search_user(search_term)
    
    if not results:
        print(f"[INFO] No users found matching '{search_term}'")
        return
    
    print(f"\n{'ID':<5} {'Name':<25} {'Emp ID':<15} {'Role':<10} {'Status':<10}")
    print("-" * 70)
    
    for user in results:
        user_id, name, emp_id, role, active = user
        status = "Active" if active else "Inactive"
        print(f"{user_id:<5} {name:<25} {emp_id:<15} {role:<10} {status:<10}")
    
    print(f"\nFound {len(results)} result(s)")
    print("="*60 + "\n")


# ==========================================
# MAIN MENU
# ==========================================
def main_menu():
    """Display main admin menu."""
    print("\n" + "="*60)
    print("  ADMIN PANEL - MAIN MENU")
    print("="*60)
    print("\n1. Add New User")
    print("2. View All Users")
    print("3. Search User")
    print("4. Deactivate User")
    print("5. Reactivate User")
    print("6. View Access Logs")
    print("7. View Admin Action Logs")
    print("8. Export Logs to CSV")
    print("9. Database Statistics")
    print("0. Exit")
    print("\n" + "="*60)


def export_logs(db):
    """Export all logs to CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"admin_logs_export_{timestamp}.csv"
    
    logs = db.get_admin_logs(limit=10000)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Admin', 'Action', 'Target User', 'Details'])
        writer.writerows(logs)
    
    print(f"\n[EXPORT] ✓ Logs exported to {filename}")


def show_statistics(db):
    """Display database statistics."""
    print("\n" + "="*60)
    print("  DATABASE STATISTICS")
    print("="*60)
    
    all_users = db.get_all_users(include_inactive=True)
    active_users = db.get_all_users(include_inactive=False)
    
    admin_count = len([u for u in all_users if u[3] == 'admin'])
    faculty_count = len([u for u in all_users if u[3] == 'faculty'])
    staff_count = len([u for u in all_users if u[3] == 'staff'])
    
    print(f"\nTotal Users: {len(all_users)}")
    print(f"  - Active: {len(active_users)}")
    print(f"  - Inactive: {len(all_users) - len(active_users)}")
    print(f"\nBy Role:")
    print(f"  - Admins: {admin_count}")
    print(f"  - Faculty: {faculty_count}")
    print(f"  - Staff: {staff_count}")
    
    # Count access logs
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            log_count = len(f.readlines()) - 1  # Exclude header
        print(f"\nTotal Access Events: {log_count}")
    
    print("\n" + "="*60 + "\n")


# ==========================================
# MAIN PROGRAM
# ==========================================
def main():
    """Main admin panel application."""
    print("\n" + "="*60)
    print("  CVSU SMART ACCESS SYSTEM - ADMIN PANEL")
    print("  Version 1.0")
    print("="*60)
    
    # Authenticate
    if not authenticate_admin():
        print("\n[EXIT] Access denied\n")
        return
    
    # Initialize database
    db = AdminDatabase()
    admin_user = ADMIN_USERNAME
    
    # Main loop
    while True:
        main_menu()
        
        choice = input("Select option: ").strip()
        
        if choice == '1':
            add_new_user(db, admin_user)
        
        elif choice == '2':
            view_all_users(db)
        
        elif choice == '3':
            search_users_interactive(db)
        
        elif choice == '4':
            deactivate_user_interactive(db, admin_user)
        
        elif choice == '5':
            emp_id = input("Enter Employee ID to reactivate: ").strip()
            if emp_id and db.reactivate_user(emp_id, admin_user):
                print(f"[SUCCESS] ✓ User {emp_id} reactivated")
            else:
                print("[ERROR] Failed to reactivate user")
        
        elif choice == '6':
            view_access_logs()
        
        elif choice == '7':
            view_admin_logs(db)
        
        elif choice == '8':
            export_logs(db)
        
        elif choice == '9':
            show_statistics(db)
        
        elif choice == '0':
            print("\n[EXIT] Thank you for using Admin Panel\n")
            break
        
        else:
            print("\n[ERROR] Invalid option\n")
        
        input("Press ENTER to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[EXIT] Admin panel closed by user\n")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()