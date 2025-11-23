# app.py - Complete Corrected Version with Fixed Questionnaire Handling
import os
import csv
import json
import time
import webbrowser
import threading
from datetime import datetime, date, timedelta
from collections import Counter
from flask import Flask, Response, render_template, jsonify, request, send_file, abort
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import atexit
import cv2
import numpy as np

# Configuration
CAPTURE_DIR = "captures"
SHARED_DIR = "shared"
LATEST_IMG = os.path.join(SHARED_DIR, "latest.jpg")
REPORTS_DIR = "reports"
CSV_PATH = os.path.join(CAPTURE_DIR, "detections.csv")
QUESTIONNAIRE_CSV = os.path.join(CAPTURE_DIR, "questionnaire.csv")

# Create directories
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(SHARED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Initialize CSV files
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "image", "posture_count", "posture_labels", "emotion_count", "emotion_labels"])

if not os.path.exists(QUESTIONNAIRE_CSV):
    with open(QUESTIONNAIRE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "stress_level", "sleep_hours", "anxious", "took_breaks", "motivation"])

# Flask app initialization
app = Flask(__name__, template_folder="templates", static_folder="static")
latest_questionnaire = {}  # Store latest questionnaire in memory

class WellnessSuggestionEngine:
    """Engine for calculating wellness scores and generating EXACTLY 3 suggestions"""
    
    def __init__(self):
        self.posture_thresholds = {'good': 0.3, 'warning': 0.5, 'bad': 0.7}
        self.last_analysis = {}

    def analyze_recent_data(self, df, seconds=10):
        """Analyze detections from the last N seconds"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        dfc = df.copy()
        dfc['datetime'] = pd.to_datetime(
            dfc['timestamp'], 
            format='%Y%m%d_%H%M%S_%f', 
            errors='coerce'
        )
        dfc['datetime'] = dfc['datetime'].fillna(
            pd.to_datetime(dfc['timestamp'], errors='coerce')
        )
        
        cutoff = datetime.now() - timedelta(seconds=seconds)
        return dfc[dfc['datetime'] >= cutoff]

    def calculate_wellness_score(self, recent, questionnaire=None):
        """
        Calculate wellness score (40-100 scale)
        
        Scoring breakdown:
        - Camera detections (60 points max):
          - Posture: 30 points
          - Emotion: 30 points
        - Daily check-in (40 points max):
          - Stress: 10 points
          - Sleep: 10 points
          - Anxiety: 10 points
          - Breaks: 5 points
          - Motivation: 5 points
        """
        score = 100
        n = len(recent)
        
        # Tracking for detailed feedback
        self.last_analysis = {
            'posture_penalty': 0,
            'emotion_penalty': 0,
            'checkin_penalty': 0,
            'has_detections': n > 0,
            'detection_count': n
        }
        
        # === CAMERA-BASED SCORING (60 points) ===
        if n == 0:
            score -= 20
            self.last_analysis['no_data_penalty'] = 20
        else:
            bad_posture = 0
            good_posture = 0
            negative_emotions = 0
            positive_emotions = 0
            neutral_emotions = 0
            
            for _, r in recent.iterrows():
                # Posture analysis
                posture_labels = str(r.get('posture_labels', ''))
                if 'incorrect' in posture_labels.lower() or 'bad' in posture_labels.lower():
                    bad_posture += 1
                elif 'correct' in posture_labels.lower() or 'good' in posture_labels.lower():
                    good_posture += 1
                
                # Emotion analysis
                emotion_labels = str(r.get('emotion_labels', ''))
                if any(em in emotion_labels for em in ['Sad', 'Angry', 'Fearful']):
                    negative_emotions += 1
                elif any(em in emotion_labels for em in ['Happy', 'Surprised']):
                    positive_emotions += 1
                elif 'Neutral' in emotion_labels:
                    neutral_emotions += 1
            
            # Posture penalty (up to 30 points)
            if bad_posture > 0:
                posture_penalty = min(30, (bad_posture / n) * 30)
                score -= posture_penalty
                self.last_analysis['posture_penalty'] = round(posture_penalty)
                self.last_analysis['bad_posture_count'] = bad_posture
            
            # Emotion penalty (up to 30 points)
            if negative_emotions > 0:
                emotion_penalty = min(30, (negative_emotions / n) * 30)
                score -= emotion_penalty
                self.last_analysis['emotion_penalty'] = round(emotion_penalty)
                self.last_analysis['negative_emotion_count'] = negative_emotions
            
            # Store positive metrics
            self.last_analysis['positive_emotions'] = positive_emotions
            self.last_analysis['neutral_emotions'] = neutral_emotions
            self.last_analysis['good_posture'] = good_posture
        
        # === CHECK-IN BASED SCORING (40 points) - FIXED ===
        if questionnaire and isinstance(questionnaire, dict):
            print(f"[score] 📋 Processing questionnaire: {questionnaire}")
            
            try:
                # CRITICAL FIX: Convert string values to proper types
                stress_level = int(questionnaire.get('stress_level', 3))
                sleep_hours = float(questionnaire.get('sleep_hours', 7))
                is_anxious = str(questionnaire.get('anxious', 'no')).lower() == 'yes'
                took_breaks = str(questionnaire.get('took_breaks', 'yes')).lower() == 'yes'
                motivation = int(questionnaire.get('motivation', 3))
                
                print(f"[score] 📊 Parsed values: stress={stress_level}, sleep={sleep_hours}, anxious={is_anxious}, breaks={took_breaks}, motivation={motivation}")
                
                checkin_penalty = 0
                
                # Stress impact (0-10 points)
                if stress_level >= 4:
                    checkin_penalty += 10
                    self.last_analysis['high_stress'] = True
                    print(f"[score] ⚠️ High stress penalty: +10 (level {stress_level})")
                elif stress_level == 3:
                    checkin_penalty += 5
                    print(f"[score] ⚠️ Medium stress penalty: +5")
                
                # Sleep impact (0-10 points)
                if sleep_hours < 5:
                    checkin_penalty += 10
                    self.last_analysis['critical_sleep'] = True
                    print(f"[score] ⚠️ Critical sleep penalty: +10 ({sleep_hours}h)")
                elif sleep_hours < 6:
                    checkin_penalty += 7
                    self.last_analysis['low_sleep'] = True
                    print(f"[score] ⚠️ Low sleep penalty: +7 ({sleep_hours}h)")
                elif sleep_hours < 7:
                    checkin_penalty += 3
                    print(f"[score] ⚠️ Moderate sleep penalty: +3 ({sleep_hours}h)")
                
                # Anxiety impact (0-10 points)
                if is_anxious:
                    checkin_penalty += 10
                    self.last_analysis['is_anxious'] = True
                    print(f"[score] ⚠️ Anxiety penalty: +10")
                
                # Breaks impact (0-5 points)
                if not took_breaks:
                    checkin_penalty += 5
                    self.last_analysis['no_breaks'] = True
                    print(f"[score] ⚠️ No breaks penalty: +5")
                
                # Motivation impact (0-5 points)
                if motivation <= 2:
                    checkin_penalty += 5
                    self.last_analysis['low_motivation'] = True
                    print(f"[score] ⚠️ Low motivation penalty: +5 (level {motivation})")
                elif motivation == 3:
                    checkin_penalty += 2
                    print(f"[score] ⚠️ Medium motivation penalty: +2")
                
                score -= checkin_penalty
                self.last_analysis['checkin_penalty'] = checkin_penalty
                
                # Store check-in values
                self.last_analysis['stress_level'] = stress_level
                self.last_analysis['sleep_hours'] = sleep_hours
                self.last_analysis['motivation'] = motivation
                
                print(f"[score] 📊 Total check-in penalty: {checkin_penalty}")
                    
            except Exception as e:
                print(f"[score] ❌ Error processing questionnaire: {e}")
                import traceback
                traceback.print_exc()
        else:
            score -= 10
            self.last_analysis['checkin_penalty'] = 10
            self.last_analysis['no_checkin'] = True
            print(f"[score] ⚠️ No check-in submitted: -10 points")
        
        final_score = max(40, round(score))
        print(f"[score] 🎯 FINAL WELLNESS SCORE: {final_score}/100")
        return final_score

    def get_suggestions(self, recent, questionnaire):
        """Generate EXACTLY 3 personalized suggestions"""
        all_issues = []
        n = len(recent)
        
        # === ANALYZE CAMERA DATA ===
        if n > 0:
            bad_posture_count = 0
            emotion_counter = Counter()
            
            for _, r in recent.iterrows():
                posture_labels = str(r.get('posture_labels', ''))
                if 'incorrect' in posture_labels.lower() or 'bad' in posture_labels.lower():
                    bad_posture_count += 1
                
                emotion_labels = str(r.get('emotion_labels', ''))
                if emotion_labels:
                    for e in emotion_labels.split(';'):
                        emotion_name = e.split(':')[0].strip() if ':' in e else e.strip()
                        if emotion_name:
                            emotion_counter[emotion_name] += 1
            
            bad_posture_percent = (bad_posture_count / n) * 100
            if bad_posture_percent > 50:
                all_issues.append({
                    'priority': 1,
                    'title': '🪑 Poor Posture Detected',
                    'message': f'You showed poor posture in {bad_posture_percent:.0f}% of recent detections. Adjust your sitting position: back straight, shoulders relaxed, screen at eye level.'
                })
            elif bad_posture_percent > 20:
                all_issues.append({
                    'priority': 3,
                    'title': '📐 Posture Improvement Needed',
                    'message': f'Some posture issues detected ({bad_posture_percent:.0f}%). Take a moment to reset your posture and check your ergonomic setup.'
                })
            
            if emotion_counter:
                dominant_emotion = emotion_counter.most_common(1)[0][0]
                emotion_percent = (emotion_counter[dominant_emotion] / sum(emotion_counter.values())) * 100
                
                if dominant_emotion in ['Sad', 'Angry', 'Fearful'] and emotion_percent > 40:
                    all_issues.append({
                        'priority': 2,
                        'title': f'😔 {dominant_emotion} Emotion Dominant',
                        'message': f'You appear {dominant_emotion.lower()} in {emotion_percent:.0f}% of detections. Take a 5-minute break, practice deep breathing, or do something that brings you joy.'
                    })
                elif dominant_emotion == 'Happy' and emotion_percent > 60:
                    all_issues.append({
                        'priority': 5,
                        'title': '😊 Great Emotional State!',
                        'message': f'You\'re happy {emotion_percent:.0f}% of the time! Keep maintaining this positive energy with regular breaks and good habits.'
                    })
        else:
            all_issues.append({
                'priority': 2,
                'title': '📹 No Detection Data',
                'message': 'Camera detection is not capturing data. Ensure your camera is working and you are visible in the frame.'
            })
        
        # === ANALYZE CHECK-IN DATA ===
        if questionnaire:
            try:
                stress_level = int(questionnaire.get('stress_level', 3))
                sleep_hours = float(questionnaire.get('sleep_hours', 7))
                is_anxious = str(questionnaire.get('anxious', 'no')).lower() == 'yes'
                took_breaks = str(questionnaire.get('took_breaks', 'yes')).lower() == 'yes'
                motivation = int(questionnaire.get('motivation', 3))
                
                if stress_level >= 4:
                    all_issues.append({
                        'priority': 1,
                        'title': '😰 High Stress Level Reported',
                        'message': f'You reported stress level {stress_level}/5. Try the 4-7-8 breathing technique: Inhale 4 seconds, hold 7 seconds, exhale 8 seconds. Repeat 4 times.'
                    })
                
                if sleep_hours < 6:
                    all_issues.append({
                        'priority': 1,
                        'title': '😴 Sleep Deficit Detected',
                        'message': f'You only got {sleep_hours} hours of sleep. Prioritize 7-9 hours tonight. Avoid screens 1 hour before bed and keep your room cool.'
                    })
                
                if is_anxious:
                    all_issues.append({
                        'priority': 2,
                        'title': '😟 Anxiety Management',
                        'message': 'You\'re feeling anxious. Try the 5-4-3-2-1 grounding technique: Name 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste.'
                    })
                
                if not took_breaks:
                    all_issues.append({
                        'priority': 3,
                        'title': '⏰ No Breaks Taken',
                        'message': 'You haven\'t taken breaks today. Use the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds. Take a 5-minute walk every hour.'
                    })
                
                if motivation <= 2:
                    all_issues.append({
                        'priority': 3,
                        'title': '🎯 Low Motivation Alert',
                        'message': f'Motivation level is {motivation}/5. Break tasks into smaller chunks, reward yourself for progress, and consider changing your environment for a fresh perspective.'
                    })
                
            except Exception as e:
                print(f"[suggestions] Error processing questionnaire: {e}")
        else:
            all_issues.append({
                'priority': 4,
                'title': '📋 Complete Daily Check-In',
                'message': 'Submit your daily wellbeing check-in to get personalized suggestions based on your stress, sleep, and motivation levels.'
            })
        
        # Sort by priority and return top 3
        all_issues.sort(key=lambda x: x['priority'])
        top_3 = all_issues[:3]
        
        # Add generic tips if needed
        while len(top_3) < 3:
            generic_tips = [
                {'priority': 5, 'title': '💧 Stay Hydrated', 'message': 'Drink a glass of water. Proper hydration improves focus, energy, and overall wellbeing.'},
                {'priority': 5, 'title': '🚶 Movement Break', 'message': 'Stand up and walk for 2 minutes. Physical movement boosts circulation and mental clarity.'},
                {'priority': 5, 'title': '👁️ Eye Rest', 'message': 'Look away from your screen and focus on a distant object for 30 seconds to reduce eye strain.'}
            ]
            top_3.extend(generic_tips)
            top_3 = top_3[:3]
        
        # Clean up before returning
        for suggestion in top_3:
            suggestion.pop('priority', None)
            suggestion.pop('category', None)
        
        return top_3

engine = WellnessSuggestionEngine()

def gen_mjpeg():
    """Generate MJPEG stream with proper error handling"""
    last_frame = None
    consecutive_failures = 0
    max_failures = 10
    
    while True:
        try:
            if os.path.exists(LATEST_IMG):
                with open(LATEST_IMG, 'rb') as f:
                    img_data = f.read()
                
                if len(img_data) > 100:
                    last_frame = img_data
                    consecutive_failures = 0
                    frame = (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            b'Content-Length: ' + str(len(img_data)).encode() + b'\r\n'
                            b'\r\n' + img_data + b'\r\n')
                    yield frame
                else:
                    consecutive_failures += 1
            else:
                consecutive_failures += 1
                
        except Exception as e:
            print(f"[video_feed] Error reading frame: {e}")
            consecutive_failures += 1
        
        if consecutive_failures >= max_failures or last_frame is None:
            black = create_fallback_jpg()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(black)).encode() + b'\r\n'
                   b'\r\n' + black + b'\r\n')
            consecutive_failures = 0
        
        time.sleep(0.05)

def create_fallback_jpg():
    """Create a fallback black frame with text"""
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(black, "Waiting for camera...", (150, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(black, "Make sure detector.py is running", (120, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    _, buf = cv2.imencode('.jpg', black)
    return buf.tobytes()

# ==================== FLASK ROUTES ====================

@app.route('/')
def dashboard():
    """Render main dashboard page"""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """Stream MJPEG video feed"""
    return Response(gen_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/report')
def api_report():
    """API endpoint for wellness report"""
    global latest_questionnaire
    
    df = load_detections()
    q = latest_questionnaire.copy() if latest_questionnaire else None
    recent = engine.analyze_recent_data(df, seconds=10)
    
    wellness_score = engine.calculate_wellness_score(recent, q)
    suggestions = engine.get_suggestions(recent, q)
    
    if not q:
        return jsonify({
            "status": "need_questionnaire",
            "message": "Complete daily check-in for personalized analysis.",
            "overall_wellness_score": wellness_score,
            "total_detections": len(recent),
            "suggestions": suggestions
        })
    
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_period": "10 seconds",
        "total_detections": len(recent),
        "suggestions": suggestions,
        "overall_wellness_score": wellness_score
    })

@app.route('/api/questionnaire', methods=['POST'])
def api_questionnaire():
    """API endpoint to submit daily questionnaire - FIXED"""
    global latest_questionnaire
    
    data = request.json or {}
    
    # CRITICAL FIX: Store as proper data types
    row = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "stress_level": int(data.get("stress_level", 3)),  # Convert to int
        "sleep_hours": float(data.get("sleep_hours", 7)),  # Convert to float
        "anxious": "yes" if data.get("anxious") in [True, "yes", "Yes"] else "no",
        "took_breaks": "yes" if data.get("took_breaks") in [True, "yes", "Yes"] else "no",
        "motivation": int(data.get("motivation", 3))  # Convert to int
    }
    
    # Save to CSV
    try:
        with open(QUESTIONNAIRE_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                row['timestamp'], 
                row['stress_level'], 
                row['sleep_hours'], 
                row['anxious'], 
                row['took_breaks'], 
                row['motivation']
            ])
        print(f"[app] ✅ Questionnaire saved: {row}")
    except Exception as e:
        print(f"[app] ❌ Could not write questionnaire: {e}")
    
    # Update in-memory questionnaire
    latest_questionnaire.clear()
    latest_questionnaire.update(row)
    
    print(f"[app] 📊 Latest questionnaire updated: {latest_questionnaire}")
    
    return jsonify({"ok": True, "message": "Daily check-in submitted successfully!", "saved": row})

@app.route('/reports/<name>')
def serve_reports(name):
    """Serve report files for download"""
    safe_path = os.path.join(REPORTS_DIR, name)
    if not os.path.exists(safe_path):
        abort(404)
    return send_file(safe_path, as_attachment=True)

# ==================== HELPER FUNCTIONS ====================

def load_detections():
    """Load detection data from CSV"""
    try:
        df = pd.read_csv(CSV_PATH)
        return df
    except Exception as e:
        print(f"[app] Could not load detections: {e}")
        return pd.DataFrame(columns=["timestamp", "image", "posture_count", "posture_labels", "emotion_count", "emotion_labels"])

def generate_daily_report_for(date_obj=None):
    """Generate daily report with charts for a specific date"""
    if date_obj is None:
        date_obj = date.today()
    
    day_str = date_obj.strftime("%Y-%m-%d")
    
    try:
        df = load_detections()
    except:
        df = pd.DataFrame()
    
    def safe_parse(ts):
        try:
            return datetime.strptime(str(ts), "%Y%m%d_%H%M%S_%f").date()
        except:
            try:
                return pd.to_datetime(ts).date()
            except:
                return None
    
    if not df.empty:
        df['date'] = df['timestamp'].apply(safe_parse)
        day_df = df[df['date'] == date_obj]
    else:
        day_df = pd.DataFrame()
    
    report = {'date': day_str, 'total_captures': int(len(day_df))}
    
    posture_counts = Counter()
    emotion_counts = Counter()
    
    if not day_df.empty:
        for _, r in day_df.iterrows():
            pl = str(r.get('posture_labels', ''))
            for p in pl.split(';'):
                if p.strip():
                    label = p.split(':')[0].strip()
                    posture_counts[label] += 1
            
            el = str(r.get('emotion_labels', ''))
            for e in el.split(';'):
                if e.strip():
                    label = e.split(':')[0].strip()
                    emotion_counts[label] += 1
    
    report.update({'posture_counts': dict(posture_counts), 'emotion_counts': dict(emotion_counts)})
    
    json_path = os.path.join(REPORTS_DIR, f"{day_str}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    csv_path = os.path.join(REPORTS_DIR, f"{day_str}_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "total_captures"])
        w.writerow([report['date'], report['total_captures']])
        w.writerow([])
        w.writerow(["posture_label", "count"])
        for k, v in report['posture_counts'].items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["emotion_label", "count"])
        for k, v in report['emotion_counts'].items():
            w.writerow([k, v])
    
    return report

# ==================== SUBPROCESS MANAGEMENT ====================

detector_process = None

def start_detector():
    """Start detector.py as a subprocess"""
    global detector_process
    
    if detector_process is None:
        print("[app] Launching detector.py ...")
        try:
            if os.name == 'nt':
                detector_process = subprocess.Popen(
                    ["python", "detector.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                def read_output():
                    for line in iter(detector_process.stdout.readline, ''):
                        if line:
                            print(f"[detector] {line.strip()}")
                    detector_process.stdout.close()
                
                output_thread = threading.Thread(target=read_output, daemon=True)
                output_thread.start()
            else:
                detector_process = subprocess.Popen(
                    ["python", "detector.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )
        except Exception as e:
            print(f"[app] Failed to launch detector: {e}")

def stop_detector():
    """Stop detector.py subprocess"""
    global detector_process
    
    if detector_process is not None:
        print("[app] Stopping detector.py ...")
        try:
            detector_process.terminate()
            detector_process.wait(timeout=5)
        except:
            try:
                detector_process.kill()
            except:
                pass
        detector_process = None

atexit.register(stop_detector)

# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        print("[app] Initial startup - launching detector...")
        start_detector()
    
    def open_browser():
        time.sleep(3)
        try:
            webbrowser.open("http://127.0.0.1:5000")
        except:
            pass
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        print("[app] Starting Flask server on http://127.0.0.1:5000")
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=True, threaded=True)
    except KeyboardInterrupt:
        print("\n[app] Shutting down...")
    finally:
        stop_detector()
