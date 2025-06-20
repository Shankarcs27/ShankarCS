from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for
import threading
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os


app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize session variables for tracking progress
@app.before_request
def before_request():
    if 'completed_poses' not in session:
        session['completed_poses'] = []
    if 'current_pose_correct' not in session:
        session['current_pose_correct'] = False

def calculate_progress():
    total_poses = 8
    completed = len(session.get('completed_poses', []))
    return int((completed / total_poses) * 100)

# Define the pose labels
pose_labels = {
    'adhomukhasvanasana': 0,
    'ashtanga namaskara': 1,
    'ashwasanchalanasana': 2,
    'bhujangasana': 3,
    'chaturanga dandasana': 4,
    'hastauttanasana': 5,
    'padangusthasana': 6,
    'pranamsana': 7
}

# Load the model
model = joblib.load('model/pose_to_angles_predictor.pkl')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

angle_columns = ['Left Elbow', 'Right Elbow', 'Left Shoulder', 'Right Shoulder', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee']

def predict_angles(pose_name):
    pose_label = pose_labels[pose_name]
    predicted_angles = model.predict([[pose_label]])[0]
    angles = dict(zip(angle_columns, predicted_angles))
    return angles

# Function to read reference angles from an Excel file
def read_reference_angles(pose_name):
    reference_angles = predict_angles(pose_name)
    reference_angles = {key.lower().replace(' ', '_'): value for key, value in reference_angles.items() if key != 'pose'}
    print(reference_angles)
    return reference_angles

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Shared resources
cap = cv2.VideoCapture(0)
cap_lock = threading.Lock()

# Function to display video preview continuously
def video_stream():
    global cap
    while True:
        with cap_lock:
            ret, frame = cap.read()
            if not ret:
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Function to capture a single image
def capture_image():
    global cap
    with cap_lock:
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return None
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return None
    return frame

def process_frame(frame, reference_angles):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        angles = {
            'left_elbow': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]),
            'right_elbow': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]),
            'left_shoulder': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]),
            'right_shoulder': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]),
            'left_hip': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]),
            'right_hip': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]),
            'left_knee': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]),
            'right_knee': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]),
        }
        flag = True
        wrong_joints = []
        for part, angle in angles.items():
            if not compare_angles(angle, part, reference_angles):
                flag = False
                wrong_joints.append(part)
        if flag:
            session['current_pose_correct'] = True
            return "Pose is correct"
        else:
            session['current_pose_correct'] = False
            return f"Incorrect pose, adjust the following joints: {', '.join(wrong_joints)}"
    else:
        session['current_pose_correct'] = False
        return "No pose detected"

def compare_angles(calculated_angle, body_part, reference_angles):
    reference_angle = reference_angles[body_part]
    if abs(calculated_angle - reference_angle) > 30:
        return False
    return True

# Define pose instructions
pose_instructions = {
    'pranamasana': [
        "Stand straight with your feet together",
        "Join your palms in front of your chest in a prayer position",
        "Keep your spine straight",
        "Relax your shoulders",
        "Breathe normally and maintain the pose"
    ],
    'hastauttanasana': [
        "From prayer pose, raise your arms above your head",
        "Keep your arms straight and palms facing each other",
        "Arch your back slightly",
        "Look up towards your hands",
        "Breathe deeply and maintain the pose"
    ],
    'padangusthasana': [
        "From raised arms pose, bend forward from your hips",
        "Keep your legs straight",
        "Try to touch your toes with your hands",
        "Keep your head down",
        "Breathe normally and maintain the pose"
    ],
    'ashwasanchalanasana': [
        "From forward bend, step your right foot back",
        "Lower your right knee to the ground",
        "Keep your left foot between your hands",
        "Look forward and keep your back straight",
        "Breathe deeply and maintain the pose"
    ],
    'chaturanga dandasana': [
        "From lunge pose, step your left foot back",
        "Lower your body parallel to the ground",
        "Keep your elbows close to your body",
        "Maintain a straight line from head to heels",
        "Breathe normally and maintain the pose"
    ],
    'ashtanga namaskara': [
        "From plank pose, lower your knees to the ground",
        "Lower your chest and chin to the ground",
        "Keep your hips slightly elevated",
        "Keep your elbows close to your body",
        "Breathe normally and maintain the pose"
    ],
    'bhujangasana': [
        "From eight-limbed pose, slide forward",
        "Lift your chest off the ground",
        "Keep your elbows slightly bent",
        "Look forward and keep your shoulders relaxed",
        "Breathe deeply and maintain the pose"
    ],
    'adhomukhasvanasana': [
        "From cobra pose, lift your hips up",
        "Straighten your legs and arms",
        "Press your heels towards the ground",
        "Keep your head between your arms",
        "Breathe deeply and maintain the pose"
    ]
}

# Define reference images for each pose
reference_images = {
    'pranamasana': 'https://static.wixstatic.com/media/84dfe1_ceae302fca8c4d9fab8eca262d198512~mv2.png/v1/fill/w_438,h_338,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/84dfe1_ceae302fca8c4d9fab8eca262d198512~mv2.png',
    'hastauttanasana': 'https://static.wixstatic.com/media/84dfe1_0f9b1de509994303be27027f83a8a897~mv2.png/v1/fill/w_438,h_338,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/84dfe1_0f9b1de509994303be27027f83a8a897~mv2.png',
    'padangusthasana': 'https://static.wixstatic.com/media/84dfe1_588704e6d0104e16881f3a7660428d44~mv2.png/v1/fill/w_438,h_338,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/84dfe1_588704e6d0104e16881f3a7660428d44~mv2.png',
    'ashwasanchalanasana': 'https://static.wixstatic.com/media/84dfe1_feca6c4c315d400cb0f410f53e3a32b8~mv2.png/v1/fill/w_438,h_338,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/84dfe1_feca6c4c315d400cb0f410f53e3a32b8~mv2.png',
    'chaturanga dandasana': 'https://static.wixstatic.com/media/84dfe1_247ccc287e8c4f68afc388b91d9c2b8b~mv2.png/v1/fill/w_438,h_338,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/84dfe1_247ccc287e8c4f68afc388b91d9c2b8b~mv2.png',
    'ashtanga namaskara': 'https://static.wixstatic.com/media/84dfe1_46b83807026a42ba9f2fa6ace4840248~mv2.png/v1/fill/w_438,h_338,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/84dfe1_46b83807026a42ba9f2fa6ace4840248~mv2.png',
    'bhujangasana': 'https://static.wixstatic.com/media/84dfe1_477ff069c538467bb6e52b32e1751e04~mv2.png/v1/fill/w_438,h_338,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/84dfe1_477ff069c538467bb6e52b32e1751e04~mv2.png',
    'adhomukhasvanasana': 'https://static.wixstatic.com/media/84dfe1_159dbaed63ad43a584a3b9c5fa8bfb5c~mv2.png/v1/fill/w_438,h_338,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/84dfe1_159dbaed63ad43a584a3b9c5fa8bfb5c~mv2.png'
}

# Route for the main page
@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/surya-namaskar')
def surya_namaskar():
    return render_template('surya_namaskar.html')

# Route to serve video stream
@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Routes for each button
@app.route('/pose1')
def pose1():
    return render_template('page1.html', 
                         progress=calculate_progress(), 
                         next_pose='pose2', 
                         pose_name='pranamsana',
                         reference_image=reference_images['pranamasana'],
                         instructions=pose_instructions['pranamasana'])

@app.route('/pose2')
def pose2():
    return render_template('page2.html', 
                         progress=calculate_progress(), 
                         next_pose='pose3', 
                         pose_name='hastauttanasana',
                         reference_image=reference_images['hastauttanasana'],
                         instructions=pose_instructions['hastauttanasana'])

@app.route('/pose3')
def pose3():
    return render_template('page3.html', 
                         progress=calculate_progress(), 
                         next_pose='pose4', 
                         pose_name='padangusthasana',
                         reference_image=reference_images['padangusthasana'],
                         instructions=pose_instructions['padangusthasana'])

@app.route('/pose4')
def pose4():
    return render_template('page4.html', 
                         progress=calculate_progress(), 
                         next_pose='pose5', 
                         pose_name='ashwasanchalanasana',
                         reference_image=reference_images['ashwasanchalanasana'],
                         instructions=pose_instructions['ashwasanchalanasana'])

@app.route('/pose5')
def pose5():
    return render_template('page5.html', 
                         progress=calculate_progress(), 
                         next_pose='pose6', 
                         pose_name='chaturanga dandasana',
                         reference_image=reference_images['chaturanga dandasana'],
                         instructions=pose_instructions['chaturanga dandasana'])

@app.route('/pose6')
def pose6():
    return render_template('page6.html', 
                         progress=calculate_progress(), 
                         next_pose='pose7', 
                         pose_name='ashtanga namaskara',
                         reference_image=reference_images['ashtanga namaskara'],
                         instructions=pose_instructions['ashtanga namaskara'])

@app.route('/pose7')
def pose7():
    return render_template('page7.html', 
                         progress=calculate_progress(), 
                         next_pose='pose8', 
                         pose_name='bhujangasana',
                         reference_image=reference_images['bhujangasana'],
                         instructions=pose_instructions['bhujangasana'])

@app.route('/pose8')
def pose8():
    return render_template('page8.html', 
                         progress=calculate_progress(), 
                         pose_name='adhomukhasvanasana',
                         reference_image=reference_images['adhomukhasvanasana'],
                         instructions=pose_instructions['adhomukhasvanasana'])

@app.route('/analyze_pose', methods=['POST'])
def analyze_pose():
    pose_name = request.json.get('pose_name')
    reference_angles = read_reference_angles(pose_name)
    frame = capture_image()
    if frame is not None:
        result = process_frame(frame, reference_angles)
        return jsonify({'result': result})
    else:
        return jsonify({'result': 'Error capturing image'}), 500

if __name__ == '__main__':
    app.run(debug=True)
