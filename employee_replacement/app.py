from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from functools import wraps

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to load and process the CSV file
def load_and_process_csv(filepath):
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip()
    data.fillna(method='ffill', inplace=True)

    label_encoders = {}
    for column in ['Location', 'Emp. Group', 'Function', 'Gender', 'Tenure Grp.', 'Marital Status', 'Hiring Source', 'Promoted/Non Promoted', 'Job Role Match', 'Stay/Left']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le

    X = data.drop(columns=['Stay/Left', 'name', 'phone number', 'table id'])
    y = data['Stay/Left']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return data, X, y, scaler, model, label_encoders

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Signup successful!', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login failed. Check your email and password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/results')
@login_required
def results():
    return render_template('results.html')

@app.route('/add_employee')
@login_required
def add_employee():
    return render_template('add_employee.html')

@app.route('/upload_csv', methods=['POST'])
@login_required
def upload_csv():
    if 'csvFile' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['csvFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Load and process the new CSV file
        global data, X, y, scaler, model, label_encoders
        data, X, y, scaler, model, label_encoders = load_and_process_csv(filepath)

        return jsonify({'filepath': filepath}), 200

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/find_replacement', methods=['POST'])
@login_required
def find_replacement():
    employee_id = int(request.form['employee_id'])
    leaving_employee = data.loc[data['table id'] == employee_id].drop(columns=['Stay/Left', 'name', 'phone number', 'table id'])

    # Scale the leaving employee's data
    leaving_employee_scaled = scaler.transform(leaving_employee)

    # Use NearestNeighbors to find the most similar employees
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(X)
    distances, indices = knn.kneighbors(leaving_employee_scaled)

    # Get the top candidates
    top_candidates = data.iloc[indices[0]]
    top_candidates = top_candidates[top_candidates['Stay/Left'] == label_encoders['Stay/Left'].transform(['Stay'])[0]]

    # Select necessary columns for output
    output_columns = ['table id', 'name', 'Age in YY.', 'Location', 'Emp. Group', 'Function', 'Gender', 'Tenure', 'Experience (YY.MM)', 'Marital Status', 'Hiring Source', 'Promoted/Non Promoted', 'Job Role Match']
    top_candidates = top_candidates[output_columns]

    return jsonify(top_candidates.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
