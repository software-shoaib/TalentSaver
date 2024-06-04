from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import pickle

app = Flask(__name__)

# Load the data
data = pd.read_csv('D:\\IQRA UNIVERSITY\\FYP-2\\working\\project\\TalentSaver\\Table_1.csv')
data.columns = data.columns.str.strip()

# Handle missing values (if any)
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
label_encoders = {}
for column in ['Location', 'Emp. Group', 'Function', 'Gender', 'Tenure Grp.', 'Marital Status', 'Hiring Source', 'Promoted/Non Promoted', 'Job Role Match', 'Stay/Left']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

# Separate features and target variable
X = data.drop(columns=['Stay/Left', 'name', 'phone number', 'table id'])
y = data['Stay/Left']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train a RandomForestClassifier and save it
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find_replacement', methods=['POST'])
def find_replacement():
    employee_id = int(request.form['employee_id'])
    leaving_employee = data.loc[data['table id'] == employee_id].drop(columns=['Stay/Left', 'name', 'phone number', 'table id'])

    # Load model and scaler
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Scale the leaving employee's data
    leaving_employee_scaled = scaler.transform(leaving_employee)

    # Use NearestNeighbors to find the most similar employees
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(X)
    distances, indices = knn.kneighbors(leaving_employee_scaled)

    # Get the top candidates
    top_candidates = data.iloc[indices[0]]
    top_candidates = top_candidates[top_candidates['Stay/Left'] == label_encoders['Stay/Left'].transform(['Stay'])[0]]

    return jsonify(top_candidates.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
