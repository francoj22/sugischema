"""
Flask Todo Demo App with Form and Table
---------------------------------
A complete Flask application with:
- Form for adding new todos
- Table to display existing todos
- Ability to mark todos as complete
- Ability to delete todos
"""

# File: app.py
from flask import render_template, request, jsonify, redirect, url_for
from app import app
import os
from PIL import Image
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import os


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Define the model (same architecture as before)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN()
model.load_state_dict(torch.load('app/model/test_model.pth', map_location=device))
model.eval()

# Using a list of dictionaries for todos with id, task, and completed status
todolist = [
    {"id": 1, "task": "do shopping", "completed": False},
    {"id": 2, "task": "cut the grass", "completed": False},
    {"id": 3, "task": "get the attic ready", "completed": False}
]

# Helper function to get the next available ID
def get_next_id():
    if not todolist:
        return 1
    return max(item["id"] for item in todolist) + 1


@app.route('/')
def home():
    return render_template('index.html', todolist=todolist)

@app.route('/dashboard')
def dashboard():
    # This would have its own template
    return render_template('home.html')  # Using home.html as placeholder

@app.route('/components')
def components():
    return render_template('components.html')

@app.route('/settings')
def settings():
    # This would have its own template
    return render_template('home.html')  # Using home.html as placeholder

@app.route('/profile')
def profile():
    # This would have its own template
    return render_template('home.html')  # Using home.html as placeholder


@app.route('/')
def index():
    """Render the main page with todo list and form"""
    return render_template('index.html', todolist=todolist)

@app.route('/add_todo', methods=['POST'])
def add_todo():
    """Add a new todo item using form data"""
    task = request.form.get('todo_task')
    
    if task and task.strip():  # Check that task is not empty or just whitespace
        new_todo = {
            "id": get_next_id(),
            "task": task,
            "completed": False
        }
        todolist.append(new_todo)
    
    # Redirect back to the index page to see the updated list
    return redirect(url_for('index'))

@app.route('/toggle_todo/<int:todo_id>', methods=['POST'])
def toggle_todo(todo_id):
    """Toggle the completed status of a todo item"""
    todo = next((item for item in todolist if item["id"] == todo_id), None)
    
    if todo:
        todo['completed'] = not todo['completed']
    
    return redirect(url_for('index'))

@app.route('/delete_todo/<int:todo_id>', methods=['POST'])
def delete_todo(todo_id):
    """Delete a todo item"""
    global todolist
    todolist = [item for item in todolist if item["id"] != todo_id]
    
    return redirect(url_for('index'))



@app.route('/api/todos', methods=['GET'])
def get_todos():
    """API endpoint to get all todos as JSON"""
    return jsonify({"todos": todolist})



# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# API endpoint for predictions
@app.route("/predict/", methods=["POST"])
def predict():
    """Handle file upload and return prediction"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Load and preprocess the image
        image = Image.open(file).convert("RGB")  # Use 'file' directly
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            class_names = ["bird", "frog"]
            prediction = class_names[predicted.item()]

        return jsonify({"prediction": prediction}), 200
    
    return jsonify({"error": "Invalid file type"}), 400


if __name__ == '__main__':
    app.run(debug=True, port=3000)