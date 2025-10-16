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
import torch
import torch.nn as nn
import torch.nn.functional as F

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Complete CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: bird, frog
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = CNN()
try:
    model.load_state_dict(torch.load('app/model/test_model.pth', map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Image preprocessing with proper normalization
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])


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



# API endpoint for predictions 
@app.route("/predict_archived/", methods=["POST"])
def predict():
    """Handle file upload and return prediction with proper confidence calibration"""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not (file and allowed_file(file.filename)):
            return jsonify({"error": "Invalid file type. Allowed: PNG, JPG, JPEG"}), 400

        # Process image
        image = Image.open(file.stream).convert("RGB")
        
        # Debug: Print image info
        print(f"Processing image: {file.filename}, Size: {image.size}")
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Debug: Print tensor stats
        print(f"Tensor shape: {input_tensor.shape}")
        print(f"Tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

        # Make prediction with proper confidence calibration
        with torch.no_grad():
            raw_output = model(input_tensor)
            print(f"Raw model output: {raw_output.cpu().numpy()}")
            
            # Apply temperature scaling to calibrate confidence
            temperature = 3.0  # Higher temperature = less confident
            calibrated_output = raw_output / temperature
            
            # Get probabilities
            probabilities = F.softmax(calibrated_output, dim=1)
            print(f"Calibrated probabilities: {probabilities.cpu().numpy()}")
            
            # Get prediction
            confidence, predicted = torch.max(probabilities, 1)
            class_names = ["bird", "frog"]
            prediction = class_names[predicted.item()]
            confidence_score = confidence.item()
            
            # Additional safeguard: Cap maximum confidence at 95%
            bird_prob = min(probabilities[0][0].item(), 0.95)
            frog_prob = min(probabilities[0][1].item(), 0.95)
            
            # Renormalize if we capped the probabilities
            total_prob = bird_prob + frog_prob
            if total_prob > 1.0:
                bird_prob = bird_prob / total_prob
                frog_prob = frog_prob / total_prob

        return jsonify({
            "prediction": prediction,
            "confidence": round(min(confidence_score, 0.95), 4),  # Cap at 95%
            "probabilities": {
                "bird": round(bird_prob, 4),
                "frog": round(frog_prob, 4)
            },
            "debug_info": {
                "raw_output": [round(raw_output[0][0].item(), 4), round(raw_output[0][1].item(), 4)],
                "temperature_used": temperature,
                "original_confidence": round(confidence_score, 4)
            }
        }), 200
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route("/predict_alt/", methods=["POST"])
def predict_alt():
    """Alternative prediction with different normalization"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        
        if file.filename == '' or not (file and allowed_file(file.filename)):
            return jsonify({"error": "Invalid file"}), 400

        # Process image with different normalization
        image = Image.open(file.stream).convert("RGB")
        
        # Try the original normalization that might match training
        alt_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        input_tensor = alt_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            raw_output = model(input_tensor)
            
            # Use higher temperature for calibration
            temperature = 5.0
            calibrated_output = raw_output / temperature
            probabilities = F.softmax(calibrated_output, dim=1)
            
            confidence, predicted = torch.max(probabilities, 1)
            class_names = ["bird", "frog"]

        return jsonify({
            "prediction": class_names[predicted.item()],
            "confidence": round(min(confidence.item(), 0.90), 4),
            "probabilities": {
                "bird": round(probabilities[0][0].item(), 4),
                "frog": round(probabilities[0][1].item(), 4)
            },
            "method": "alternative_normalization"
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route("/predict_clean/", methods=["POST"])
def predict_clean():
    """Clean prediction without debugging - """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        
        if file.filename == '' or not (file and allowed_file(file.filename)):
            return jsonify({"error": "Invalid or no file selected"}), 400

        # Process image
        image = Image.open(file.stream).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Prediction with temperature scaling
        with torch.no_grad():
            raw_output = model(input_tensor)
            
            # Temperature scaling to reduce overconfidence
            temperature = 1.5
            scaled_output = raw_output / temperature
            probabilities = F.softmax(scaled_output, dim=1)
            
            # Get prediction
            confidence, predicted = torch.max(probabilities, 1)
            class_names = ["bird", "frog"]
            
            # Add uncertainty threshold
            max_prob = confidence.item()
            if max_prob < 0.6:  # If less than 60% confident
                return jsonify({
                    "prediction": "uncertain",
                    "confidence": round(max_prob, 4),
                    "message": "Model is uncertain about this image",
                    "probabilities": {
                        "bird": round(probabilities[0][0].item(), 4),
                        "frog": round(probabilities[0][1].item(), 4)
                    }
                }), 200

        return jsonify({
            "prediction": class_names[predicted.item()],
            "confidence": round(max_prob, 4),
            "probabilities": {
                "bird": round(probabilities[0][0].item(), 4),
                "frog": round(probabilities[0][1].item(), 4)
            }
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route("/predict_fixed/", methods=["POST"])
def predict_fixed():
    """Prediction with logit clipping to handle extreme outputs"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        
        if file.filename == '' or not (file and allowed_file(file.filename)):
            return jsonify({"error": "Invalid file"}), 400

        # Process image
        image = Image.open(file.stream).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            raw_output = model(input_tensor)
            print(f"Raw output: {raw_output.cpu().numpy()}")
            
            # Clip extreme logits to reasonable range
            clipped_output = torch.clamp(raw_output, min=-10, max=10)
            print(f"Clipped output: {clipped_output.cpu().numpy()}")
            
            # Apply temperature scaling
            temperature = 2.0
            calibrated_output = clipped_output / temperature
            
            # Get probabilities
            probabilities = F.softmax(calibrated_output, dim=1)
            print(f"Final probabilities: {probabilities.cpu().numpy()}")
            
            confidence, predicted = torch.max(probabilities, 1)
            class_names = ["bird", "frog"]
            
            # Ensure minimum uncertainty (max 90% confidence)
            bird_prob = probabilities[0][0].item()
            frog_prob = probabilities[0][1].item()
            
            # Add minimum uncertainty of 10%
            if bird_prob > 0.9:
                bird_prob = 0.9
                frog_prob = 0.1
            elif frog_prob > 0.9:
                bird_prob = 0.1
                frog_prob = 0.9

        return jsonify({
            "prediction": class_names[predicted.item()],
            "confidence": round(max(bird_prob, frog_prob), 4),
            "probabilities": {
                "bird": round(bird_prob, 4),
                "frog": round(frog_prob, 4)
            },
            "debug_info": {
                "raw_output": raw_output.cpu().numpy().tolist(),
                "clipped_output": clipped_output.cpu().numpy().tolist(),
                "method": "logit_clipping"
            }
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=3000)