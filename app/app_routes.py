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

if __name__ == '__main__':
    app.run(debug=True, port=3000)