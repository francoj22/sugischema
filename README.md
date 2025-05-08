# Streamlit project with Flask

CRUD application running with flask
UI design with streamlit

## Project Structure
```
sugischema/
├── app/                   # Main Directory 
  ├── schemas/             # Directory to validate the schemas
  ├── static/              # Directory to handle html pages
  ├── app_routes.py        # API routes 
├── requirements.txt     # Dependencies
```

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API credentials:


3. Run the main script:
```bash
EXPORT FLASK_APP=run
flask run --port=8080
streamlit run app.py
```

## Features

-Create, Update, Delete, And add Todo

## Note

CRUD project for educational purpose
