from flask import Flask
from flask_cors import CORS


app = Flask(__name__)
CORS(app, support_credentials=True)
@app.context_processor
def inject_social_links():
    """Add social links to all templates"""
    social_links = {
        'linkedin': 'https://www.linkedin.com/in/franco-gutierrez-4a073483',
        'github': 'https://github.com/francoj22',
        'website': 'https://francoj22.github.io/'
    }
    return dict(social_links=social_links)
from app import app_routes
