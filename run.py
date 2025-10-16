from app import app
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    # Bind to 0.0.0.0 so Docker can access it
    app.run(debug=debug_mode, host='0.0.0.0', port=port)