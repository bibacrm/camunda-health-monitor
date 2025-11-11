"""
WSGI Entry Point for Production Deployment
Use with Gunicorn: gunicorn -c gunicorn.conf.py wsgi:app
"""
from app_new import create_app

# Create application instance
app = create_app('production')

if __name__ == "__main__":
    app.run()

