import logging
import sys

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

print("Importing create_app...")
from app import create_app

print("Running create_app()...")
try:
    app = create_app()
    print("App created successfully!")
except Exception as e:
    print(f"Error: {e}")
