"""Web interface for predicting a new patient's risk level with every model.

    python3 app.py                 # then open http://127.0.0.1:5050
    python3 app.py --port 8000 --no-browser

Run `python3 main.py` once first so the results exist.
"""
from src.ui.web_app import main

if __name__ == '__main__':
    main()