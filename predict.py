"""Classify NEW patients with every model.

    python predict.py --age 35 --systolic 140 --diastolic 90 --bs 13 --temp 98 --hr 80
    python predict.py --csv new_patients.csv
    python predict.py                      # asks for each value

See src/prediction/predict_patient.py for how the predictions are made.
"""
from src.prediction.predict_patient import main

if __name__ == '__main__':
    main()
