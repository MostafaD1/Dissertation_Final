@echo off
echo Installing packages...
"C:\Users\Mostafa's PC\AppData\Local\Programs\Python\Python313\python.exe" -m pip install streamlit pandas joblib scikit-learn xgboost catboost

echo Launching app...
"C:\Users\Mostafa's PC\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run App.py
pause