@echo off
echo Setting up ML Project Environment...

REM Create virtual environment
python -m venv mlproject_env

REM Activate environment
call mlproject_env\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install numpy pandas seaborn matplotlib scikit-learn jupyter ipykernel

REM Install project in editable mode
pip install -e .

REM Add kernel to jupyter
python -m ipykernel install --user --name=mlproject --display-name="ML Project"

echo Environment setup complete!
echo To activate: mlproject_env\Scripts\activate.bat
pause