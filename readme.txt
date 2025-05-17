Install NSE tools
NSEPython Documentation:

https://pypi.org/project/nsepython/

*python -m pip install --upgrade pip==21.0.1


If unable to set virtual environment:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

Creating Virtual Env to run locally
1. python -m venv myenv
2. myenv\Scripts\activate.bat  : cmd
3. source myenv/Scripts/activate  :git bash
3. .\myenv\Scripts\Activate.ps1 :PowerShell: Use 

To stop env: deactivate

to remove env: Remove-Item -Recurse -Force myenv


-- pyhton 3.10x above
pip install --upgrade Flask==3.1.0
pip install --upgrade Werkzeug==3.0.0
pip install --upgrade pandas==1.5.0  # Or a compatible version

myenv/
├── Include/
├── Lib/
├── Scripts/
    ├── Activate.ps1
    ├── activate.bat


mysource code/
├── app.py/
├── other_file.py/
├── templates/
    ├── index.html



Reference:\
https://pypi.org/project/nsepython/

Installation:

pip install nsepython

pip install nsepy

pip install plotly

Run:
http://localhost:5000/stock-price?symbol=RELIANCE


Docker:
pip install -r requirements.txt

docker build -t flask-stock-app .(FIleName must be Dockerfile).
docker stop flask-container
docker rm flask-container
docker run -d -p 5000:5000 --name flask-container flask-app

pip install flask-cors


Git Repo: https://github.com/Saumya528/StockPredictModels.git

ipykernel for nsepy

.venv\Lib\site-packages\nsepy\history.py


Replace del(kwargs['frame'])
 with if 'frame' in kwargs:
    try:
        del kwargs['frame']
    except TypeError:
        pass  # Handle FrameLocalsProxy issue
