# Streamlit time series forecast application

## Task

Your task is to pack [Streamlit](https://streamlit.io) ML application into working executable .exe file.

## Requirements

You are expected to address the following points in your solution:

1. Get familiar with [ETNA](https://github.com/tinkoff-ai/etna) time-series library and concept of time-series back-testing.
2. Use [CatBoostPerSegment](https://etna-docs.netlify.app/api/etna.models.catboost.catboostpersegmentmodel#etna.models.catboost.CatBoostPerSegmentModel) and [pipeline](https://etna-docs.netlify.app/api/etna.pipeline.pipeline.pipeline#etna.pipeline.pipeline.Pipeline) to build and validate your forecasting model. You basically may use code from [Get Started](https://github.com/tinkoff-ai/etna#get-started) of the library. 
3. Build very simple Streamlit offline web-app where user can train and validate model. Make user able to choose ETNA transforms of his choice (just a few of them) via Streamlit API.
4. Visualize the results of model backtest and forecasts in the app.
5. Pack it into excecutable`.exe` installer using pyinstaller. Install the app on your machine and test it. It must work.
6. The executable offline application is required to utilize the CatBoostPerSegment and the pipeline from the ETNA library and to visualize forecasts. This .msi installer should open a window in a web browser where the CatBoost model within the pipeline is executed. Connection to the internet is not allowed.
7. Provide a concise ReadMe.txt with instructions how to build the standalone app. The instruction should have the steps required to go through in order to build the `.exe` that can be executed offline by final user. The solution must be reproducible.

![Image alt](https://github.com/AlexShmigelskii/Aramco_Inn_test/raw/master/result.png)


To build the .exe app , follow the instructions listed below:

! before you start, make sure you have python version >=3.9 installed !

1) open a cmd and navigate to the current folder using 

```sh
cd \to\your\folder
```

2) Then use virtualenv. If you don't have it installed yet, install it by running the 

```sh
pip install virtualenv
```

command in your command line. Then, create a new virtual environment in your project directory using the command 

```sh
python -m venv venv
```

3) Next, activate the virtual environment using the command 

```sh
.\venv\Scripts\activate
```

4) After that, you need to install the project's dependencies which are listed in a requirements.txt file. Use the command

```sh
pip install -r requirements.txt
```

5) Run the command 

```sh
pyinstaller run_app.spec --clean
```

6) As soon as the script finishes executing, you will find the "dist" folder. Clicking on it, you will find the file "run_app.exe"

7) ! Move the "run_app.exe" application to the folder where the app.py and "data" folder are located (where "dist" folder located)

8) Run "run_app.exe" application 
