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
