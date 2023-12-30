# Route Planning
Route planning project aims to calculate the best route you can take using a bicycle from your current address to the required destination and can give you an estimate travel time through taking inputs of source address and destination address and whether you want to use bike sharing or use your own bike.

# Setup

- First, you need to create a virtual environment using the following command:
`python3 -m venv env`
make sure python venv is installed and if not install it with the following command:
`python3 -m pip install venv`
- Activate the virtual environment
`source env/bin/activate`
- Install all the dependencies from the requirements folder
`pip install --upgrade pip`
`pip install -r requirements.txt`
- install tkinter to be able to run the interface
For Linux `sudo apt-get install python3-tk`
For macos `brew install python-tk`

# Run the project

The GUI interface is available in `interface.py` file and `utils.py` is the file that has the classes and methods required to calculate the travel time, plan the route and plot it graphically. Also, there is an ipynb file that you can use to run the project through notebook.
`python interface.py`