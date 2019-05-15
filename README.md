# eagle-eye

HOW TO INSTALL:
- Clone the repo
- Go to the project folder
- install python virtual environment with the following commend
sudo apt install virtualenv
- create your virtual environment
virtualenv -p python3 env
- activate your env 
source env/bin/activate
make sure that (env) shown before your username in the terminal
- install the dependancies 
pip3 install -r requirements.txt

HOW TO RUN:
python3 main.py 
open /run?imshow=false
open /

HOW TO CALC ACC:
- run python3 main.py
- run python3 ground_truth_parser.py
- run python3 calc_acc.py