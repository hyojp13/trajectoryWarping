## Installation Instructions

Please run the following commands to install:
```
git clone git@github.com:hyojp13/retargeting.git && cd retargeting
conda create --name ret && conda activate ret
conda install anaconda::pip
pip install -r requirements.txt
```

Run the test file using:
```
mjpython test.py
```

Running the test file will run a simple retargeted motion of a fryingpan. Press 0 in MuJoCo to remove the red blocks around the scene.

To see the original motion, run:
```
mjpython playTrajectory.py
```

To modify the desired trajectory to retarget, edit the AGENT and TASK variables for each respective file, where AGENT is the folder inside startingTrajectories, and TASK is the trajectory inside the AGENT folder.