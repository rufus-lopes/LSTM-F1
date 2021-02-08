# Formula 1 Data Science Project

Project Aims:

1. Create and capture synthetic data from the Codemasters Formula 1 Game

2. Create a tool to visualise car sensor data and performance metrics

3. Build machine learning models to predict the lap time for a driver mid-race

4. Integrate machine learning predictions into visualisation tool


# Using the Software

1. Clone the repository move to the base directory.

2. In a terminal window run the command 'python run.py'


Note:  

- Lap data is stored in sqlite3 files

- Currently a very basic visualisation tool exists in the analyses folder

# Software Architecture

- Main Script - capturepacket.py

  Runs 5 threads:

    1. Receiver Thread - Receives incoming telemetry packets via the network and passes them to the Recorder for storage

    2. Recorder Thread - Takes packet from the receiver thread and saves these packets to the session SQL file

    3. WaitConsole Thread - Runs until console input is available (or it is asked to quit before)

    4. Live Merging Thread - Collects live packet data and creates a merged pandas dataframe of all key data for the current lap

    5. Live Average Thread - Receives live packet dataframe from the merged frame and calculates the rolling average and cumulative sum of key variables

  Plus key class: live_storage - collects live packets and stores in memory for the Live Merging thread
