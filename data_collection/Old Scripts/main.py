import subprocess
import time

subprocess.Popen('start /wait python CytonDataPackager.py', shell=True)
time.sleep(1)
subprocess.Popen('start /wait python DataCollector.py', shell=True)
subprocess.Popen('start /wait python GUI_code.py', shell=True)