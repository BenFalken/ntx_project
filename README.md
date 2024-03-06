# How to Collect Data (as of 3/1/24)
1. Have two computers (call them A and B)
	1. This could all be done on 1 computer if yours is beefy enough
2. Connect the OpenBCI headset to connect to computer A following this [OpenBCI Tutorial](https://docs.google.com/document/d/1t7J3HIWkAL-0ryzGdOWXVq6LT_B7E0RlaZHDBdEPonE/edit)
3. Connect these electrodes:
	1. GND (black)
 	2. REF (white)
	3. CH1 (gray) - C3
	4. CH2 (purple) - Cz
	5. CH3 (blue) - C4
	6. CH4 (green) - P3
	7. CH5 (yellow) - Pz
	8. CH6 (orange) - P4
	9. CH7 (red) - O1
	10. CH8 (brown) - O2
4. Open the [Emotions (Core) Google form](https://docs.google.com/forms/d/1aDTQFd7sgwAe5aftb-0PMRY1PHD_xGIRps1IsHuxMGY/) on computer B
5. On computer B, download the core videos from the [core videos Google folder](https://drive.google.com/drive/u/4/folders/1S_godbS0hgS7s1HXCP3pEBIY3kTHCRBT) and personalized videos for the particular participant in the [videos Google folder](https://drive.google.com/drive/u/4/folders/1try2zBZe23IJKfLDlCuc-OsLTiF9RCjJ)
6. If you haven't already, clone the [ntx_project Github repo](https://github.com/BenFalken/ntx_project/) on computer B
	1. I recommend downloading and using the [Github Desktop](https://desktop.github.com/) app for ease of committing/syncing/branching/cloning
 	2. To clone using Github Desktop, follow this [Yotube tutorial](https://www.youtube.com/watch?v=PoZNIbs_wx8)
7. Place all the videos in a folder called "Videos" in ntx_project/data_collection
	1. This will allow the code to play the videos
8. Have the participant put on the headset and inject gel
9. Start recording EEG data in OpenBCI GUI on computer A
10. Open and run GUI_code.py in the ntx_project/data_collection folder on computer B
11. Participant should now be watching the videos on computer B while EEG data is being recorded on computer A. In between the videos, have them fill out the Google form question for that particular video on computer B
12. When the participant has gone through all the videos, stop the data recording on OpenBCI GUI and upload the data (it saves it in User/Documents/OpenBCI_GUI/Recordings by default) to [EEG Data Google folder](https://drive.google.com/drive/u/4/folders/1t2ojmeJQUX4dkLSUQ174-I80VX6uHoQK). The GUI_code should have also made a video_timestamps_PID_... file, which you should commit and push to the GitHub repo
