# How to Collect Data (as of 3/1/24)
1. Have two computers (call them A and B)
	1. This could all be done on 1 computer if yours is beefy enough
2. Install PsychoPy on the [PsychoPy website](https://www.psychopy.org/download.html) if you haven't already
3. Connect the OpenBCI headset to connect to computer A following this [OpenBCI Tutorial](https://docs.google.com/document/d/1t7J3HIWkAL-0ryzGdOWXVq6LT_B7E0RlaZHDBdEPonE/edit)
4. Connect these electrodes:
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
5. Open the [Emotions (Core) Google form](https://docs.google.com/forms/d/1aDTQFd7sgwAe5aftb-0PMRY1PHD_xGIRps1IsHuxMGY/) on computer B
6. On computer B, download the core videos from the [core videos Google folder](https://drive.google.com/drive/u/4/folders/1S_godbS0hgS7s1HXCP3pEBIY3kTHCRBT) and personalized videos for the particular participant in the [videos Google folder](https://drive.google.com/drive/u/4/folders/1try2zBZe23IJKfLDlCuc-OsLTiF9RCjJ)
7. If you haven't already, clone the [ntx_project Github repo](https://github.com/BenFalken/ntx_project/) on computer B
	1. It's recommended to download and use the [Github Desktop](https://desktop.github.com/) app for ease of committing/syncing/branching/cloning
 	2. To clone using Github Desktop, follow this [Yotube tutorial](https://www.youtube.com/watch?v=PoZNIbs_wx8)
8. Place all the videos in a folder called "Videos" in ntx_project/data_collection
	1. The videos need to be labeled as so (replace # with the PID number):
	* #_amusement.mp4
	* #_anger.mp4
	* #_contentment.mp4
	* #_disgust.mp4
	* #_fear.mp4
	* #_sadness.mp4
	* amusement_core_1.mp4
	* amusement_core_2.mp4
	* anger_core_1.mp4
	* anger_core_2.mp4
	* contentment_core_1.mp4
	* contentment_core_2.mp4
	* disgust_core_1.mp4
	* disgust_core_2.mp4
	* fear_core_1.mp4
	* fear_core_2.mp4
	* neutral_core_1.mp4
	* neutral_core_2.mp4
	* neutral_core_3.mp4
	* neutral_core_4.mp4
	* neutral_core_5.mp4
	* sadness_core_1.mp4
	* sadness_core_2.mp4
9. Have the participant put on the headset and inject gel
10. Start recording EEG data in OpenBCI GUI on computer A
11. Open and run GUI_code.py in the ntx_project/data_collection folder on computer B using PsychoPy (specifically the editor)
12. Participant should now be watching the videos on computer B while EEG data is being recorded on computer A. In between the videos, have them fill out the Google form question for that particular video on computer B
13. When the participant has gone through all the videos, stop the data recording on OpenBCI GUI and upload the data (it saves it in User/Documents/OpenBCI_GUI/Recordings by default) to [EEG Data Google folder](https://drive.google.com/drive/u/4/folders/1t2ojmeJQUX4dkLSUQ174-I80VX6uHoQK). The GUI_code should have also made a video_timestamps_PID_... file, which you should commit and push to the GitHub repo
