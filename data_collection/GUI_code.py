PERSONALIZED = 5
CORE = 0
EXPANDED = 0

PID = "09"
NAME = "Ethan"
TRIAL_TYPE = "Core"

# Modify these globals to change the max length of the videos
MAX_VIDEO_LENGTH = 90 # in seconds
FPS = 60

from psychopy import visual, core, event
import os, time
import numpy as np
import pickle as pkl
import csv
from videoprops import get_video_properties

start_time = time.time()

win_width=1200
win_height=800

# Create a window
win = visual.Window(size=(win_width,win_height), monitor='testMonitor', color='black')
color = (130, 180, 230)
width = 2

# Define the two screens ([stim] and [questionnaire_text]) we'll be using during GUI operation
stim = visual.TextStim(win, 'Click any key to continue. \n The video may take a moment to start.', wrapWidth=width)
stim.colorSpace = 'rgb255'
stim.color = color

questionnaire_text = visual.TextStim(win, 'Please fill out the questionnaire. \n When you are finished press the space bar to continue. \n The video may take a moment to start.\n \n Press ESC to end the trial.', wrapWidth=width)
questionnaire_text.colorSpace = 'rgb255'
questionnaire_text.color = color

#define array containing start and end times of each video clip
video_timestamps = np.empty((PERSONALIZED + CORE + EXPANDED, 2))
video_timestamps[:] = np.nan

# Specify the path to your movie file
movie_path = os.path.join(os.path.dirname(__file__), "Videos/") ## Place video files in a folder called "Videos" in the ntx_project folder

movies = os.listdir(movie_path)

# check to make sure the number of movies is equal to the number of timestamps
if len(movies) != PERSONALIZED + CORE + EXPANDED:
	print("Different number of movies and timestamps!")
	exit(1)

#Draw the [stim] screen
stim.draw()
win.flip()
event.waitKeys()

breakOut = False

for movie_idx in range(len(movies)):
	exit_trial = False
	movie = movies[movie_idx]
	time_elapsed = 0

	props = get_video_properties(movie_path+movie)

	width, height = props['display_aspect_ratio'].split(":")
	width, height = int(width), int(height)
	width_multiplier = win_height/height


	movie = visual.MovieStim(win, filename=movie_path+movie, size=(width_multiplier*width, win_height), flipVert=False, flipHoriz=False, loop=False)
	movie.pos = (0, 0)
	# Start the movie playback
	movie.play()
	video_timestamps[movie_idx][0] = time.time()

	# Run the movie until ESC key is pressed or the movie ends
	while not exit_trial and movie.status != visual.FINISHED and time_elapsed < MAX_VIDEO_LENGTH * FPS:
		if event.getKeys(keyList=['escape']):
			exit_trial=True
		movie.draw()
		win.flip()
		time_elapsed += 1

	# Record the end time of the movie and load the questionnaire screen
	movie.stop()
	video_timestamps[movie_idx][1] = time.time()
	questionnaire_text.draw()
	win.flip()

	#Close the window when ESC is pressed, advance to next video when SPACE is pressed
	while True:
		event.clearEvents()
		keys = event.getKeys()
		for key in keys:
			if key == 'escape':
				win.close()
				breakOut = True
				break

			elif key == 'space':
				print('Completed video #'+ str(movie_idx+1))
				# if movie_idx == len(movies) - 1:
				# 	win.close()
				# 	core.quit()
				break
		if 'escape' in keys or 'space' in keys:
			break
	if breakOut:
		break

win.close()
# pkl.dump(video_timestamps, open('video_timestamps', 'wb'))
np.savetxt("video_timestamps_PID_{}_{}_{}.csv".format(PID, NAME, TRIAL_TYPE), video_timestamps, delimiter=",")
# csv_file = 'video_timestamps.csv'
# with open(csv_file, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(video_timestamps)