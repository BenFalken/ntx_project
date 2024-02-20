from psychopy import visual, core, event
import os, time
import numpy as np
import pickle as pkl

# Modify these globals to change the max length of the videos
MAX_VIDEO_LENGTH = 60 # in seconds
REFRESH_RATE = 60

# Create a window
win = visual.Window(size=(1000,800), monitor='testMonitor', color='black')
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
video_timestamps = np.empty((20, 2))
video_timestamps[:] = np.nan

# Specify the path to your movie file
movie_path = os.path.join(os.path.dirname(__file__), "Videos/") ## Place video files in a folder called "Videos" in the ntx_project folder
movies = os.listdir(movie_path)

#Draw the [stim] screen
stim.draw()
win.flip()
event.waitKeys()

for movie_idx in range(len(movies)):
	exit_trial = False
	movie = movies[movie_idx]
	time_elapsed = 0

	#set movie dimensions

	# movie = visual.MovieStim3(win, filename=movie_path+movie, flipVert=False, flipHoriz=False, loop=False)
	# video_width, video_height = movie.getMovieFrameSize()
	# window_width, window_height = win.size
	# width_scale = window_width / video_width
	# height_scale = window_height / video_height
	# scale_factor = min(width_scale, height_scale)
	# scaled_width = int(video_width * scale_factor)
	# scaled_height = int(video_height * scale_factor)
	# pos = (0,0)

	movie = visual.MovieStim3(win, filename=movie_path+movie, flipVert=False, flipHoriz=False, loop=False)


	# Start the movie playback
	movie.play()
	video_timestamps[movie_idx][0] = time.time()

	# Run the movie until ESC key is pressed or the movie ends
	while not exit_trial and movie.status != visual.FINISHED and time_elapsed < MAX_VIDEO_LENGTH * REFRESH_RATE:
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
				core.quit()
			elif key == 'space':
				print('Completed video #'+ str(movie_idx+1))
				break
		if 'escape' in keys or 'space' in keys:
			break

win.close()
pkl.dump(video_timestamps, open('video_timestamps', 'wb'))
print(video_timestamps)