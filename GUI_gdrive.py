from psychopy import visual, core, event
import os, time
import numpy as np
import pickle as pkl

# Modify these globals to change the max length of the videos
MAX_VIDEO_LENGTH = 60 # in seconds
REFRESH_RATE = 60

# Create a window
win = visual.Window(size=(800,600), monitor='testMonitor', color='black')

# A light green text
stim = visual.TextStim(win, 'Click any key to continue', color=(1, 1, 1), colorSpace='rgb')
stim.colorSpace = 'rgb255'
stim.color = (255, 255, 255)

questionnaire_text = visual.TextStim(win, 'Please fill out the questionnaire. Press the space bar to continue. ', color=(1, 1, 1), colorSpace='rgb')
questionnaire_text.colorSpace = 'rgb255'
questionnaire_text.color = (255, 255, 255)

video_timestamps = np.empty((100, 2))
video_timestamps[:] = np.nan

# Specify the path to your movie file
movie_path = os.path.join(os.path.dirname(__file__), "Videos/") ## Place video files in a folder called "Videos" in the ntx_project folder
movies = os.listdir(movie_path)

exit_trial = False

for movie_idx in range(len(movies)):
	movie = movies[movie_idx]
	time_elapsed = 0
	movie = visual.MovieStim3(win, filename=movie_path+movie, flipVert=False, flipHoriz=False, loop=False)

	# Start the movie playback
	movie.play()
	video_timestamps[movie_idx][0] = time.time()

	# Run the movie until a key is pressed or the movie ends
	while not exit_trial and movie.status != visual.FINISHED and time_elapsed < MAX_VIDEO_LENGTH * REFRESH_RATE:
		if event.getKeys(keyList=['escape', 'q']):
			exit_trial=True
			break
		movie.draw()
		win.flip()
		time_elapsed += 1
		event.clearEvents()

	movie.stop()
	video_timestamps[movie_idx][1] = time.time()

	if exit_trial:
		win.close()
		core.quit()
		break

	while True:
		if event.getKeys(keyList=['space']):
			break
		questionnaire_text.draw()
		win.flip()

win.close()
pkl.dump(video_timestamps, open('video_timestamps', 'wb'))