from psychopy import visual, core, event
import os, time
import numpy as np

# Create a window
win = visual.Window(size=(800, 600), monitor='testMonitor', color='black')

# A light green text
stim = visual.TextStim(win, 'Click any key to Continue',
                       color=(1, 1, 1), colorSpace='rgb')
stim.colorSpace = 'rgb255'
stim.color = (255, 255, 255)

video_timestamps = np.empty((100, 2))
video_timestamps[:] = np.nan

# Specify the path to your movie file
movie_path = "/Users/benfalken/Desktop/VN_Videos/"
movies = os.listdir(movie_path)

for movie_idx in range(len(movies)):
	movie = movies[movie_idx]
	time_elapsed = 0
	if movie[-3:] == 'mp3':
		continue
	movie = visual.MovieStim3(win, filename=movie_path+movie, size=(800, 600), flipVert=False, flipHoriz=False, loop=False)

	# Start the movie playback
	movie.play()
	video_timestamps[movie_idx][0] = time.time()

	# Run the movie until a key is pressed or the movie ends
	while not event.getKeys() and movie.status != visual.FINISHED and time_elapsed < 200:
	    movie.draw()
	    win.flip()
	    time_elapsed += 1
	video_timestamps[movie_idx][1] = time.time()

	# Wait button
	while not event.getKeys():
		stim.draw()
		win.flip()

# Close the window
win.close()
pkl.dump(video_timestamps, open('video_timestamps', 'wb'))