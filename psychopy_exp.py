from psychopy import visual, core, event
import os, time
import numpy as np

# Modify these globals to change the max lenght of the videos
MAX_VIDEO_LENGTH = 40 # in seconds
REFRESH_RATE = 60

# Create a window
win = visual.Window(size=(800, 600), monitor='testMonitor', color='black')

if ((showVideo !== undefined)) {
let src = 'youtube.html';
params = {};  // Results added here after form is submitted
continue_routine = true; // Routines can't be ended from within Begin Routine
$(document).ready(function() {
    // Add custom contents from html file using an iframe:
    $('body').append('<div id="iframe-o" style="visibility: hidden; position: relative; display: table; margin: auto;"><div id="iframe-m" style="display: table-cell; vertical-align: middle;"><div id="iframe-i" style="display: inline-block; width:100%; overflow-y: auto; overflow-x: hidden;"><iframe id="iframe" src="'+src+'" frameborder=0 style="width: 100%"></iframe></div></div></div>');

    $('#iframe').on('load',function(iframe){
        // Auto-adjust iframe size:
        $(this).contents().find('html').css({ 'display': 'table', 'width': '100%', 'overflow-x': 'hidden' });
        $('#iframe-o').height($(window).height()-20, true);
        $('#iframe-m').width($(this).contents().find('html').width()+20);
        $('#iframe-i').height ( Math.min ( $(this).contents().find('html').height()+20, $(window).height()-20 ), true );
        $(this).height($(this).contents().find('html').height());
        $('#iframe-o').css('visibility','visible');

        // If iframe contains a form, then capture its output:
        $(this).contents().find('form').on('submit',function(e){
            e.preventDefault();
            $.each($(this).serializeArray(),function(i, param){
                params[param.name] = param.value;
                psychoJS.experiment.addData(param.name, param.value);
            });
            console.log ( 'DEBUG:FRM', params );
            // Remove iframe and continue to next routine when done:
            $('#iframe-o').remove();
            continue_routine = false;
        });
    });
});
//$('#iframe').attr( 'src', function(i,val){ return val;} );
}

# A light green text
stim = visual.TextStim(win, 'Click any key to Continue', color=(1, 1, 1), colorSpace='rgb')
stim.colorSpace = 'rgb255'
stim.color = (255, 255, 255)

video_timestamps = np.empty((100, 2))
video_timestamps[:] = np.nan

# Specify the path to your movie file
movie_path = os.path.join(os.path.dirname(__file__), "Videos/")
#movie_path = "/Users/ethan/Desktop/Code/ntx_project/Videos/"
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
	while not event.getKeys() and movie.status != visual.FINISHED and time_elapsed < MAX_VIDEO_LENGTH * REFRESH_RATE:
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
#pkl.dump(video_timestamps, open('video_timestamps', 'wb'))