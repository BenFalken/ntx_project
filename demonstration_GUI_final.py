import tkinter as tk
import random
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

RED = "red"
GREEN = "green"
WIDTH, HEIGHT = 1000, 500

NUM_POINTS = (
    20  # Number of points to display at a time (last 1 minute / 3 seconds per update)
)

# Initialize VAD tracking
cumulative_valence = 0
cumulative_arousal = 0
cumulative_dominance = 0

# Store history of updates to draw a line graph
valence_history = [0] * NUM_POINTS
arousal_history = [0] * NUM_POINTS
dominance_history = [0] * NUM_POINTS


# Function to draw the dot
def draw_dot(canvas, color, x, y):
    canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill=color, outline="", tags="dot")
    

# Function to generate random numbers
def generate_number():
    return random.randint(0, 1)

def update_vad():
    global cumulative_valence, cumulative_arousal, cumulative_dominance
    number = generate_number()
    number1 = generate_number()
    number2 = generate_number()

    cumulative_valence += 1 if number == 1 else -1
    cumulative_arousal += 1 if number1 == 1 else -1
    cumulative_dominance += 1 if number2 == 1 else -1

    valence_history.append(cumulative_valence)
    arousal_history.append(cumulative_arousal)
    dominance_history.append(cumulative_dominance)
    del valence_history[0]
    del arousal_history[0]
    del dominance_history[0]

def animate(i):
    update_vad()
    ax.clear()
    ax.plot(valence_history, label="Valence", color="blue", alpha = 0.5)
    ax.plot(arousal_history, label="Arousal", color="gray", alpha=0.5)
    ax.plot(dominance_history, label="Dominance", color="red", alpha=0.5)
    ax.legend(loc="upper left")
    ax.set_ylim([-20, 20])  # Adjust based on expected range of VAD values
    ax.set_title("Cumulative VAD over Time")
    ax.set_ylabel("Cumulative Value")
    ax.set_xlabel("Time (s)")


def update_periodically(canvas):
    number = generate_number()
    number1 = generate_number()
    number2 = generate_number()
    update_dot(canvas, number, number1, number2)
    root.after(3000, update_periodically, canvas)  # Update every 3 seconds

# Function to update the dot position based on the generated number
def update_dot(canvas, number, number1, number2):
    canvas.delete("dot")  # Clear previous dot
    if number == 1:
        draw_dot(canvas, "white", 125, HEIGHT // 4)  # Top half of the screen
    if number == 0:
        draw_dot(canvas, "white", 125, 3* HEIGHT // 4)  # Bottom half of the screen
    if number1 == 1:
        draw_dot(canvas, "white", 325, HEIGHT // 4)  # Top half of the screen
    if number1 == 0:
        draw_dot(canvas, "white", 325, 3* HEIGHT // 4)  # Bottom half of the screen
    if number2 == 1:
        draw_dot(canvas, "white", 525, HEIGHT // 4)  # Top half of the screen
    if number2 == 0:
        draw_dot(canvas, "white", 525, 3* HEIGHT // 4)  # Bottom half of the screen

## Function to update the Tkinter canvas
#def update_tkinter_canvas():
#    update_vad()
#    root.after(3000, update_tkinter_canvas)  # Update every 3 seconds


# Create the main Tkinter window
root = tk.Tk()
root.title("Demonstration GUI")

# Create a canvas to draw on
canvas = tk.Canvas(root, width=1000, height=500, bg="white")
canvas.pack()

# Draw the rectangles
canvas.create_rectangle(50, 50, 200, 250, fill="#9fc5e8", outline="")
canvas.create_rectangle(50, 250, 200, 450, fill="#0b5394", outline="")  

canvas.create_rectangle(250, 50, 400, 250, fill="#d3d3d3", outline="")
canvas.create_rectangle(250, 250, 400, 450, fill="#999999", outline="")

canvas.create_rectangle(450, 50, 600, 250, fill="#ecb3cc", outline="")  
canvas.create_rectangle(450, 250, 600, 450, fill="#a64d79", outline="")

valence = tk.Label(root, text="Valence", fg="black", bg="white", font=("Consolas", 20))
valence.place(x=75, y=10)

arousal = tk.Label(root, text="Arousal", fg="black", bg="white", font=("Consolas", 20))
arousal.place(x=275, y=10)

dominance = tk.Label(root, text="Dominance", fg="black", bg="white", font=("Consolas", 20))
dominance.place(x=475, y=10)

high = tk.Label(root, text="HIGH", fg="black", bg="white", font=("Consolas", 10))
high.place(x=10, y=150)

low = tk.Label(root, text="LOW", fg="black", bg="white", font=("Consolas", 10))
low.place(x=10, y=350)


update_periodically(canvas)

# Set up Matplotlib animation
fig, ax = plt.subplots()
ani = FuncAnimation(fig, animate, interval=3000)

# Show Matplotlib plot in non-blocking mode
plt.show(block=False)

root.mainloop()