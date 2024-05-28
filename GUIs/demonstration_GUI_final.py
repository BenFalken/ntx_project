import tkinter as tk
import random
import time

RED = "red"
GREEN = "green"
WIDTH, HEIGHT = 1000, 500

# Function to draw the dot
def draw_dot(canvas, color, x, y):
    canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill=color, outline="", tags="dot")
    

# Function to generate random numbers
def generate_number():
    return random.randint(0, 1)

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


# Create the main window
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

root.mainloop()