import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time

def update_point(scatter, coords):
    # Update the offsets for the scatter plot
    scatter._offsets3d = (np.array([coords[0]]), np.array([coords[1]]), np.array([coords[2]]))
    plt.draw()

def generate_random_coordinates():
    return np.random.rand(3) * 10  # Generate random coordinates within [0, 10) range

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initial coordinates
    initial_coords = generate_random_coordinates()
    scatter = ax.scatter(initial_coords[0], initial_coords[1], initial_coords[2], color='b')

    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')


    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 10])

    plt.ion()  # Turn on interactive mode
    plt.show()

    try:
        while True:
            new_coords = generate_random_coordinates()
            update_point(scatter, new_coords)
            plt.pause(3)  # Pause for 3 seconds
    except KeyboardInterrupt:
        print("Plotting stopped.")

if __name__ == "__main__":
    main()
