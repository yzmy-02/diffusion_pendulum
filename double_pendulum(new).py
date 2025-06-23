import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import os

# Physical parameters
g = 9.81  # Gravitational acceleration
l1, l2 = 2.2, 0.8  # Pendulum lengths
m1, m2 = 1.0, 1.0  # Masses
damping = 0.05  # Damping coefficient (energy loss)


# Equations of motion (with damping)
def equations(t, y):
    θ1, z1, θ2, z2 = y  # Angles and angular velocities
    c, s = np.cos(θ1 - θ2), np.sin(θ1 - θ2)

    # Angular acceleration (with damping term -damping * z)
    θ1_ddot = ((m2 * g * np.sin(θ2) * c - m2 * s * (l1 * z1 ** 2 * c + l2 * z2 ** 2) -
                (m1 + m2) * g * np.sin(θ1)) / (l1 * (m1 + m2 * s ** 2))) - damping * z1

    θ2_ddot = (((m1 + m2) * (l1 * z1 ** 2 * s - g * np.sin(θ2) + g * np.sin(θ1) * c) +
                m2 * l2 * z2 ** 2 * s * c) / (l2 * (m1 + m2 * s ** 2))) - damping * z2

    return [z1, θ1_ddot, z2, θ2_ddot]


# Generate animation and save it
def create_animation(y0, output_path):
    # Time range
    t_span = (0, 10)  # 10 seconds
    t_eval = np.linspace(0, 10, 300)  # 300 time steps

    # Solve the differential equation
    sol = solve_ivp(equations, t_span, y0, t_eval=t_eval, method='RK45')

    # Extract angle data
    θ1, θ2 = sol.y[0], sol.y[2]

    # Calculate the coordinates of the pendulum ends
    x1, y1 = l1 * np.sin(θ1), -l1 * np.cos(θ1)
    x2, y2 = x1 + l2 * np.sin(θ2), y1 - l2 * np.cos(θ2)

    # Create animation
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.3, 3.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # 蓝色连线（不含原点）
    line, = ax.plot([], [], '-', lw=2, color='blue')

    # 红色小球
    mass1, = ax.plot([], [], 'o', color='red', markersize=8)
    mass2, = ax.plot([], [], 'o', color='red', markersize=8)

    def init():
        line.set_data([], [])
        mass1.set_data([], [])
        mass2.set_data([], [])
        return line, mass1, mass2

    def update(frame):
        line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
        mass1.set_data([x1[frame]], [y1[frame]])
        mass2.set_data([x2[frame]], [y2[frame]])
        return line, mass1, mass2

    # Generate the animation
    ani = animation.FuncAnimation(fig, update, frames=len(t_eval), init_func=init, interval=10, blit=False, repeat=False)
    ani.save(output_path, writer='ffmpeg', fps=30)
    plt.close(fig)


# Initialize multiple angles and generate videos in batch
epsilon = 1e-7
initial_angles = []
for i in range(240):
    initial_angles.append(np.pi/3 + epsilon + (2*np.pi*(i+1))/360) # Modify initial angles as needed
output_dir = '/Users/yuqijin/Desktop/output_videos/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

for i, angle in enumerate(initial_angles):
    y0 = [angle, 0, angle, 0]  # Initial angle with different values, initial angular velocity is 0
    output_path = os.path.join(output_dir, f"double_pendulum_{i+1}.mp4")
    create_animation(y0, output_path)
    print(f"Video {output_path} has been saved")
