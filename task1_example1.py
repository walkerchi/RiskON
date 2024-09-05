############## Some Preset Parameters ##############
n_teams = 2 # number of teams
l = 1 # Length scale of the RBF kernel 
var = 1 # Variance of the RBF kernel
ylims = (0, 1)
n_control = 6 # Number of control points for the spline
start_date = "2020-01-01"
end_date = "2025-01-01"
noise = 0.1 # Standard deviation of the noise
seed = 2333 # Seed for reproducibility
color = "dodgerblue" # color for the curve find the colors here https://matplotlib.org/stable/gallery/color/named_colors.html
scatter_color = ["dodgerblue","orangered","limegreen"] # color for the observations
scatter_marker = ["1","2","3"] # marker for the observations
scatter_size = 10 # size of the observations
legend_loc = "upper left" # location of the legendS
dpi = 200 # DPI for the saved figure
fps = 10 # FPS for the saved animation
figsize = (10, 5) # Size of the figure
only_static = True
####################################################
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.dates import DateFormatter, AutoDateLocator, MonthLocator
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
from typing import List 
from datetime import datetime
import GPy

sns.set_style("darkgrid")
matplotlib.use('Agg')

np.random.seed(seed)

def gen_curve(grid:int=100, n:int=5, xlims=(0,1), ylims=(0,1)):
    x_control = np.linspace(xlims[0], xlims[1], n)
    y_control = np.random.rand(n) * ylims[1] + ylims[0]
    spline = make_interp_spline(x_control, y_control, k=3)
    x = np.linspace(xlims[0], xlims[1], grid)
    return x, spline(x)

def add_noise(x, std:float=0.1):
    return x * (1 + np.random.normal(0, std, x.shape))

def date_range(start_date:str="2020-01-01", 
               end_date:str="2025-01-01"
               )->List[datetime]:
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  # 'MS' stands for Month Start

    # Convert to list of datetime objects (if needed)
    datetime_list = date_range.to_pydatetime()
    return datetime_list

dates = date_range(start_date,end_date)

# Initial dataset (empty, will add points during animation)
y_trues = []
for _ in range(n_teams):
    x, y_true = gen_curve(grid=len(dates), n=n_control, ylims=ylims)
    y_trues.append(y_true)
y_trues = np.stack(y_trues, 0) # [n_team, len(dates)]
y = add_noise(y_trues, std=noise).T # [len(dates), n_team]
x = x.reshape(-1, 1)

# Test data (for plotting GP predictions)
X_test = np.linspace(0, 1, len(dates)).reshape(-1, 1)

# Step 2: Define kernel and GP model
kernel = GPy.kern.RBF(input_dim=1, variance=var, lengthscale=l)
# gp_model = GPy.models.GPRegression(x, y, kernel)


if not only_static:
    # Step 3: Set up the figure, axis, and plot element to animate
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(dates[0], dates[-1])
    ax.set_ylim(ylims[0], ylims[1]*n_teams)
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount in CHF(Million)")

    line, = ax.plot([], [], lw=2, label="Mean Prediction", color=color)
    fill_between = None  # For confidence intervals
    observations = []
    for i in range(n_teams):
        observations.append(ax.scatter([], [], s=scatter_size, c=scatter_color[i], marker=scatter_marker[i], label=f"Team {i} Observations"))
    observations.append(ax.scatter([], [], s=scatter_size, c=scatter_color[n_teams], marker=scatter_marker[n_teams], label=f"Observations Sum"))

    # Set date format on x-axis
    date_format = DateFormatter("%Y-%m")
    ax.xaxis.set_major_formatter(date_format)
    months = MonthLocator(bymonth=(1, 6))
    ax.xaxis.set_major_locator(months)

    ax.legend(loc=legend_loc)

    # Step 4: Initialize the plot elements
    def init():
        line.set_data([], [])
        for observation in observations:
            observation.set_offsets(np.c_[[], []])
        return [line, *observations]

    pbar = tqdm(total=len(x)-1, desc="Animating", colour="green")

    # Step 5: Define the update function for the animation
    def update(frame):
        global x, y, fill_between, color

        if frame == 0:
            return [line, *observations]

        # Add a new point from the true function (simulate more observations over time)
        new_x = x[:frame].repeat(n_teams, 1)
        new_y = y[:frame]

        # Update the GP model with new data
        gp_model = GPy.models.SparseGPRegression(new_x, new_y, kernel, num_inducing=32)
        gp_model.optimize()

        # Predict using the GP
        Y_pred, Y_var = gp_model.predict(X_test)
        Y_pred = Y_pred.sum(-1)
        Y_var  = Y_var[:,-1]

        # Update plot data
        line.set_data(dates, Y_pred)

        # Remove old confidence interval if it exists
        if fill_between:
            fill_between.remove()

        # Update the confidence intervals (mean +/- 1.96 * stddev)
        fill_between = ax.fill_between(dates, 
                                    (Y_pred - 1.96 * np.sqrt(Y_var)).flatten(), 
                                    (Y_pred + 1.96 * np.sqrt(Y_var)).flatten(), 
                                    color=color, alpha=0.3)

        # Update observed points
        for i in range(n_teams):
            observations[i].set_offsets(np.c_[dates[:frame], new_y[:,i]])
        observations[n_teams].set_offsets(np.c_[dates[:frame], new_y.sum(-1)])

        pbar.update(1)

        return [line, *observations, fill_between]

    # Step 6: Animate the plot using FuncAnimation
    ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, interval=200)
    os.makedirs("outputs",exist_ok=True)
    ani.save("outputs/gp_animation.mp4", fps=fps, dpi=dpi)

fig, ax = plt.subplots(figsize=(12,6))
ax.set_xlim(dates[0], dates[-1])
ax.set_ylim(ylims[0], ylims[1]*n_teams)
ax.set_xlabel("Date", fontdict={'fontsize':14})
ax.set_ylabel("Amount in CHF(Million)", fontdict={'fontsize':14})

new_x = x.repeat(n_teams, 1)
new_y = y

gp_model = GPy.models.SparseGPRegression(new_x, new_y, kernel, num_inducing=32)
gp_model.optimize()

Y_pred, Y_var = gp_model.predict(X_test)
Y_pred = Y_pred.sum(-1)
Y_var  = Y_var[:,-1]

ax.plot(dates, Y_pred, color=color, label="Mean")


fill_between = ax.fill_between(dates, 
                            (Y_pred - 1.96 * np.sqrt(Y_var)).flatten(), 
                            (Y_pred + 1.96 * np.sqrt(Y_var)).flatten(), 
                            color=color, alpha=0.3)

# Update observed points
for i in range(n_teams):
    ax.scatter(dates, new_y[:,i],s=scatter_size, c=scatter_color[i], marker=scatter_marker[i], label=f"Team {i} Observations")
ax.scatter(dates, new_y.sum(-1), s=scatter_size, c=scatter_color[n_teams], marker=scatter_marker[n_teams], label=f"Observations Sum")

ax.legend()
fig.savefig("outputs/gp_static.png")