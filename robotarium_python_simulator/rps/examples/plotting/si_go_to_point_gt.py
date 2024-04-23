import pdb

import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox



pred1_locations = [[2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 4], [2, 4], [1, 4], [1, 4], [0, 4], [0, 4], [1, 4], [0, 4], [0, 4], [0, 4], [1, 4], [1, 4], [2, 4], [3, 4], [3, 4], [3, 4], [1, 3]]
pred1_locations_c = [[2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 4], [2, 4], [1, 4], [1, 4], [0, 4], [0, 4], [1, 4], [0, 4], [0, 4], [0, 4], [1, 4], [1, 4], [2, 4], [3, 4], [3, 4], [3, 4], [1, 3]]
pred2_locations = [[0, 2], [0, 3], [0, 3], [1, 3], [1, 4], [1, 4], [1, 4], [0, 4], [0, 4], [0, 4], [0, 4], [1, 4], [0, 4], [0, 4], [0, 4], [1, 4], [1, 4], [2, 4], [3, 4], [3, 4], [3, 4], [3, 1]]
pred2_locations_c = [[0, 2], [0, 3], [0, 3], [1, 3], [1, 4], [1, 4], [1, 4], [0, 4], [0, 4], [0, 4], [0, 4], [1, 4], [0, 4], [0, 4], [0, 4], [1, 4], [1, 4], [2, 4], [3, 4], [3, 4], [3, 4], [3, 1]]
action1_locations = [[1, 1], [2, 1], [2, 1], [2, 2], [1, 2], [1, 3], [1, 4], [1, 4], [0, 4], [0, 4], [0, 4], [0, 3], [1, 3], [0, 3], [0, 3], [1, 3], [1, 4], [2, 4], [2, 4], [3, 4], [3, 4], [4, 2]]
action1_locations_c = [[1, 1], [2, 1], [2, 1], [2, 2], [1, 2], [1, 3], [1, 4], [1, 4], [0, 4], [0, 4], [0, 4], [0, 3], [1, 3], [0, 3], [0, 3], [1, 3], [1, 4], [2, 4], [2, 4], [3, 4], [3, 4], [4, 2]]
fire1_locations = [[3, 4], [2, 4], [1, 4], [0, 4], [0, 4], [0, 4], [0, 4], [0, 4], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [2, 4], [2, 4], [2, 4], [2, 4], [3, 4], [3, 4], [0, 4]]
fire2_locations = [None, [3, 4], [2, 4], [1, 4], [1, 4], [1, 4], [1, 4], [2, 4], [0, 4], [0, 4], [2, 4], [2, 4], [2, 4], [2, 4], [3, 4], [3, 4], [3, 4], [3, 4], None, None, None]
fire3_locations = [None, None, [3, 4], [2, 4], [2, 4], [2, 4], [2, 4], [3, 4], [2, 4], [2, 4], [3, 4], [3, 4], [3, 4], [3, 4], None, None, None, None, None, None, None]
fire4_locations = [None, None, None, [3, 4], [3, 4], [3, 4], [3, 4], None, [3, 4], [3, 4], None, None, None, None, None, None, None, None, None, None, None]

coord_transform_x = {0:-.8, 1:-.4, 2:0, 3:.4, 4:.8}
coord_transform_y = {0:.8, 1:.4, 2:0, 3:-.4, 4:-.8}

for e, (coord0, coord1) in enumerate(pred1_locations):
    pred1_locations[e][0] = coord_transform_x[coord0]
    pred1_locations[e][1] = coord_transform_y[coord1]

for e, (coord0, coord1) in enumerate(pred2_locations):
    pred2_locations[e][0] = coord_transform_x[coord0]
    pred2_locations[e][1] = coord_transform_y[coord1]

for e, (coord0, coord1) in enumerate(action1_locations):
    action1_locations[e][0] = coord_transform_x[coord0]
    action1_locations[e][1] = coord_transform_y[coord1]
scaling_factor = 1
def create_goal_points(i):
    goal_points = np.zeros((3,3))
    goal_points[:2,:] = scaling_factor * np.array([pred1_locations[i], pred2_locations[i], action1_locations[i]]).T
    return goal_points

# Instantiate Robotarium object
N = 3
initial_conditions = create_goal_points(0) # np.array(np.mat('1 0.5 -0.5 0 0.28; 0.8 -0.3 -0.75 0.1 0.34; 0 0 0 0 0'))

r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=False)

# Define goal points by removing orientation from poses
goal_points = create_goal_points(1) # generate_initial_conditions(N, width=r.boundaries[2]-2*r.robot_diameter, height = r.boundaries[3]-2*r.robot_diameter, spacing=0.5)
print(goal_points)
# Create single integrator position controller
single_integrator_position_controller = create_si_position_controller()

# Create barrier certificates to avoid collision
#si_barrier_cert = create_single_integrator_barrier_certificate()
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

_, uni_to_si_states = create_si_to_uni_mapping()

# Create mapping from single integrator velocity commands to unicycle velocity commands
si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()

#Read in and scale image
gt_img = plt.imread('GTLogo.png')
x_img = np.linspace(-1.0, 1.0, gt_img.shape[1])
y_img = np.linspace(-1.0, 1.0, gt_img.shape[0])

gt_img_handle = r.axes.imshow(gt_img, extent=(-1, 1, -1, 1))


# define x initially
x = r.get_poses()
x_si = uni_to_si_states(x)

# Plotting Parameters
CM = np.random.rand(N,3) # Random Colors
goal_marker_size_m = 0.2
robot_marker_size_m = 0.15
marker_size_goal = determine_marker_size(r,goal_marker_size_m)
marker_size_robot = determine_marker_size(r, robot_marker_size_m)
font_size = determine_font_size(r,0.1)
line_width = 5

# # Create Goal Point Markers
# #Text with goal identification
# goal_caption = ['G{0}'.format(ii) for ii in range(goal_points.shape[1])]
# #Plot text for caption
# goal_points_text = [r.axes.text(goal_points[0,ii], goal_points[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-2)
# for ii in range(goal_points.shape[1])]
# goal_markers = [r.axes.scatter(goal_points[0,ii], goal_points[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-2)
# for ii in range(goal_points.shape[1])]
robot_markers = [r.axes.scatter(x[0,ii], x[1,ii], s=marker_size_robot, marker='o', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width)
for ii in range(goal_points.shape[1])]



r.step()

# While the number of robots at the required poses is less
# than N...
for i in range(1, 21):
    print(i)
    goal_points = create_goal_points(i)
    print('predator_1 loc',pred1_locations_c[i])
    print('predator_2_loc', pred2_locations_c[i])
    print('action_1_loc', action1_locations_c[i])
    # print(goal_points)
    time.sleep(2)



    if i > 1:
        ab.remove()
        try:
            bc.remove()
            cd.remove()
            de.remove()
        except:
            pass
    fire_img = plt.imread('FireLogo.png')
    imagebox = OffsetImage(fire_img, zoom=0.13)  # switch to .23
    ab = AnnotationBbox(imagebox, (coord_transform_x[fire1_locations[i][0]], coord_transform_y[fire1_locations[i][1]]), bboxprops=dict(edgecolor='none'))
    r.axes.add_artist(ab)

    if fire2_locations[i] is not None:
        fire_img = plt.imread('FireLogo.png')
        imagebox2 = OffsetImage(fire_img, zoom=0.13)  # switch to .23
        bc = AnnotationBbox(imagebox2, (coord_transform_x[fire2_locations[i][0]], coord_transform_y[fire2_locations[i][1]]), bboxprops=dict(edgecolor='none'))
        r.axes.add_artist(bc)

    if fire3_locations[i] is not None:
        fire_img = plt.imread('FireLogo.png')
        imagebox3 = OffsetImage(fire_img, zoom=0.13)
        cd = AnnotationBbox(imagebox3, (coord_transform_x[fire3_locations[i][0]], coord_transform_y[fire3_locations[i][1]]), bboxprops=dict(edgecolor='none'))
        r.axes.add_artist(cd)

    if fire4_locations[i] is not None:
        fire_img = plt.imread('FireLogo.png')
        imagebox4 = OffsetImage(fire_img, zoom=0.13)
        de = AnnotationBbox(imagebox4, (coord_transform_x[fire4_locations[i][0]], coord_transform_y[fire4_locations[i][1]]), bboxprops=dict(edgecolor='none'))
        r.axes.add_artist(de)



    # # # x_img = np.linspace(np.random.uniform(-2, 1), 1.0, fire_img.shape[1])
    # # # y_img = np.linspace(-1.0, 1.0, fire_img.shape[0])
    # fire_img_handle = r.axes.imshow(fire_img, extent=(np.random.uniform(np.random.randint(3),.2), .1, -.1, .1))
    # remove fire_img




    while (np.size(at_pose(np.vstack((x_si,x[2,:])), goal_points, position_error=.14, rotation_error=100)) != N):

        # Get poses of agents
        x = r.get_poses()
        x_si = uni_to_si_states(x)

        #Update Plot
        # Update Robot Marker Plotted Visualization
        for i in range(x.shape[1]):
            robot_markers[i].set_offsets(x[:2,i].T)
            # This updates the marker sizes if the figure window size is changed.
            # This should be removed when submitting to the Robotarium.
            robot_markers[i].set_sizes([determine_marker_size(r, robot_marker_size_m)])

        # for j in range(goal_points.shape[1]):
        #     goal_markers[j].set_sizes([determine_marker_size(r, goal_marker_size_m)])

        # Create single-integrator control inputs
        dxi = single_integrator_position_controller(x_si, goal_points[:2][:])

        # Create safe control inputs (i.e., no collisions)
        dxi = si_barrier_cert(dxi, x_si)

        # Transform single integrator velocity commands to unicycle
        dxu = si_to_uni_dyn(dxi, x)

        # Set the velocities by mapping the single-integrator inputs to unciycle inputs
        r.set_velocities(np.arange(N), dxu)

        # Iterate the simulation
        r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
