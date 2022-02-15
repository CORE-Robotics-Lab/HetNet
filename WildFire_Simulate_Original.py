"""
# *******************<><><><><>**************************
# * Script for Wildfire Environment Python  Translation *
# *******************<><><><><>**************************
#
# This script and all its dependencies are implemented by: Esmaeil Seraj
#   - Esmaeil Seraj, CORE Robotics Lab, Robotics & Intelligent Machines,
#   Georgia Tech, Atlanta, GA, USA
#   - email <eseraj3@gatech.edu>
#   - website <https://github.gatech.edu/MCG-Lab/DistributedControl>
#
# Published under GNU GENERAL PUBLIC LICENSE ver. 3 (or any later version)
#
"""

import numpy as np


# wildfire simulation
class WildFire(object):

    def __init__(self, terrain_sizes=None, hotspot_areas=None, num_ign_points=None, duration=None,
                 time_step=1, radiation_radius=10, weak_fire_threshold=0.5, flame_height=3, flame_angle=np.pi/3):

        if terrain_sizes is None or hotspot_areas is None or num_ign_points is None or duration is None:
            raise ValueError(">>> Oops! 'WildFire' environment cannot be initialized without any parameters.")

        self.terrain_sizes = [int(terrain_sizes[0]), int(terrain_sizes[1])]  # sizes of the terrain
        self.initial_terrain_map = np.zeros(shape=self.terrain_sizes)  # initializing the terrain
        self.hotspot_areas = hotspot_areas  # format:: [[x_min, x_max, y_min, y_max]]
        self.num_ign_points = num_ign_points  # number of fire-spots in each area
        self.duration = duration  # total runtime steps
        self.time_step = time_step  # time step
        self.radiation_radius = radiation_radius  # the maximum effective thermal radiation range (default:: 10 [m])
        self.weak_fire_threshold = weak_fire_threshold  # fire intensity threshold [W/m], where a fire is burnt out if below this

        # max vertical extension of the flame [m] (ignore the occasional flashes which rise above the general level of fire)
        self.flame_height = flame_height

        # flame tilt angle (angle between flame heading and a vertical axis going through the center of fire spot on ground) [rad]
        self.flame_angle = flame_angle

    # initializing hotspots
    def hotspot_init(self):
        """
        This function generates the initial hotspot areas

        :return: ignition points across the entire map
        """

        ign_points_all = np.zeros(shape=[0, 2])
        for hotspot in self.hotspot_areas:
            x_min, x_max = hotspot[0], hotspot[1]
            y_min, y_max = hotspot[2], hotspot[3]
            ign_points_x = np.random.randint(low=x_min, high=x_max, size=(self.num_ign_points, 1))
            ign_points_y = np.random.randint(low=y_min, high=y_max, size=(self.num_ign_points, 1))
            ign_points_this_area = np.concatenate([ign_points_x, ign_points_y], axis=1)
            ign_points_all = np.concatenate([ign_points_all, ign_points_this_area], axis=0)

        # computing the fire intensity
        counter = 0
        ign_points = np.zeros(shape=[ign_points_all.shape[0], 3])
        for point in ign_points_all:
            heat_source_diff = np.tile(point, (ign_points_all.shape[0], 1)) - ign_points_all
            heat_source_dists = np.sqrt((heat_source_diff[:, 0] ** 2) + (heat_source_diff[:, 1] ** 2))
            idx = np.where(heat_source_dists <= self.radiation_radius)[0]
            fire_intensity = self.fire_intensity(point, ign_points_all[idx.tolist(), :].tolist())
            ign_points[counter] = np.array([point[0], point[1], fire_intensity])

            counter += 1

        return ign_points

    # fire intensity calculation
    def fire_intensity(self, current_fire_spot=None, heat_source_spots=None, deviation_min=9, deviation_max=11):
        """
        this function performs the fire intensity calculation according to [1] for each new fire front.

        [1] http://www.cfs.nrcan.gc.ca/bookstore_pdfs/21396.pdf

        :param current_fire_spot: the fire location for which the intensity is going to be computed
        :param heat_source_spots: the fire source location close to the new fire spot
        :param deviation_min: min of the radiation range
        :param deviation_max: max of the radiation range
        :return: fire intensity at the new fire spot location [W/m]
        """

        if current_fire_spot is None or heat_source_spots is None:
            raise ValueError(">>> Oops! Current fire location and included vicinity are required.")

        x = current_fire_spot[0]
        y = current_fire_spot[1]

        x_dev = np.random.randint(low=deviation_min, high=deviation_max, size=(1, 1))[0][0] + np.random.normal()
        y_dev = np.random.randint(low=deviation_min, high=deviation_max, size=(1, 1))[0][0] + np.random.normal()

        if np.cos(self.flame_angle) == 0:
            intensity_coeff = (259.833 * (self.flame_height ** 2.174)) / 1e3  # 1e3 is to change the unit to [MW/m]
        else:
            intensity_coeff = (259.833 * ((self.flame_height/np.cos(self.flame_angle)) ** 2.174)) / 1e3  # 1e3 is to change the unit to [MW/m]

        intensity = []
        for spot in heat_source_spots:
            x_f = spot[0]
            y_f = spot[1]
            intensity.append((1 / (2 * np.pi * x_dev * y_dev)) *
                             np.exp(-0.5 * ((((x - x_f) ** 2) / x_dev ** 2) + (((y - y_f) ** 2) / y_dev ** 2))))
        accumulated_intensity = sum(intensity) * intensity_coeff

        return 1e3 * accumulated_intensity

    # calculating the flame length as a function of fire intensity
    @staticmethod
    def fire_flame_length(accumulated_intensity=None):
        """
        this function computes the fire length as a function of fire intensity according to [1].

        [1] http://www.cfs.nrcan.gc.ca/bookstore_pdfs/21396.pdf

        :param accumulated_intensity: fire intensity at the current fire spot location [kW/m]
        :return: flame length at the current fire spot location
        """

        if accumulated_intensity is None:
            raise ValueError(">>> oops! The intensity at current fire location is required.")

        flame_length = 0.0775 * (accumulated_intensity ** 0.46)

        return flame_length

    # initialize the geo-physical information
    def geo_phys_info_init(self, max_fuel_coeff=7, avg_wind_speed=5, avg_wind_direction=np.pi/8):
        """
        This function generates a set of Geo-Physical information based on user defined ranges for each parameter

        :param max_fuel_coeff: maximum fuel coefficient based on vegetation type of the terrain
        :param avg_wind_speed: average effective mid-flame wind speed
        :param avg_wind_direction: wind azimuth
        :return: a dictionary containing geo-physical information
        """

        min_fuel_coeff = 1e-15
        fuel_rng = max_fuel_coeff - min_fuel_coeff
        spread_rate = fuel_rng*np.random.rand(self.terrain_sizes[0], self.terrain_sizes[1])+min_fuel_coeff
        wind_speed = np.random.normal(avg_wind_speed, 2, size=(self.terrain_sizes[0], 1))
        wind_direction = np.random.normal(avg_wind_direction, 2, size=(self.terrain_sizes[0], 1))

        geo_phys_info = {'spread_rate': spread_rate,
                         'wind_speed': wind_speed,
                         'wind_direction': wind_direction}

        return geo_phys_info

    # wildfire propagation
    def fire_propagation(self, world_Size, ign_points_all=None, geo_phys_info=None,
                         previous_terrain_map=None, pruned_List = None):
        """
        This function implements the simplified FARSITE wildfire propagation mathematical model

        :param ign_points_all: array including all fire-fronts and their intensities across entire terrain [output of hotspot_init()]
        :param geo_phys_info: a dictionary including geo-physical information [output of geo_phys_info_inti()]
        :param previous_terrain_map: the terrain including all fire-fronts and their intensities as an array
        :return: new fire front points and their corresponding geo-physical information
        """

        if ign_points_all is None or geo_phys_info is None or previous_terrain_map is None or pruned_List is None:
            raise ValueError(">>> Oops! Fire propagation function needs ALL of its inputs to operate!")

        current_geo_phys_info = np.zeros(shape=[ign_points_all.shape[0], 3])
        new_fire_front = np.zeros(shape=[ign_points_all.shape[0], 3])
        counter = 0
        for point in ign_points_all:
            # extracting the data
            x, y = point[0], point[1]
            # Ensure that all the fire spots to be displayed must be within the window scope
            if ((x <= (world_Size - 1)) and (y <= (world_Size - 1)) and (x > 0) and (y > 0)):
                spread_rate = geo_phys_info['spread_rate']
                wind_speed = geo_phys_info['wind_speed']
                wind_direction = geo_phys_info['wind_direction']

                # extracting the required information
                R = spread_rate[int(round(x)), int(round(y))]
                U = wind_speed[np.random.randint(low=0, high=self.terrain_sizes[0])][0]
                Theta = wind_direction[np.random.randint(low=0, high=self.terrain_sizes[0])][0]
                current_geo_phys_info[counter] = np.array([R, U, Theta])  # storing GP information

                # Simplified FARSITE
                LB = 0.936 * np.exp(0.2566 * U) + 0.461 * np.exp(-0.1548 * U) - 0.397
                HB = (LB + np.sqrt(np.absolute(np.power(LB, 2) - 1))) / (LB - np.sqrt(np.absolute(np.power(LB, 2) - 1)))
                C = 0.5 * (R - (R / HB))

                x_diff = C * np.sin(Theta)
                y_diff = C * np.cos(Theta)

                # updating the fire location
                if [int(x), int(y)] not in pruned_List:
                    x_new = x + x_diff * self.time_step
                    y_new = y + y_diff * self.time_step
                else:
                    x_new = x
                    y_new = y

                # computing the fire intensity
                heat_source_diff1 = np.tile(point, (ign_points_all.shape[0], 1)) - ign_points_all
                heat_source_dists1 = np.sqrt((heat_source_diff1[:, 0] ** 2) + (heat_source_diff1[:, 1] ** 2))
                idx1 = np.where(heat_source_dists1 <= self.radiation_radius)[0]
                fire_intensity1 = self.fire_intensity(point, ign_points_all[idx1.tolist(), :].tolist())

                heat_source_diff2 = np.tile(point, (previous_terrain_map.shape[0], 1)) - previous_terrain_map
                heat_source_dists2 = np.sqrt((heat_source_diff2[:, 0] ** 2) + (heat_source_diff2[:, 1] ** 2))
                idx2 = np.where(heat_source_dists2 <= self.radiation_radius)[0]
                fire_intensity2 = self.fire_intensity(point, previous_terrain_map[idx2.tolist(), :].tolist())

                fire_intensity = fire_intensity1 + fire_intensity2

                # storing new fire-front locations and intensity
                new_fire_front[counter] = np.array([x_new, y_new, fire_intensity])

                counter += 1

        return new_fire_front, current_geo_phys_info

    # dynamic fire decay
    def fire_decay(self, terrain_map=None, time_vector=None, geo_phys_info=None, decay_rate=0.01):
        """
        this function performs the dynamic fire decay over time due to fuel exhaustion.

        :param terrain_map: the terrain including all fire-fronts and their intensities as an array
        :param time_vector: a vector containing how long has passed after the ignition of each point until now
        :param geo_phys_info: a dictionary including geo-physical information [output of geo_phys_info_inti()]
        :param decay_rate: fuel exhaustion rate (greater means faster exhaustion)
        :return: the new fire map with updated intensities and the time vector
        """

        if terrain_map is None or geo_phys_info is None or time_vector is None:
            raise ValueError(">>> Oops! The fire decay function requires ALL its inputs (except for 'decay_rate=0.01' as default) to operate.")

        spread_rate = geo_phys_info['spread_rate']

        step_vector = self.time_step * np.ones(terrain_map.shape[0])
        updated_time_vector = time_vector + step_vector

        # updating the intensities
        counter = 0
        updated_terrain_map = np.zeros(shape=[terrain_map.shape[0], 3])
        for spot in terrain_map:
            x = spot[0]
            y = spot[1]
            intensity = spot[2]
            R = spread_rate[int(round(x)), int(round(y))]

            I_new = intensity * np.exp(-decay_rate * updated_time_vector[counter]/R)

            updated_terrain_map[counter] = np.array([x, y, I_new])

            counter += 1

        # pruning dead fire spots from the fire map
        updated_terrain_map, updated_time_vector, burnt_out_fires_new = self.pruning_fire_map(
            updated_terrain_map=updated_terrain_map, updated_time_vector=updated_time_vector)

        return updated_terrain_map, updated_time_vector, burnt_out_fires_new

    # pruning dead fire spots from the fire map
    def pruning_fire_map(self, updated_terrain_map=None, updated_time_vector=None):
        """
        this functions performs the fire map pruning and puts out fire spots that have "weak enough" intensity

        :param updated_terrain_map: most recent terrain map updated after the intensity measures
        :param updated_time_vector: most recent time vector
        :return: new terrain map, time vector and pruned fire spots
        """

        burnt_out_fires_idx = np.where(updated_terrain_map[:, 2] < self.weak_fire_threshold)
        burnt_out_fires_new = updated_terrain_map[burnt_out_fires_idx]
        updated_terrain_map = np.delete(updated_terrain_map, burnt_out_fires_idx, 0)
        updated_time_vector = np.delete(updated_time_vector, burnt_out_fires_idx)

        return updated_terrain_map, updated_time_vector, burnt_out_fires_new
