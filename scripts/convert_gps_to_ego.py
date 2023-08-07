import os

import numpy
import json
from tqdm import tqdm

dataset_path = "/home/dennis/Media_HDD/ETH/Robot_Learning/preprocessed_diffusedrive_dataset/weather-0/data"
# dataset_path = "/home/dennis/Media_HDD/ETH/Robot_Learning/DiffuseDrive/test"
n_past_waypoints = 10
n_future_waypoints = 10

def main():
    for scenario in tqdm(os.listdir(dataset_path)):
        if not os.path.exists(os.path.join(dataset_path, scenario, 'ego_data')):
            os.makedirs(os.path.join(dataset_path, scenario, 'ego_data'))
        measurement_path = os.path.join(dataset_path, scenario, 'measurements')
        for idx, measurement in enumerate(sorted(os.listdir(measurement_path))):
            with open(os.path.join(measurement_path, measurement), 'r') as f:
                current_json = json.load(f)

            current_waypoints_gps = numpy.array(
                [current_json["gps_x"], current_json["gps_y"], current_json["theta"]])
            past_waypoints_gps = get_past_waypoint_gps(idx, measurement_path)
            past_waypoints_ego = numpy.array(convert_gps_to_ego(past_waypoints_gps, current_waypoints_gps))
            future_waypoints_gps = get_future_waypoint_gps(idx, measurement_path)
            future_waypoints_ego = numpy.array(convert_gps_to_ego(future_waypoints_gps, current_waypoints_gps))
            past_commands = get_past_cmd(idx, measurement_path)
            future_commands = get_future_cmd(idx, measurement_path)

            data = {}
            if len(past_waypoints_ego) == 0:
                data["ego_past_waypoints_x"] = []
                data["ego_past_waypoints_y"] = []
                data["ego_past_waypoints_theta"] = []
                data["past_commands"] = []
            else:
                past_waypoints_ego = past_waypoints_ego.reshape((len(past_waypoints_ego),3))
                data["ego_past_waypoints_x"] = past_waypoints_ego[:,0].tolist()
                data["ego_past_waypoints_y"] = past_waypoints_ego[:,1].tolist()
                data["ego_past_waypoints_theta"] = past_waypoints_ego[:,2].tolist()
                data["past_commands"] = [i for i in past_commands if i is not None]

            data["current_command"] = current_json["command"]

            if len(future_waypoints_ego) == 0:
                data["ego_future_waypoints_x"] = []
                data["ego_future_waypoints_y"] = []
                data["ego_future_waypoints_theta"] = []
                data["future_commands"] = []
            else:
                future_waypoints_ego = future_waypoints_ego.reshape((len(future_waypoints_ego),3))
                data["ego_future_waypoints_x"] = future_waypoints_ego[:,0].tolist()
                data["ego_future_waypoints_y"] = future_waypoints_ego[:,1].tolist()
                data["ego_future_waypoints_theta"] = future_waypoints_ego[:,2].tolist()
                data["future_commands"] = [i for i in future_commands if i is not None]

            with open(os.path.join(dataset_path, scenario, f'ego_data/{str(idx).zfill(4)}.json'), 'w') as f:
                json.dump(data, f, indent=4)


def convert_gps_to_ego(gps_waypoints, current_gps_waypoint):
    ego_waypoint = []
    for t in range(len(gps_waypoints)):
        if not isinstance(gps_waypoints[t], numpy.ndarray):
            break

        ego_waypoint.append(transform_2d_points(
            numpy.zeros((1, 3)),
            numpy.pi / 2 - gps_waypoints[t][2],
            -gps_waypoints[t][0],
            -gps_waypoints[t][1],
            numpy.pi / 2 - current_gps_waypoint[2],
            -current_gps_waypoint[0],
            -current_gps_waypoint[1],
        ))
    return ego_waypoint


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1
    xy1_front = xy1.copy()
    xy1_front[:, 0] += 1.0

    c, s = numpy.cos(r1), numpy.sin(r1)
    r1_to_world = numpy.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = numpy.asarray(r1_to_world @ xy1.T)
    world_front = numpy.asarray(r1_to_world @ xy1_front.T)

    c, s = numpy.cos(r2), numpy.sin(r2)
    r2_to_world = numpy.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = numpy.linalg.inv(r2_to_world)

    out = numpy.asarray(world_to_r2 @ world).T
    out_front = numpy.asarray(world_to_r2 @ world_front).T
    # reset z-coordinate
    out[:, 2] = numpy.arctan2(out_front[:, 1] - out[:, 1], out_front[:, 0] - out[:, 0])

    return out


def get_past_waypoint_gps(file_id, folder_path):
    past_waypoint_gps = [None] * n_past_waypoints
    for t in range(n_past_waypoints):
        previous_file_id = file_id - t - 1
        previous_file_path_measurements = os.path.join(folder_path, str(previous_file_id).zfill(4) + ".json")
        if not os.path.exists(previous_file_path_measurements):
            continue

        previous_json = json.load(open(previous_file_path_measurements))
        past_waypoint_gps[t] = numpy.array([previous_json["gps_x"], previous_json["gps_y"], previous_json["theta"]])

    return past_waypoint_gps

def get_future_waypoint_gps(file_id, folder_path):
    future_waypoint_gps = [None] * n_future_waypoints
    for t in range(n_future_waypoints):
        next_file_id = file_id + t + 1
        next_file_path_measurements = os.path.join(folder_path, str(next_file_id).zfill(4) + ".json")

        if not os.path.exists(next_file_path_measurements):
            continue

        next_json = json.load(open(next_file_path_measurements))
        future_waypoint_gps[t] = numpy.array([next_json["gps_x"], next_json["gps_y"], next_json["theta"]])

    return future_waypoint_gps

def get_past_cmd(file_id, folder_path):
    past_cmd = [None] * n_past_waypoints
    for t in range(n_past_waypoints):
        prev_file_id = file_id - t - 1
        prev_file_path_measurements = os.path.join(folder_path, str(prev_file_id).zfill(4) + ".json")

        if not os.path.exists(prev_file_path_measurements):
            continue

        prev_json = json.load(open(prev_file_path_measurements))
        past_cmd[t] = prev_json["command"]
    return past_cmd

def get_future_cmd(file_id, folder_path):
    future_cmd = [None] * n_future_waypoints
    for t in range(n_future_waypoints):
        next_file_id = file_id + t + 1
        next_file_path_measurements = os.path.join(folder_path, str(next_file_id).zfill(4) + ".json")

        if not os.path.exists(next_file_path_measurements):
            continue
        next_json = json.load(open(next_file_path_measurements))
        future_cmd[t] = next_json["command"]

    return future_cmd


main()
