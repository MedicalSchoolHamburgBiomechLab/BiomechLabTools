def get_apdm_location_map(hdf5_data: dict) -> dict:
    sensor_data = hdf5_data['Sensors']
    location_map = dict()
    for sensor_id, sensor_data in sensor_data.items():
        loc = sensor_data['Configuration']['Config Strings'][0][-1].decode('utf-8')
        location_map[sensor_id] = loc
    return location_map


def get_apdm_sensor_data_by_location(hdf5_data: dict, location: str) -> dict:
    location_map = get_apdm_location_map(hdf5_data)
    if location not in location_map.values():
        raise ValueError(f'Location {location} not found in the sensor data.')
    sensor_data = dict()
    for sensor_id, loc in location_map.items():
        if loc == location:
            sensor_data = hdf5_data['Sensors'][sensor_id]
    return sensor_data
