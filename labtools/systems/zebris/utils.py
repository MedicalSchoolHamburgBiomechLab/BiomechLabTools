import numpy as np

CHANNEL_INDEX_RAW = ['force', 'cop_x', 'cop_y', 'sync']
CHANNEL_INDEX_GAIT_MODE = ['belt_velocity',
                           'left_foot_force',
                           'left_foot_cop_x',
                           'left_foot_cop_y',
                           'left_foot_max_pressure',
                           'left_foot_cop_x_loc',
                           'left_foot_cop_y_loc',
                           'right_foot_force',
                           'right_foot_cop_x',
                           'right_foot_cop_y',
                           'right_foot_max_pressure',
                           'right_foot_cop_x_loc',
                           'right_foot_cop_y_loc',
                           'sync']


def get_force(c3d_data: dict, separate: bool = False):
    if c3d_data['analog'].shape[1] == 14:
        i1 = CHANNEL_INDEX_GAIT_MODE.index('right_foot_force')
        i2 = CHANNEL_INDEX_GAIT_MODE.index('left_foot_force')
        if separate:
            return (np.array(c3d_data['analog'][:, i1]),
                    np.array(c3d_data['analog'][:, i2]))
        return np.array(c3d_data['analog'][:, i1]) + np.array(c3d_data['analog'][:, i2])
    elif c3d_data['analog'].shape[1] == 4:
        i1 = CHANNEL_INDEX_RAW.index('force')
        return np.array(c3d_data['analog'][:, i1])
    else:
        raise ValueError('Unexpected number of analog channels found')


def get_cop(c3d_data: dict, local: bool = False):
    if c3d_data['analog'].shape[1] == 14:
        if local:
            raise warnings.warn("LOCAL COP NOT IMPLEMENTED")
        i1 = CHANNEL_INDEX_GAIT_MODE.index('left_foot_cop_x')
        i2 = CHANNEL_INDEX_GAIT_MODE.index('left_foot_cop_y')
        i3 = CHANNEL_INDEX_GAIT_MODE.index('right_foot_cop_x')
        i4 = CHANNEL_INDEX_GAIT_MODE.index('right_foot_cop_y')
        x = np.nan_to_num(c3d_data['analog'][:, i1], nan=0.0) + np.nan_to_num(c3d_data['analog'][:, i3], nan=0.0)
        x[x == 0.0] = np.nan
        y = np.nan_to_num(c3d_data['analog'][:, i2], nan=0.0) + np.nan_to_num(c3d_data['analog'][:, i4], nan=0.0)
        y[y == 0.0] = np.nan
        return {'x': x,
                'y': y}
    elif c3d_data['analog'].shape[1] == 4:
        i1 = CHANNEL_INDEX_RAW.index('cop_x')
        i2 = CHANNEL_INDEX_RAW.index('cop_y')
        return {'x': c3d_data['analog'][:, i1],
                'y': c3d_data['analog'][:, i2]}
    else:
        raise ValueError('Unexpected number of analog channels found')
