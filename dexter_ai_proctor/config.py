def get_config():

    Dict = {
        'face': {
            'count': 1, 
            'min_continous_frames': [10, 20],
            'frame_counter': [0, 0],
            'required_counts': [1, 1],
            'level': ['yellow', 'red'],
            'message': ['face count violation', 'face count violation']
            },
        'person': {
            'count': 1, 
            'min_continous_frames': [10, 20],
            'frame_counter': [0, 0],
            'required_counts': [1, 1],
            'level': ['yellow', 'red'],
            'message': ['person count violation', 'person count violation']
            },
        'mobile phone': {
            'count': 0,
            'min_continous_frames': [10, 20],
            'frame_counter': [0, 0],
            'required_counts': [0, 0],
            'level': ['yellow', 'red'],
            'message': ['mobile phone violation', 'mobile phone violation']
        },
        'laptop': {
            'count': 0,
            'min_continous_frames': [10, 20],
            'frame_counter': [0, 0],
            'required_counts': [0, 0],
            'level': ['yellow', 'red'],
            'message': ['laptop violation', 'laptop violation']
        },
        'face center': {
            'count': 0,
            'min_continous_frames': [10],
            'frame_counter': [0],
            'required_counts': [0],
            'level': ['yellow'],
            'message': ['face not centered']
        },
        'mouth open': {
            'count': 0,
            'min_continous_frames': [10],
            'frame_counter': [0],
            'required_counts': [0],
            'level': ['yellow'],
            'message': ['mouth open']
        },
        'mouth hidden': {
            'count': 0,
            'min_continous_frames': [10],
            'frame_counter': [0],
            'required_counts': [0],
            'level': ['yellow'],
            'message': ['mouth hidden']
        },
        'yaw': {
            'count': 0,
            'min_continous_frames': [10],
            'frame_counter': [0],
            'required_counts': [0],
            'level': ['yellow'],
            'message': ['yaw violation']
        },
        'pitch': {
            'count': 0,
            'min_continous_frames': [10],
            'frame_counter': [0],
            'required_counts': [0],
            'level': ['yellow'],
            'message': ['pitch violation']
        },
        'roll': {
            'count': 0,
            'min_continous_frames': [10],
            'frame_counter': [0],
            'required_counts': [0],
            'level': ['yellow'],
            'message': ['roll violation']
        },
    }
    return Dict