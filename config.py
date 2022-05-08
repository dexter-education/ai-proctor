def get_config():

    Dict = {
        'face': {
            'count': 0, 
            'min_continous_frames': [10, 20],
            'frame_counter': [0, 0],
            'required_counts': [1, 1],
            'level': ['yellow', 'red'],
            'message': ['face count violation', 'face count violation']
            },
        'person': {
            'count': 0, 
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
        }
    }
    return Dict