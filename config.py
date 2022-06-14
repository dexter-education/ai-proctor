def get_config():

    Dict = {
        'face': {
            'count': 1, 
            'min_continous_frames': [10, 20],
            'frame_counter': [0, 0],
            'required_counts': [1, 1],
            'level': ['yellow', 'red'],
            'message': ['face count violation', 'face count violation'],
            'confidence': 0.4
            },
        'person': {
            'count': 1, 
            'min_continous_frames': [10, 20],
            'frame_counter': [0, 0],
            'required_counts': [1, 1],
            'level': ['yellow', 'red'],
            'message': ['person count violation', 'person count violation'],
            'confidence': 0.25
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
            'message': ['face not centered'],
            'center_ratio': 0.25
        },
        'mouth open': {
            'count': 0,
            'min_continous_frames': [10],
            'frame_counter': [0],
            'required_counts': [0],
            'level': ['yellow'],
            'message': ['mouth open'],
            'confidence': 0.5,
            'pixels_required': 30
        },
        'mouth hidden': {
            'count': 0,
            'min_continous_frames': [10],
            'frame_counter': [0],
            'required_counts': [0],
            'level': ['yellow'],
            'message': ['mouth hidden'],
            'pixels_required': 20
        },
        'yaw': {
            'count': 0,
            'min_continous_frames': [10],
            'frame_counter': [0],
            'required_counts': [0],
            'level': ['yellow'],
            'message': ['yaw violation'],
            'confidence': 0.5,
            'angle': 30
        },
        'pitch': {
            'count': 0,
            'min_continous_frames': [10],
            'frame_counter': [0],
            'required_counts': [0],
            'level': ['yellow'],
            'message': ['pitch violation'],
            'angle': 20
        },
        'roll': {
            'count': 0,
            'min_continous_frames': [10],
            'frame_counter': [0],
            'required_counts': [0],
            'level': ['yellow'],
            'message': ['roll violation'],
            'angle': 20
        },
    }
    return Dict