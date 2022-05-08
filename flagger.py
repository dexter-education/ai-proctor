class flagger:
    """Class to count and raise flags
    """

    def __init__(self, config_dict):
        self.config_dict = config_dict
        
    def flag(self, level, frame_num, message):
        """Function to raise the flag

        Args:
            level (string): Level of the flag, i.e. red or yellow
            frame_num (int): Frame number at which violation starts
            message (string): Message to be raised
        """
        print(level, frame_num, message)

    def add_count(self, Dict):
        """Function to add the count of the current frame

        Args:
            Dict (dictionary): Dictionary containing the count with the appropriate keys
        """
        for key in Dict.keys():
            self.config_dict[key]['count'] = Dict[key]

    def check_counts(self, frame_num):
        """Check the counts of the current frame and raise flags if necessary

        Args:
            frame_num (int): current frame number
        """

        for key, value in self.config_dict.items():
            for i in range(len(value['min_continous_frames'])):

                if value['frame_counter'][i] >= value['min_continous_frames'][i]:
                    self.flag(value['level'][i], frame_num - value['min_continous_frames'][i], value['message'][i])
                    self.config_dict[key]['frame_counter'][i] = 0

                if value['count'] != value['required_counts'][i]:
                    value['frame_counter'][i] += 1
                else:
                    self.config_dict[key]['frame_counter'][i] = 0