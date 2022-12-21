class Savable:
    """
    This interface regulates the method to save parameters by get_data, and set parameters by set_data.
    """
    def set_data(self, data_iter):
        pass

    def get_data(self):
        return []
