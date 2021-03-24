class ParkingIsFull(Exception):
    def __init__(self, message='Parking has no more available spaces'):
        self.message = message
        super.__init__(self.message)
