class SimpleAverage(object):

    def __init__(self):
        self.value = 0
        self.count = 0

    def update(self, val, counts=1):
        self.value += val
        self.count += counts

    def average(self):
        return self.value / self.count
