class Probe():
    @check_units
    def __init__(self, node, angle, tag=None):
        self.node = node
        self.angle = angle
        if tag is not None:
            self.tag = tag
            
    def probe_result(self):
        if self.tag is not None:
            return [(self.node, self.angle, self.tag)]
        else:
            return [(self.node, self.angle)]