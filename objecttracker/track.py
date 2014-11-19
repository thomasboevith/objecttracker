class Track:
    def __init__(self):
        self.parent = None
        self.trackpoints = []
        self.id = None
        self.parent_id = None
        self.children_ids = []

    def add_trackpoint(self, trackpoint):
        self.trackpoints.append(trackpoint)
    
    def set_parent(self, parent):
        self.parent = parent

    def length(self):
        sum = 0
        raise NotImplementedError("Implement the geographic length of the track in pixels. Do it! Do it!")

    def number_of_trackpoints(self):
        return len(self.trackpoints)
        

    def save(self, filename):
        with open(filename, 'w') as fp:
            for trackpoint in self.trackpoints:
                fp.write(trackpoint)
        
        
        
