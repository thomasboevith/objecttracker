class TrackPoint:
    def __init__(self, row, col, size, color, shape):
        self.row = row
        self.col = col
        self.size = size
        self.color = color
        self.shape = shape

    def length_to(self, tp):
        """
        Calculate the lengt between two track points.
        """
        assert(isinstance(tp), TrackPoint)
        return np.sqrt((self.row - tp.row) ** 2 + (self.col - tp.col) ** 2)

    def __str__(self):
        return "({row}, {col}) {size} {color} {shape}".format(row=self.row,
                                                              col=self.col,
                                                              size=self.size,
                                                              color=self.color,
                                                              shape=self.shape)
