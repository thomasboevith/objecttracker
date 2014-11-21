import unittest
from objecttracker import track
from objecttracker import trackpoint


class TestTracks(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_create_new_track(self):
        t = track.Track()
        self.assertIsNotNone(t)

    def test_number_of_trackpoints(self):
        t = track.Track()
        t.add_trackpoint(trackpoint.Trackpoint(10, 20))
        self.assertTrue(t.number_of_trackpoints() == 1)

    def test_length_of_track(self):
        trackpoints = [(10, 30), (10, 40), (20, 40), (20, 30)]
        t = track.Track()
        for row, col in trackpoints:
            t.add_trackpoint(trackpoint.Trackpoint(row, col))
        self.assertEqual(round(t.length(), 5), 30)

    def test_parent_of_split_track(self):
        trackpoints_1 = [(10, 25), (15, 25), (20, 30),
                         (25, 30), (30, 40), (35, 35)]
        trackpoints_2 = [(11, 21), (15, 22), (16, 28),
                         (20, 39), (26, 60), (78, 92)]
        t = track.Track()
        for row, col in trackpoints_1:
            t.add_trackpoint(trackpoint.Trackpoint(row, col))
        t_child_1, t_child_2 = t.split()
        for row, col in trackpoints_2:
            t_child_1.add_trackpoint(trackpoint.Trackpoint(row, col))
        for row, col in trackpoints_1:
            t_child_2.add_trackpoint(trackpoint.Trackpoint(row, col))

        self.assertIs(t_child_2.parent, t)
        self.assertIs(t_child_1.parent, t_child_2.parent)
        self.assertIsNot(t_child_2.parent, t_child_1)

    def test_length_of_parent_track(self):
        trackpoints_1 = [(10, 25), (15, 25), (20, 30),
                         (25, 30), (30, 40), (35, 35)]
        trackpoints_2 = [(11, 21), (15, 22), (16, 28),
                         (20, 39), (26, 60), (78, 92)]
        t = track.Track()
        for row, col in trackpoints_1:
            t.add_trackpoint(trackpoint.Trackpoint(row, col))

        t_child_1, t_child_2 = t.split()
        for row, col in trackpoints_2:
            t_child_1.add_trackpoint(trackpoint.Trackpoint(row, col))

        for row, col in trackpoints_1:
            t_child_2.add_trackpoint(trackpoint.Trackpoint(row, col))

        self.assertEqual(round(t.length(), 5), 35.32248)
        self.assertEqual(round(t_child_1.length(), 5), 104.80825)
        self.assertEqual(t_child_2.length(), t.length())
        self.assertNotEqual(t_child_1.length(), t.length())
        self.assertNotEqual(t_child_2.length(include_parents=True),
                            t.length())
        self.assertNotEqual(t_child_2.length(True),
                            t.length())
        self.assertNotEqual(t_child_1.length(include_parents=True), t.length())
        self.assertNotEqual(t_child_1.length(include_parents=True),
                            t_child_1.length())
        self.assertEqual(round(t_child_1.length(include_parents=True), 5),
                         167.91561)
        self.assertEqual(round(t_child_2.length(include_parents=True), 5),
                         97.57078)
        self.assertIs(t_child_1.parent, t)

    def test_save_track(self):
        trackpoints_1 = [(10, 25), (15, 25), (20, 30),
                         (25, 30), (30, 40), (35, 35)]
        trackpoints_2 = [(11, 21), (15, 22), (16, 28),
                         (20, 39), (26, 60), (78, 92)]
        t = track.Track()
        for row, col in trackpoints_1:
            t.add_trackpoint(trackpoint.Trackpoint(row, col))
        t_child_1, t_child_2 = t.split()
        for row, col in trackpoints_2:
            t_child_1.add_trackpoint(trackpoint.Trackpoint(row, col))
        for row, col in trackpoints_1:
            t_child_2.add_trackpoint(trackpoint.Trackpoint(row, col))

        t.save("test/output/t.track")
        t_child_1.save("test/output/t_child_1.track")
        t_child_2.save("test/output/t_child_2.track")
        t_child_1.save("test/output/t_child_1_parent.track",
                       include_parents=True)
        t_child_2.save("test/output/t_child_2_parent.track",
                       include_parents=True)

        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
