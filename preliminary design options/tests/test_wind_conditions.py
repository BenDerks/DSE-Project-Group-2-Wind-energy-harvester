import unittest

from inputs.wind_conditions import Wind_scalar


class TestWindScalar(unittest.TestCase):
    def test_wind_speed_at_reference_height(self):
        # input values from data for 10 m and 100 m hard coded for now
        wind_speed_10m = 5.0  # m/s
        actual_wind_speed = 7.92  # m/s hardcoded for 100 m, imported from data later
        altitude = 100  # m
        calculated_wind_speed = Wind_scalar(altitude, wind_speed_10m)
        self.assertAlmostEqual(calculated_wind_speed, actual_wind_speed, places=2)

if __name__ == "__main__":
    unittest.main()
