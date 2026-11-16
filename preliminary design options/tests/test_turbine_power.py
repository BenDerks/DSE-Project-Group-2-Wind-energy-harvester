import unittest

from analysis.performance import calculate_design_power
from models.turbine import required_rotor_area, turbine_power

# test the turbine power against known values from literature
class TestTurbinePower(unittest.TestCase):
    def test_power_for_areas(self):
        for area, expected_power in [(1, 216), (2, 432), (4, 864)]: # different input areas and expected powers
            with self.subTest(area=area):
                power = turbine_power(1.2, 10, area, 0.4, 0.9)
                self.assertAlmostEqual(power, expected_power) # check if the calculated power is almost equal to the expected power

    def test_required_area(self):
        area, diameter = required_rotor_area(30_000, 1.225, 12, 0.45, 0.9) #hard coded input to check against code

        self.assertAlmostEqual(area, 69.986842, places=5) # check if the calculated area is almost equal to the expected area
        self.assertAlmostEqual(diameter, 9.439810, places=5) # check if the calculated diameter is almost equal to the expected diameter

    def test_hawt(self):
        design = {"architecture": "HAWT", "rotor_diameter": 4}
        result = calculate_design_power(design, 1.2, 10)

        self.assertAlmostEqual(result["power_W"], 3053.63, places=2) # check if the calculated power is almost equal to the expected power
        self.assertAlmostEqual(result["power_kW"], 3.05363, places=5) # check if the calculated power is almost equal to the expected power
        self.assertEqual(design["rotor_diameter"], 4) # check if the rotor diameter is as expected

    def test_vawt(self):
        design = {
            "architecture": "VAWT",
            "rotor_diameter": 3,
            "rotor_height": 8,
        } # hardcoded input to check against code
        result = calculate_design_power(design, 1.2, 10)

        self.assertAlmostEqual(result["power_W"], 3888, places=2) # check if the calculated power is almost equal to the expected power
        self.assertAlmostEqual(result["power_kW"], 3.888, places=3) # check if the calculated power is almost equal to the expected power

if __name__ == "__main__":
    unittest.main()
