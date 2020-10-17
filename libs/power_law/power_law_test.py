import unittest
from libs.power_law.power_law import PowerLaw
import config
import pandas as pd


class PowerLawTest(unittest.TestCase):
    def setUp(self):
        self.polarizations = ['horizontal','vertical']


    def test_create_basic_power_law(self):
        PowerLaw(chosen_power_law='Basic', frequency=400,polarization='vertical', L=10)
        PowerLaw(chosen_power_law='Basic', frequency=400, polarization='horizontal', L=10)

        # with self.assertRaises(ValueError):
        PowerLaw(chosen_power_law='Basic', frequency=400, polarization='cyrcular', L=10)
        PowerLaw(chosen_power_law='Basic',frequency=101,polarization='vertical',L=10)


    def test_basic_all_frequecies(self):
        df = pd.read_csv(config.basic_db_path)

        for pol in self.polarizations:
            for frequency in df['frequency[Ghz]']:
                self.assertTrue(PowerLaw(chosen_power_law='Basic', frequency=frequency,polarization=pol, L=10))




if __name__ == '__main__':
    unittest.main()
