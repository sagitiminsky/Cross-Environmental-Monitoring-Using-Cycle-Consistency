import unittest
from libs.power_law.power_law import PowerLaw
import config
import pandas as pd


class PowerLawTest(unittest.TestCase):
    def setUp(self):
        self.polarizations = ['Horizontal','Vertical']


    def test_create_basic_power_law(self):
        PowerLaw(chosen_power_law='Basic', frequency=400,polarization='Vertical', L=10)
        PowerLaw(chosen_power_law='Basic', frequency=400, polarization='Horizontal', L=10)

        with self.assertRaises(ValueError):
            PowerLaw(chosen_power_law='Basic', frequency=400, polarization='cyrcular', L=10)
            PowerLaw(chosen_power_law='Basic',frequency=101,polarization='Vertical',L=10)


    def test_basic_all_frequecies(self):
        df = pd.read_csv(config.basic_db_path)

        for pol in self.polarizations:
            for frequency in df['frequency[Ghz]']:
                self.assertTrue(PowerLaw(chosen_power_law='Basic', frequency=frequency,polarization=pol, L=10))

    def test_real_frequecy(self):
        real_freq=18.058
        self.assertTrue(PowerLaw(chosen_power_law='Basic', frequency=real_freq, polarization='Vertical', L=10))




if __name__ == '__main__':
    unittest.main()
