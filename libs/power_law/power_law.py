import pandas as pd
import numpy as np
import config


class PowerLaw:
    def __init__(self, chosen_power_law, frequency, L):
        '''

        :param chosen_power_law: a mux which decides what power law is chosen
        :param frequency: the antena freq
        :param L: the signal path length
        '''
        self.L = L

        if chosen_power_law == 'Basic':
            self.db_path =config.basic_db_path
            self.a, self.b = self.calculate_basic_power_law_constants(frequency)

            self.basic_attinuation_to_rain = self.basic_attinuation_to_rain
            self.basic_rain_to_attinuation = self.basic_rain_to_attinuation
        else:
            raise NotImplementedError('power law was not implemented {}'.format(chosen_power_law))

    def calculate_basic_power_law_constants(self, frequency):
        df = pd.read_csv(self.db_path)
        row = df[df['frequency[Ghz]']==frequency]

        if len(row)!=0:



            # taken from https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.838-2-200304-S!!PDF-E.pdf
            # k_h k_v alpha_h alpha_v

            theta = 90
            tau = 45

            k_h = row.k_h
            k_v = row.k_v
            a_h = row.a_h
            a_v = row.a_v

            b = (k_h + k_v + (k_h - k_v) * np.cos(theta) ** 2 * np.cos(2 * tau)) / 2
            a = (k_h * a_v + k_v * a_v + (k_h * a_h - k_v * a_v) * np.cos(theta) ** 2 * np.cos(2 * tau)) / (2 * b)

            return a, b
        else:
            raise ValueError('provided frequency does not exist in table: {}'.format(frequency))

    def basic_attinuation_to_rain(self, A):
        '''
        :param A: a single dim. ndarray that represent the link attenuation in db/km
        :return: a single dim. ndarray that represent the rain amount in mm/h
        '''

        return (A / (self.L * self.a)) ** self.b

    def basic_rain_to_attinuation(self, R):
        '''
        :param R: a single dim. ndarray that represent the rain amount in mm/h
        :return: a single dim. ndarray that represent the link attenuation in db
        '''
        return self.a * (R) ** self.b * self.L


if __name__=="__main__":
    PowerLaw(chosen_power_law='Basic', frequency=400, L=10)