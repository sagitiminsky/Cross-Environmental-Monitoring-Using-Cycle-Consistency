import pandas as pd
import numpy as np
import config


class PowerLaw:
    def __init__(self, frequency, polarization, L,chosen_power_law='Basic'):
        '''

        :param chosen_power_law: a mux which decides what power law is chosen
        :param frequency: the antena freq
        :param L: the signal path length
        '''
        self.L = float(L)

        if chosen_power_law == 'Basic':
            self.db_path = config.basic_db_path
            self.a, self.b = self.calculate_basic_power_law_constants(frequency,polarization)
        else:
            raise NotImplementedError('power law was not implemented {}'.format(chosen_power_law))

    def calculate_basic_power_law_constants(self, frequency,polarization):
        df = pd.read_csv(self.db_path)
        row = df[df['frequency[Ghz]'] == frequency]

        if len(row) != 0:

            # taken from https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.838-3-200503-I!!PDF-E.pdf
            # k_h k_v alpha_h alpha_v

            k_h = float(row.k_h)
            k_v = float(row.k_v)
            a_h = float(row.a_h)
            a_v = float(row.a_v)

            if polarization=='Horizontal':
                return k_h, a_h

            elif polarization=='Vertical':
                return k_v, a_v

            else:
                raise ValueError('polarization type is not supported: {}'.format(polarization))


        else:
            raise ValueError('provided frequency does not exist in table: {}'.format(frequency))

    def basic_attinuation_to_rain_single(self, A):
        '''
        :param A: a single dim. ndarray that represent the link attenuation in db/km
        :return: a single dim. ndarray that represent the rain amount in mm/h
        '''

        if type(A) is not float:
            raise ValueError('Wrong type:{} needs to be float'.format(type(A)))


        """
        Errors always exist because of quantization errors. 
        We have no way around it. Indeed, we have a way of knowing that there is never a negative attunuation. 
        Therefore, the best we can do is assume there is an error, and reset the attenuation.
        By the way - it also means that there is a slight bias to our estimates. 
        Because - an error that bounces the attenuation to a negative area is eliminated 
        (or at least reduces the noise that bounces the attenuation to a negative area, because we cut it to zero),
        but positive errors are not offset at all. But - this is not the most critical, because in medium and heavy rains,
        this phenomenon is negligible, because the attenuation in the first place will never be close to Zero Lebel,
         because the rain entails a strong attenuation. 
        """
        if A<0:
            A=0

        return (A / (self.L * self.a)) ** self.b

    def basic_attinuation_to_rain_multiple(self,A_array):
        R_array=[]
        for A in A_array:
            R_array.append(self.basic_attinuation_to_rain_single(A))

    def basic_rain_to_attinuation(self, R):
        '''
        :param R: a single dim. ndarray that represent the rain amount in mm/h
        :return: a single dim. ndarray that represent the link attenuation in db
        '''

        if type(R) is not float:
            raise ValueError('Wrong type:{} needs to be float'.format(type(R)))

        return self.a * (R) ** self.b * self.L


if __name__ == "__main__":
    PowerLaw(chosen_power_law='Basic',frequency=101,polarization='vertical',L=10)
