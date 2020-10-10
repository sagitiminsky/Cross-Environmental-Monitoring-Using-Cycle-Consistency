class PowerLaw:
    def __init__(self,chosen_power_law):
        if chosen_power_law=='Basic':

            self.b=1
            self.L=1
            self.a=1

            self.basic_attinuation_to_rain=self.basic_attinuation_to_rain
            self.basic_rain_to_attinuation=self.basic_rain_to_attinuation
        else:
            raise NotImplementedError('power law was not implemented {}'.format(chosen_power_law))


    def basic_attinuation_to_rain(self,A):
        '''
        :param A: a single dim. ndarray that represent the link attenuation in db
        :return: a single dim. ndarray that represent the rain amount in mm/h
        '''

        return (A/(self.L*self.a))**self.b

    def basic_rain_to_attinuation(self,R):
        '''
        :param R: a single dim. ndarray that represent the rain amount in mm/h
        :return: a single dim. ndarray that represent the link attenuation in db
        '''
        return self.a*(R)**self.b*self.L
