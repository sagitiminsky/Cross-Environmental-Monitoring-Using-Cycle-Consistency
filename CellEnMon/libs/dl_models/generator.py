import config
from keras.layers import Dense
from keras.models import Sequential
import wandb
from libs.dl_models.model_manager import Model_Manager
from libs.dl_models.extractor import Extractor


class Generator(Extractor):
    def __init__(self, version):

        super().__init__()
        self.generator, self.generated_features = self.pre_train_generator()

        self.model_manager = Model_Manager(config.generator_path, version)

        self.generator = self.pre_train_generator()

    def pre_train_generator(self):

        if not config.load_pre_trained_generator:
            run = wandb.init()
            wandb_config = run.config
            wandb_config.encoding_dim = config.generator_encoding_dim
            wandb_config.epochs = config.generator_epoches

            model = Sequential()
            model.add(Dense(self.n))
            model.add(Dense(config.generator_encoding_dim, activation='relu'))
            model.add(Dense(self.m, activation='sigmoid'))
            model.compile(optimizer='adam', loss='mse')

            model.fit(self.dme_data.T[:int(len(self.dme_data) * 0.8)], self.ims_data.T[:int(len(self.ims_data) * 0.8)],
                      epochs=config.generator_epoches,
                      validation_data=(
                          self.dme_data[int(len(self.dme_data) * 0.8):], self.ims_data[int(len(self.dme_data) * 0.8):]))

            self.model_manager.save(model)
        else:
            model = self.model_manager.load()

        return model, [model.predict(x) for x in self.dme_data.T]



if __name__ == "__main__":
    Generator(version="0.0.0")
