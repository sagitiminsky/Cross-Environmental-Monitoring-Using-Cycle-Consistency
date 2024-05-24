import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

root="CellEnMon/datasets/ims"
content=sorted(os.listdir(f"{root}/01012015_01022015/predict/only_dynamic"))
content.remove(".ipynb_checkpoints")
import glob


class Preprocess:
    def __init__(self,link,gauge):
        
        self.link = link.replace("-","_")
        self.gauge=gauge
        
        # Construct the file pattern
        file_pattern = f"{root}/01012015_01022015/processed/{gauge}_*.csv"

        # Use glob to find all files matching the pattern
        gauge_gt_file = glob.glob(file_pattern)[0]
        
        
        
        fake_station = pd.DataFrame(columns=["Time","RainAmout[mm/h]"])


        for file in content:
            df = pd.read_csv(f"{root}/01012015_01022015/predict/only_dynamic/{file}")
            if f"{link}-{gauge}" in file:
                fake_station = pd.concat([fake_station,df], ignore_index=True)



        # Replace small fake values with zero
        fake_station.loc[fake_station['RainAmout[mm/h]'] <= 0.5, 'RainAmout[mm/h]'] = 0


        # Sort by the 'Time' column in ascending order
        fake_station = fake_station.sort_values(by='Time')


        # Change naming to reflect prediction
        fake_station = fake_station.rename(columns={'RainAmout[mm/h]': 'RainAmoutPredicted[mm/h]'})


        # Add GT
        fake_station["RainAmoutGT[mm/h]"]=pd.read_csv(f"{gauge_gt_file}")["RR[mm/h]"][:len(fake_station)]



        # Replace real values with zero
        fake_station.loc[fake_station['RainAmoutGT[mm/h]'] <= 1, 'RainAmoutGT[mm/h]'] = 0



        # Accumulative
        fake_station["RainAmoutPredictedCumSum"]=fake_station['RainAmoutPredicted[mm/h]'].cumsum()
        fake_station["RainAmoutGTCumSum"]=fake_station['RainAmoutGT[mm/h]'].cumsum()

        # Save to csv
        fake_station.to_csv(f"{root}/01012015_01022015/predict/{self.link}_{gauge}.csv", index=False)


        self.excel_data=fake_station

        self.fake = np.asarray(self.excel_data["RainAmoutPredictedCumSum"])
        self.real = np.asarray(self.excel_data["RainAmoutGTCumSum"])

        #mmax=np.max(np.asarray(excel_data["RainAmoutGT[mm/h]"]))

#         time=np.asarray(excel_data.Time)
#         plt.plot(time, fake, label="CML")
#         plt.plot(time, real, "--", label="Gauge")
#         plt.grid()
#         plt.xlabel("Time")
#         plt.ylabel("Accumulated Rain Rate [mm]")
#         plt.legend()
#         plt.tight_layout()

#         # Specify the number of ticks you want on the x-axis
#         num_ticks = 10

#         # Calculate the step size between ticks
#         step_size = len(time) // num_ticks

#         # Set the ticks on the x-axis
#         plt.xticks(time[::step_size], rotation=45)

#         plt.show()


if __name__=="__main__":
    pass
