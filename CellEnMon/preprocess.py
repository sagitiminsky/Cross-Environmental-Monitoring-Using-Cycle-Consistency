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
                
        # Use glob to find all files matching the pattern
#         gauge_gt_file = glob.glob(file_pattern)[0]
  
        gauge_gt_file = f"{root}/01012015_01022015/processed/LAHAV_34.87_31.3.csv"
        
        filenames = os.listdir(f"{root}/01012015_01022015/predict/only_dynamic")

        # Remove any item that does not match the pattern of containing a numerical identifier before 'b394'
        cleaned_filenames = [f for f in filenames if f.split('-')[0].isdigit()]

        # Sort the list by the numerical identifier, converting the first split part to integer
        sorted_filenames = sorted(cleaned_filenames, key=lambda x: int(x.split('-')[0]))
        
        
                                  
#         print(sorted_filenames)
#         assert(False)
                                  
        
        fake_station = pd.DataFrame(columns=["Time","RainAmout[mm/h]"])


        for file in sorted_filenames:
            df = pd.read_csv(f"{root}/01012015_01022015/predict/only_dynamic/{file}")
            if f"{link}-{gauge}" in file:
                fake_station = pd.concat([fake_station,df], ignore_index=True)



        # Replace neg fake values with zero
        fake_station.loc[fake_station['RainAmout[mm/h]'] <= 0.0, 'RainAmout[mm/h]'] = 0


        # Sort by the 'Time' column in ascending order
        fake_station = fake_station.sort_values(by='Time')


        # Change naming to reflect prediction
        fake_station = fake_station.rename(columns={'RainAmout[mm/h]': 'RainAmoutPredicted[mm/h]'})


        # Add GT
        fake_station["RainAmoutGT[mm/h]"]=pd.read_csv(f"{gauge_gt_file}")["RR[mm/h]"][:len(fake_station)]



        # Replace real values with zero
#         fake_station.loc[fake_station['RainAmoutGT[mm/h]'] <= 1, 'RainAmoutGT[mm/h]'] = 0
        # Set 'RainAmout[mm/h]' to zero for the specified time range
        df.loc[(df['Time'] >= '2015-01-01 00:00:00') & (df['Time'] <= '2015-01-03 11:30:00'), 'RainAmout[mm/h]'] = 0
        df.loc[(df['Time'] >= '2015-01-04 07:20:00') & (df['Time'] <= '2015-01-07 07:50:00'), 'RainAmout[mm/h]'] = 0
        df.loc[(df['Time'] >= '2015-01-11 10:20:00') & (df['Time'] <= '2015-01-16 23:50:00'), 'RainAmout[mm/h]'] = 0



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
