import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

root="CellEnMon/datasets/ims"
content=sorted(os.listdir(f"{root}/01012015_01022015/predict/only_dynamic"))
# content.remove(".ipynb_checkpoints")
import glob
import matplotlib.dates as mpl_dates



class Preprocess:
    def __init__(self,link,gauge,epoch,T,real,fake,detections):
        
        self.link = link.replace("-","_")
        self.gauge=gauge

        df_detections = pd.DataFrame(data={'Time': T, 'Detections': detections })
        detections = np.asarray(df_detections['Detections'],dtype=float)
        
        
        d = {'Time':pd.to_datetime(T), 'RainAmoutGT[mm/h]':real, 'RainAmoutPredicted[mm/h]': fake , 'Detections':detections, "FakeWithDetections": fake*detections } #
        df = pd.DataFrame(data=d)




        # Replace neg fake values with zero
        df.loc[df['RainAmoutPredicted[mm/h]'] < 0.1, 'RainAmoutPredicted[mm/h]'] = 0
        # df.loc[df['RainAmoutPredicted[mm/h]'] > 3.3, 'RainAmoutPredicted[mm/h]'] = 3.3

        self.fake=np.asarray(df['RainAmoutPredicted[mm/h]'],dtype=float)
        self.real=np.asarray(df['RainAmoutGT[mm/h]'],dtype=float)
        self.detections = np.asarray(df["Detections"], dtype=float)
        self.fake_with_detection = np.array(df["FakeWithDetections"], dtype=float)

        # Accumulative
        df["RainAmoutPredictedCumSum"]=df['RainAmoutPredicted[mm/h]'].cumsum()
        df["RainAmoutGTCumSum"]=df['RainAmoutGT[mm/h]'].cumsum()
        df["FakeWithDetectionsCumSum"]=df["FakeWithDetections"].cumsum()

        # Save to csv
        df.to_csv(f"{root}/01012015_01022015/predict/{self.link}_{gauge}.csv", index=False)


        self.excel_data=df

        self.fake_cumsum = np.asarray(self.excel_data["RainAmoutPredictedCumSum"])
        self.real_cumsum = np.asarray(self.excel_data["RainAmoutGTCumSum"])
        self.detections = np.asarray(self.excel_data["Detections"])
        self.fake_with_detection_cumsum = np.asarray(self.excel_data["FakeWithDetectionsCumSum"])



if __name__=="__main__":
    pass
