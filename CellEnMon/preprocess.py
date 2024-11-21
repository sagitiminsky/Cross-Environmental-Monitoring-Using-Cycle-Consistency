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
    def __init__(self,link,gauge,epoch,T,real,fake,rec,fake_detections,rec_detections):
        
        self.link = link.replace("-","_")
        self.gauge=gauge

        df_detections = pd.DataFrame(data={'Time': T, 'fake_detections': fake_detections, 'rec_detections': rec_detections })
        fake_detections = np.asarray(df_detections['fake_detections'],dtype=float)
        rec_detections = np.asarray(df_detections['rec_detections'],dtype=float)
        
        
         #* fake_detections
         #* rec_detections

        d = {'Time':pd.to_datetime(T),\
                'real':real,\
                'fake':fake,\
                'rec': rec,\
                'fake_det': fake_detections,\
                'rec_det': rec_detections,\
                "fake_dot_det": fake ,\
                "rec_dot_det": rec 
                
            }
        df = pd.DataFrame(data=d)


        # Replace neg fake values with zero
        df.loc[df['rec'] < 0.1, 'rec'] = 0
        df.loc[df['fake'] < 0.1, 'fake'] = 0
        df.loc[df["rec_dot_det"] < 0.1, "rec_dot_det"] = 0
        df.loc[df["fake_dot_det"] < 0.1, "fake_dot_det"] = 0
        # df.loc[df['RainAmoutPredicted'] > 3.3, 'RainAmoutPredicted'] = 3.3

        self.real               =np.asarray(df['real'],dtype=float)
        self.fake               =np.asarray(df['fake'],dtype=float)
        self.rec                =np.asarray(df['rec'],dtype=float)
        self.fake_det           =np.asarray(df['fake_det'],dtype=float)
        self.rec_det            = np.asarray(df["rec_det"], dtype=float)
        self.fake_dot_det       = np.asarray(df["fake_dot_det"], dtype=float)
        self.rec_dot_det        = np.asarray(df["rec_dot_det"], dtype=float)

        # Accumulative
        df["real_cumsum"]=df['real'].cumsum()
        df["fake_cumsum"]=df['fake'].cumsum()
        df["rec_cumsum"]=df['rec'].cumsum()
        df["fake_dot_det_cumsum"]=df['fake_dot_det'].cumsum()
        df["rec_dot_det_cumsum"]=df["rec_dot_det"].cumsum()

        # Save to csv
        df.to_csv(f"{root}/01012015_01022015/predict/{self.link}_{gauge}.csv", index=False)


        self.excel_data=df

        self.real_cumsum = np.asarray(self.excel_data["real_cumsum"])
        self.fake_cumsum = np.asarray(self.excel_data["fake_cumsum"])
        self.rec_cumsum =  np.asarray(self.excel_data["rec_cumsum"])
        self.fake_dot_det_cumsum = np.array(self.excel_data["fake_dot_det_cumsum"])
        self.rec_dot_det_cumsum = np.asarray(self.excel_data["rec_dot_det_cumsum"])



if __name__=="__main__":
    pass
