import os
import shutil

import numpy as np
import folium
from pathlib import Path
import sys
import pandas as pd
import json
import vincent
import CellEnMon.config as config
from math import radians, cos, sin, asin, sqrt

class Visualizer:
    '''Create a Folium interactive map of cmls:
    out_path: str, path to output
    data_path: str, path to metadata file
    metadata_file_nameh: str, .csv file name
    handle: folium.vector_layers.PolyLine, a handle of an exsisting map you wish to
    edit
    name_of_map_file: str, name of the output file
    num_of_gridlines: int, number of gridlines for lat and for lon
    area_min_lon, area_max_lon, area_min_lat, area_max_lat: float, filter area
    of interest by setting coordinates boundaries
    list_of_link_id_to_drop: list of strings, links you wish to discard
    color_of_links: str, color of links from a given csv file

    The function returns a handle for further edditing of the .html file.
    By using the handle multiple companies can be plotted by calling the finction
    for each of them while drawing them in different colors.
    '''

    def __init__(self, experiment_name='dynamic_and_static',virtual_gauge_coo={}):
        self.dates_range = f"{config.start_date_str_rep_ddmmyyyy}_{config.end_date_str_rep_ddmmyyyy}"
        self.map_name = f"{config.export_type}.html"
        self.data_path_dme = Path(f"./CellEnMon/datasets/dme/{self.dates_range}/rawzz")
        self.data_path_ims = Path(f"./CellEnMon/datasets/ims/{self.dates_range}/rawzz")
        self.data_path_produced_ims = Path(f"./CellEnMon/datasets/ims/{self.dates_range}/predict/{experiment_name}")
        self.out_path = Path(f"./CellEnMon/datasets/visualize/{self.dates_range}")
        if not os.path.exists(Path(f"./CellEnMon/datasets/visualize/{self.dates_range}")):
            os.makedirs(self.out_path)

        self.virtual_gagues={}
        self.real_gagues={}
        self.real_links={}
        self.color_of_links = 'red'
        self.color_of_gauges = 'blue'
        self.color_of_produced_gauges = 'green'
        self.color_of_validation = 'black'
        self.gridlines_on = False
        self.num_of_gridlines = 30
    
    def real_and_fake_metric(self,path_to_real_gauge,path_to_fake_gauge):
        real_df=pd.read_csv(path_to_real_gauge, sep=",", index_col=False)
        fake_df=pd.read_csv(path_to_fake_gauge, sep=",", index_col=False)
        
        df_merged=pd.merge(real_df,fake_df, how='inner', on=['Time'])
        df_merged.to_csv(f'./CellEnMon/datasets/dme/{self.dates_range}/merged/merged.csv')
        if len(df_merged):
            return sum((df_merged["RR[mm/h]_x"]-df_merged["RR[mm/h]_y"])**2),len(df_merged),df_merged["RR[mm/h]_x"].to_numpy(),df_merged["RR[mm/h]_y"].to_numpy(),df_merged["Time"].tolist()
        else:
            return None
        
        
    def is_within_radius(self,stations,radius):
        fake_longitude= radians(float(stations["fake_longitude"]))
        fake_latitude=radians(float(stations["fake_latitude"]))
        real_longitutde=radians(float(stations["real_longitude"]))
        real_latitude=radians(float(stations["real_latitude"]))
        
        #1: fake
        #2: real
        

        # Haversine formula
        dlon = real_longitutde - fake_longitude
        dlat = real_latitude - fake_latitude
        a = sin(dlat / 2) ** 2 + cos(fake_latitude) * cos(real_latitude) * sin(dlon / 2) ** 2

        c = 2 * asin(sqrt(a))

        # Radius of earth in kilometers (6371). Use 3956 for miles
        r = 6371
        return c*r < radius #in km
    

    def parse_instances(self, instance,virtual_gauge_coo):
        instance_arr = instance.split("_")
        if len(instance_arr) == 6:
            # Real links
            
            ID=f"{instance_arr[0]}_{instance_arr[3]}"
            self.real_links[ID]={
                "Longitude": (float(instance_arr[1]) + float(instance_arr[4])) / 2,
                "Latitude": (float(instance_arr[2]) + float(instance_arr[5].replace(".csv", ""))) / 2
            }
            
            return {
                "ID": ID,
                "Tx Site Longitude": float(instance_arr[1]),
                "Tx Site Latitude": float(instance_arr[2]),
                "Rx Site Longitude": float(instance_arr[4]),
                "Rx Site Latitude": float(instance_arr[5].replace(".csv", "")),

            }
        elif len(instance_arr)==3:
            # dutch gauges (including virtual)
                        
            ID = f"{instance_arr[0]}"
            Tx_Site_Longitude = instance_arr[2].replace(".csv", "")
            Tx_Site_Latitude = instance_arr[1]
            Rx_Site_Longitude = instance_arr[2].replace(".csv", "")
            Rx_Site_Latitude = instance_arr[1]
            
            self.real_gagues[ID]={
                "Longitude": Tx_Site_Longitude,
                "Latitude": Tx_Site_Latitude
            }
            
            
            return {
                "ID": f"{ID}",
                "Tx Site Longitude": Tx_Site_Longitude,
                "Tx Site Latitude": Tx_Site_Latitude,
                "Rx Site Longitude": Rx_Site_Longitude,
                "Rx Site Latitude": Rx_Site_Latitude
            }
        
        elif len(instance_arr) == 5:
            # israel gauges(including virtual)
            
            ID = f"{instance_arr[2]}"
            Tx_Site_Longitude = instance_arr[3]
            Tx_Site_Latitude = instance_arr[4].replace(".csv", "")
            Rx_Site_Longitude = instance_arr[3]
            Rx_Site_Latitude = instance_arr[4].replace(".csv", "")
            
            self.real_gagues[ID]={
                "Longitude": Tx_Site_Longitude,
                "Latitude": Tx_Site_Latitude
            }
            
            return {
                "ID": f"{instance_arr[2]}",
                "Tx Site Longitude": float(instance_arr[3]),
                "Tx Site Latitude": float(instance_arr[4].replace(".csv", "")),
                "Rx Site Longitude": float(instance_arr[3]),
                "Rx Site Latitude": float(instance_arr[4].replace(".csv", "")),
            }
        else:
            instance_name = instance_arr[1].replace(".csv", "")
            if virtual_gauge_coo:
                # produced ims in dynamic and static experiment
                self.virtual_gagues[instance_name]={
                    "longitude": virtual_gauge_coo["longitude"],
                    "latitude": virtual_gauge_coo["latitude"]
                }

            elif instance_name not in self.virtual_gagues:
                self.virtual_gagues[instance_name]={
                    "longitude": 32.251,
                    "latitude": 35.154
                }
            return {
                "ID": f"{instance_arr[1]}",
                "Tx Site Longitude": float(self.virtual_gagues[instance_name]["longitude"]),
                "Tx Site Latitude": float(self.virtual_gagues[instance_name]["latitude"]),
                "Rx Site Longitude": float(self.virtual_gagues[instance_name]["longitude"]),
                "Rx Site Latitude": float(self.virtual_gagues[instance_name]["latitude"]),
            }

    def draw_cml_map(self,virtual_gauge_name=None, virtual_gauge_coo=None):
        num_links_map = len([file for file in os.listdir(self.data_path_dme) if ".csv" in file])
        num_gagues_map = len([file for file in os.listdir(self.data_path_ims) if ".csv" in file])
        num_produced_gagues_map = len(self.virtual_gagues)

        station_types = {
            "link": self.data_path_dme,
            "gauge": self.data_path_ims,
#             "produced_gague": self.data_path_produced_ims
        }
        num_stations_map = num_gagues_map + num_links_map

        print(f"Number of links on map:{num_links_map}")
        print(f"Number of gauges on map:{num_gagues_map}")
        print(f"Number of stations on map:{num_stations_map}")
        print(f"Number of produced gagues on map:{num_produced_gagues_map}")

        grid = []

        map_1 = folium.Map(location=[32, 35],
                           zoom_start=8,
#                            tiles='Stamen Terrain',
                           control_scale=True)

        lat_min = sys.maxsize
        lon_min = sys.maxsize
        lat_max = -sys.maxsize
        lon_max = -sys.maxsize

        for station_type, data_path in station_types.items():
            for instance in os.listdir(data_path):
                
                if ".csv" in instance:
                    print(f"working on:{instance}...")
                    
                    if station_type=="produced_gague" and virtual_gauge_name in instance:
                        instace_dict = self.parse_instances(instance,virtual_gauge_coo)
                    else:
                        instace_dict = self.parse_instances(instance, virtual_gauge_coo=None)

                    lat_min = min(lat_min, float(instace_dict["Tx Site Latitude"]),
                                  float(instace_dict["Rx Site Latitude"]))
                    lon_min = min(lon_min, float(instace_dict["Tx Site Longitude"]),
                                  float(instace_dict["Rx Site Longitude"]))
                    lat_max = max(lat_max, float(instace_dict["Tx Site Latitude"]),
                                  float(instace_dict["Rx Site Latitude"]))
                    lon_max = max(lon_max, float(instace_dict["Tx Site Longitude"]),
                                  float(instace_dict["Rx Site Longitude"]))

                    # metadata
                    if station_type == "link":
                        color = self.color_of_links
                    elif station_type == "gauge":
                        color = self.color_of_gauges
                    else:
                        color = self.color_of_produced_gauges

                    ## create json of each cml timeseries for plotting

                    df_ts = pd.read_csv(data_path.joinpath(str(instance)))
                    df_ts['Time'] = pd.to_datetime(df_ts['Time'], format='%Y-%m-%d %H:%M')
                    df_ts.set_index('Time', inplace=True, drop=True)
                    timeseries = vincent.Scatter(
                        df_ts,
                        height=150,
                        width=450).axis_titles(
                        x='Time',
                        y='MTSL-mTSL-MRSL-mRSL (dBm)' if station_type == "link" else 'RainRate[mm/h]'
                    )
                    timeseries.marks[0].marks[0].properties.enter.size.value = 10
                    timeseries.legend(title=instace_dict["ID"])
                    data_json = json.loads(timeseries.to_json())

                    v = folium.features.Vega(data_json, width=600, height=200)
                    p = folium.Popup(max_width=1150)

                    if station_type == "link":
                        to_mark=['hOQe_gKVi', 'QLPN_NBOl', 'NBOl_QLPN'] if config.export_type=="dutch" else ['b394_ts04', 'j033_261c']
                        if instace_dict["ID"] in to_mark: 
                            color='black'
                        pl = folium.PolyLine([(instace_dict['Rx Site Longitude'], instace_dict['Rx Site Latitude']),
                                              (instace_dict['Tx Site Longitude'], instace_dict['Tx Site Latitude'])
                                              ],
                                             color=color,
                                             opacity=1.0
                                             ).add_to(map_1)
                    else:
                        to_mark=['260', '348'] if config.export_type == "dutch" else ['LAHAV', 'NIZZAN']
                        if instace_dict["ID"] in to_mark: 
                            color='black'
                        pl = folium.Marker(
                            location=[instace_dict['Rx Site Longitude'], instace_dict['Rx Site Latitude']],
                            popup=folium.Popup(f"ID:{instace_dict['ID']}"),
                            icon=folium.Icon(color=color, prefix='fa', icon='circle')
                        ).add_to(map_1)

                    pl.add_child(p)
                    p.add_child(v)


        # plot gridlines
        lats = np.linspace(lat_min, lat_max, self.num_of_gridlines)
        lons = np.linspace(lon_min, lon_max, self.num_of_gridlines)

        for lat in lats:
            grid.append([[lat, -180], [lat, 180]])

        for lon in lons:
            grid.append([[-90, lon], [90, lon]])

        if self.gridlines_on:
            counter = 0
            for g in grid:
                if counter < len(lats):
                    folium.PolyLine(g, color="black", weight=0.5,
                                    opacity=0.5, popup=str(round(g[0][0], 5))).add_to(map_1)
                    counter += 1
                else:
                    folium.PolyLine(g, color="black", weight=0.5,
                                    opacity=0.5, popup=str(round(g[0][1], 5))).add_to(map_1)

        map_1.save(f"{str(self.out_path)}/{self.map_name}")
        print(f"Map under the name {self.map_name} was generated")

        return map_1


if __name__ == "__main__":
    v = Visualizer(experiment_name="dynamic_and_static").draw_cml_map()
