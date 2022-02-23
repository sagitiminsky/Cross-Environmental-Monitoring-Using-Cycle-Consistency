import os
import numpy as np
import folium
from pathlib import Path
import sys

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
    def __init__(self):
        self.map_name="TRY_MAP.html"
        self.dates_range="01012013_01022013"
        self.base_path=Path(f"/datasets/{self.dates_range}")
        self.data_path=self.base_path.joinpath("processed")
        self.out_path = self.base_path.joinpath("visualization")
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.list_of_link_id_to_drop=['4673-7HZ4','7HZ4-4673']
        self.list_of_link_id_to_color=['A065-H059','j220-s220']
        self.color_of_links = 'red'
        self.gridlines_on = False
        self.num_of_gridlines=30

        self.handle=self.draw_cml_map()

    def parse_instances(self,instance):
        instance_arr=instance.split("-")
        return {
            "ID": f"{instance_arr[0]}-{instance[3]}",
            "Tx Site Latitude":instance_arr[1],
            "Tx Site Longitude":instance_arr[2],
            "Rx Site Latitude": instance_arr[4],
            "Rx Site Longitude": instance_arr[5],
        }

    def draw_cml_map(self):
        num_cmls_map = len(os.listdir(self.data_path))
        grid = []

        map_1 = folium.Map(location=[32, 35],
                           zoom_start=8,
                           tiles='Stamen Terrain',
                           control_scale=True)

        lat_min=sys.maxsize
        lon_min=sys.maxsize
        lat_max=-sys.maxsize
        lon_max=-sys.maxsize

        for instance in os.listdir(self.data_path):
            instace_dict=self.parse_instances(instance)
            lat_min=min(lat_min,instace_dict["Tx Site Latitude"], instace_dict["Rx Site Latitude"])
            lon_min=min(lon_min,instace_dict["Tx Site Longitude"], instace_dict["Rx Site Longitude"])
            lat_max=max(lat_max, instace_dict["Tx Site Latitude"], instace_dict["Rx Site Latitude"])
            lon_max=max(lon_max, instace_dict["Tx Site Longitude"],instace_dict["Rx Site Longitude"] )

            if instace_dict['ID'] in self.list_of_link_id_to_drop:
                print('Link ID' + str(instace_dict['ID']) + ' has been dropped')
                num_cmls_map = num_cmls_map - 1
                continue
            else:
                folium.PolyLine([(instace_dict['Rx Site Latitude'],
                                  instace_dict['Rx Site Longitude']),
                                 (instace_dict['Tx Site Latitude'],
                                  instace_dict['Tx Site Longitude'])],
                                color=self.color_of_links,
                                opacity=0.6,
                                popup=f"ID:{instace_dict['ID']}"
                                ).add_to(map_1)

        print('Number of links in map: ')
        print(num_cmls_map)

        # for l_color in list_of_link_id_to_color:
        #     link = df_md.loc[df_md['Link ID'] == l_color]
        #     folium.PolyLine([(float(link['Rx Site Latitude'].values),
        #                       float(link['Rx Site Longitude'].values)),
        #                      (float(link['Tx Site Latitude'].values),
        #                       float(link['Tx Site Longitude'].values))],
        #                     color=color_of_specific_links,
        #                     opacity=0.8,
        #                     popup=str(link['Link Carrier'].values) + '\nID: ' + str(link['Link ID'].values)
        #                     ).add_to(map_1)

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

        map_1.save(self.out_path)

        print(f"Map under the name {self.map_name} was generated")

        return map_1

