from geopy.distance import geodesic
from geojson import Point, Polygon, Feature
import shutil
import os



path_to_raw="/home/azureuser/codebase/CDEM/CellEnMon/datasets/dme/09062011_13082011/rawzz"
path_to_raw2="/home/azureuser/codebase/CDEM/CellEnMon/datasets/dme/09062011_13082011/raw00"

gauges = [(5.180,52.097),   #260
            (5.145,51.857), #356
            (4.926,51.968), #348
]
counter=0
for link in os.listdir(path_to_raw2):
    if link!=".ipynb_checkpoints":
        T,XStart,YStart,R,XEnd,YEnd=link.split("_")

        YEnd=YEnd.replace(".csv","")
        YStart=float(YStart)
        YEnd=float(YEnd)
        YCenter=(YStart+YEnd)/2

        XStart=float(XStart)
        XEnd=float(XEnd)
        XCenter=(XStart+XEnd)/2
        print(YCenter, XCenter)

        link_point = Feature(geometry=Point((YCenter, XCenter)))

        # Extract coordinates from the geojson feature
        link_point = (link_point['geometry']['coordinates'][0], link_point['geometry']['coordinates'][1])

        for gauge in gauges:

            gauge_point = (gauge[0], gauge[1])
            print(geodesic(gauge_point, link_point).kilometers)
            if geodesic(gauge_point, link_point).kilometers <= 15:
                print("Condition holds!")
                shutil.copy(f"{path_to_raw2}/{link}", f"{path_to_raw}/{link}")