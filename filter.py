import pandas as pd
from geopy.distance import distance

target_coordinates = (22.2983676, 114.168157)  # (latitude, longitude)


csv_file = "/Users/ouyifan/Documents/25fall-IP/data_test/sanitized-tst-poi.csv"  


def is_within_30m(row):
    point = (row['latitude'], row['longitude'])
    return distance(target_coordinates, point).meters <= 200


filtered_df = df[df.apply(is_within_30m, axis=1)]

filtered_df.to_csv("/Users/ouyifan/Documents/25fall-IP/data_filteredpoi/filtered_coordinates.csv", index=False)

print("过滤完成，结果已保存至 'filtered_poi.csv'")
