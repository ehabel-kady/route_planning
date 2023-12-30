import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import osmnx as ox
import pyproj
import overpy
import contextily as ctx
from shapely.ops import unary_union, polygonize
from shapely.geometry import Point, LineString, Polygon

import matplotlib.pyplot as plt


class HandleCycleData:
    def __init__(self, city="Tartu"):
        self.city = ox.features_from_place(
            f"{city}, Estonia", tags={"name": f"{city} linn", "type": "boundary"}
        )
        self.city_graph = None
        self.nodes, self.edges, self.filtered_edges = None, None, None

    def check(self, highway):
        cycle = ["cycleway"]
        if isinstance(highway, list):
            return any(value in highway for value in cycle)
        else:
            return highway in cycle

    def get_cycling_data(self):
        self.city = self.city.query("name == 'Tartu linn' and admin_level == '9'")
        self.crs_web = pyproj.CRS.from_epsg("3857")
        self.crs_gps = pyproj.CRS.from_epsg("4326")

        polygon = unary_union(list(self.city.to_crs(crs=self.crs_gps).geometry))
        self.city_graph = ox.graph_from_polygon(polygon)

        self.nodes, self.edges = ox.graph_to_gdfs(self.city_graph)
        self.mask = self.edges["highway"].apply(self.check)
        self.filtered_edges = self.edges[self.mask]
        return self.filtered_edges, polygon

    def plot_filtered_graph(self):
        temp = ox.graph_from_gdfs(self.nodes, self.filtered_edges)
        tartu_cyclable = self.city_graph.edge_subgraph(temp.edges)

        fig, ax = ox.plot_graph(
            tartu_cyclable, node_size=3, edge_color="blue", figsize=(14, 14)
        )


class BikeData:
    def __init__(self, bike_data):
        self.bikes = pd.read_csv(bike_data)
        self.city_bikes = None

    def get_bike_locations(self, polygon):
        points = []
        locs = []
        for index, row in self.bikes.iterrows():
            points.append(Point(row["X"], row["Y"]))
            locs.append(row["Asukoht"])

        d = {"geometry": points, "location": locs}
        gdf_bikes = gpd.GeoDataFrame(d).set_crs(
            "epsg:3301"
        )  # Estonian Coordinate System of 1997
        gdf_bikes = gdf_bikes.to_crs("epsg:4326")  # WGS 84

        self.city_bikes = gdf_bikes[
            gdf_bikes.apply(lambda row: row["geometry"].within(polygon), axis=1)
        ].reset_index()

    def plot_base_map(self, polygon, crs_gps, routes=None):
        from matplotlib.colors import (
            ListedColormap,
        )  # This is necessary to have all the points in the same color. Weird implementation, I know.

        mycolor = ListedColormap("blue")

        map_source = ctx.providers.OpenStreetMap.DE

        tartu_boundary = gpd.GeoDataFrame(geometry=[polygon]).set_crs(epsg="4326")
        ax = tartu_boundary.plot(
            edgecolor="red", facecolor="none", figsize=(14, 14), alpha=0.5
        )
        ctx.add_basemap(ax=ax, crs=crs_gps, source=map_source, zoom=15)

        cols = ["green", "blue", "orange"]

        if routes:
            if len(routes) == 3:
                graph = ox.graph_from_polygon(polygon)
                ox.plot_graph_routes(
                    graph,
                    ax=ax,
                    routes=routes,
                    bgcolor="none",
                    node_color="none",
                    node_size=0,
                    edge_color="none",
                    edge_linewidth=0,
                    route_colors=cols,
                    orig_dest_size=6,
                    route_linewidth=1,
                )
            elif len(routes) == 1:
                graph = ox.graph_from_polygon(polygon)
                ox.plot_graph_route(
                    graph,
                    ax=ax,
                    route=routes[0],
                    bgcolor="none",
                    node_color="none",
                    node_size=0,
                    edge_color="none",
                    edge_linewidth=0,
                    route_color="blue",
                    orig_dest_size=6,
                    route_linewidth=1,
                )
        else:
            self.city_bikes.plot(ax=ax, column="geometry", cmap=mycolor)


class BikeRouter:
    def __init__(self, bike_data_path):
        self.data_handler = HandleCycleData("Tartu")
        self.bike_data = BikeData(bike_data_path)
        self.filtered_edges, self.polygon = None, None

    def add_remaining_nodes(self):
        cols = ["osmid", "highway", "reversed", "length", "geometry"]
        data = self.filtered_edges[cols]

        tartu_filled = pd.DataFrame(data, columns=cols, index=self.filtered_edges.index)
        tartu_filled["weight"] = 1.0  # initially all are cycleways
        self.tartu_graph = ox.graph_from_polygon(self.polygon)

        self.nodes, self.edges = ox.graph_to_gdfs(self.tartu_graph)

        inverse_filtered = self.edges[~self.data_handler.mask]
        inverse_filtered = pd.DataFrame(
            inverse_filtered, columns=cols, index=inverse_filtered.index
        )
        inverse_filtered["weight"] = 2.5  # remaining classes have higher weight

        self.tartu_filled = pd.concat([tartu_filled, inverse_filtered])
        self.tartu_filled["wlen"] = tartu_filled.apply(
            lambda row: row["length"] * row["weight"], axis=1
        )

    def preprocess(self):
        self.filtered_edges, self.polygon = self.data_handler.get_cycling_data()
        # self.data_handler.plot_filtered_graph()
        self.bike_data.get_bike_locations(self.polygon)
        # self.bike_data.plot_base_map(self.polygon, self.data_handler.crs_gps)
        self.add_remaining_nodes()

    def calculate_shortest_path(self):
        gdf = gpd.GeoDataFrame(self.tartu_filled)
        temp2 = ox.graph_from_gdfs(self.nodes, gdf)
        self.tartu_filled_cyclable = self.tartu_graph.edge_subgraph(temp2.edges)

        # Copy over the wlen attribute
        wlens = {}
        for i, j in self.tartu_filled.iterrows():
            wlens[i] = {"wlen": j["wlen"]}

        nx.set_edge_attributes(self.tartu_filled_cyclable, wlens)

        # Add speed limits
        new_maxspeeds = {}
        for u, v, k, d in self.tartu_filled_cyclable.edges(keys=True, data=True):
            if "maxspeed" not in d:
                new_maxspeeds[(u, v, k)] = {"maxspeed": 28}

        nx.set_edge_attributes(self.tartu_filled_cyclable, new_maxspeeds)

        # As we are going to use the "speed_kph" attribute later anyway,
        # let's use the automated method to set this attribute on all roads
        ox.speed.add_edge_speeds(self.tartu_filled_cyclable)

        # Get estimated travel times for all roads (with 28 km/h limit)
        self.tartu_filled_cyclable = ox.speed.add_edge_travel_times(
            self.tartu_filled_cyclable
        )

    def calculate_route(self, p1, p2, share):
        p1 = ox.geocode_to_gdf(p1)
        p2 = ox.geocode_to_gdf(p2)

        # Combine discovered building shapes, assign labels and path direction
        gdf_addr = (
            gpd.GeoDataFrame(pd.concat([p1, p2]))
            .assign(label=["Start", "Stop"], path=["start", "stop"])
            .set_index("path")
            .to_crs(crs=self.data_handler.crs_web)
        )

        # Find coordinates of the two points
        lons, lats = np.vstack(
            gdf_addr.geometry.centroid.to_crs(crs=self.data_handler.crs_gps)
            .apply(lambda gdf: np.asarray(gdf.xy).reshape(-1))
            .values
        ).T

        # Find the closest nodes in a road graph
        (start, stop), dists = ox.nearest_nodes(
            self.tartu_filled_cyclable, lons, lats, return_dist=True
        )
        print(f"Closest road to P1: {round(dists[0],1)}m")
        print(f"Closest road to P2: {round(dists[1],1)}m")

        # EVA WROTE THIS PART ON THURSDAY
        if share:
            gdf_parking = self.bike_data.city_bikes
            points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
            source = points[0]
            dest = points[1]

            graph = ox.graph_from_polygon(self.polygon)

            dists_from_source = []
            dists_from_dest = []
            for point in gdf_parking["geometry"]:
                dists_from_source.append(
                    ox.distance.euclidean(source.y, source.x, point.y, point.x)
                )
                dists_from_dest.append(
                    ox.distance.euclidean(dest.y, dest.x, point.y, point.x)
                )

            gdf_parking["dist_from_source"] = dists_from_source
            gdf_parking["dist_from_dest"] = dists_from_dest

            # Shapely POINT
            parking_source = gdf_parking.loc[
                gdf_parking["dist_from_source"].idxmin(), "geometry"
            ]
            parking_dest = gdf_parking.loc[
                gdf_parking["dist_from_dest"].idxmin(), "geometry"
            ]

            # Finding the nearest nodes to the points
            start_p, dist_start_p = ox.nearest_nodes(
                graph, parking_source.x, parking_source.y, return_dist=True
            )
            stop_p, dist_stop_p = ox.nearest_nodes(
                graph, parking_dest.x, parking_dest.y, return_dist=True
            )

            route_to_parking = ox.shortest_path(
                graph, start_p, start, weight="length", cpus=None
            )
            route = ox.shortest_path(
                self.tartu_filled_cyclable, start_p, stop_p, weight="wlen", cpus=None
            )
            route_from_parking = ox.shortest_path(
                graph, stop_p, stop, weight="length", cpus=None
            )

            return (route_to_parking, route, route_from_parking)

        else:
            # Calculate the route
            # Use length as the metric, use all processors available
            route = ox.shortest_path(
                self.tartu_filled_cyclable, start, stop, weight="wlen", cpus=None
            )
            return route

    def get_travel_time(self, routes):
        ttotal = 0.0
        for route in routes:
            for n1, n2 in zip(route, route[1:]):
                e = self.tartu_filled_cyclable.get_edge_data(n1, n2)
                # Add up the travel time for each road
                if "travel_time" in e[0]:
                    ttotal += float(e[0]["travel_time"])

        return ttotal

    def plot_routes(self, routes):
        self.bike_data.plot_base_map(self.polygon, self.data_handler.crs_gps, routes)


def count_ways(A, B, current_it, max_it, memo):
    # Check if already calculated
    if (A, B, current_it) in memo:
        return memo[(A, B, current_it)]

    # If maximum iterations reached, check if terminal state is reached
    if current_it == max_it:
        return 1 if A == 1 else 0

    # Initialize count
    count = 0

    # For variable A
    if A == 0:
        # Transition A from 0 to 1
        count += count_ways(1, B, current_it + 1, max_it, memo)
    else:
        # Terminal state reached for A = 1
        count += 1

    # For variable B
    if B == 2:
        # Transition B from 2 to 1 or 3
        count += count_ways(A, 1, current_it + 1, max_it, memo)
        count += count_ways(A, 3, current_it + 1, max_it, memo)
    elif B == 1 or B == 3:
        # Transition B from 1 or 3 to adjacent values or stay
        count += count_ways(A, B - 1, current_it + 1, max_it, memo)
        count += count_ways(A, B + 1, current_it + 1, max_it, memo)
    else:
        # For B = 0 or B = 4, transition to adjacent values
        count += (
            count_ways(A, B + 1, current_it + 1, max_it, memo)
            if B == 0
            else count_ways(A, B - 1, current_it + 1, max_it, memo)
        )

    # Memoize the result
    memo[(A, B, current_it)] = count
    return count


def recurse(A, B, current_it, max_it):
    # Initialize memoization dictionary
    memo = {}
    return count_ways(A, B, current_it, max_it, memo)


# Test cases
print(recurse(0, 2, 0, 1))  # Output: 1
print(recurse(0, 2, 0, 2))  # Output: 2
print(recurse(0, 2, 0, 3))  # Output: 4
