import geopandas as gpd
import folium


# reading data of districts

gdf = gpd.read_file(r"D:\ArcGISpro\NewGEOt\districts_sample_200.geojson")
print("File loaded successfully")


# checking of crs parameters

print("Original CRS:", gdf.crs)


# convert to metric CRS to calculate area
gdf = gdf.to_crs(epsg=3857)
gdf["area_km2"] = gdf.geometry.area / 1_000_000


# crs conversion to geographic and checking crs parameters

gdf = gdf.to_crs(epsg=4326)
print("CRS after conversion:", gdf.crs)



# get the centroids coordinates from geometry column

gdf["centroid"] = gdf.geometry.centroid
gdf["centroid_lon"] = gdf.centroid.x
gdf["centroid_lat"] = gdf.centroid.y



# riyadh city center coordinates

riyadh_coords = (24.7136, 46.6753)


# create function to get district direction from riyadh center

def get_direction(lat, lon):
    if lat > riyadh_coords[0]:
        ns = "North"
    else:
        ns = "South"

    if lon > riyadh_coords[1]:
        ew = "East"
    else:
        ew = "West"

    return f"{ns}-{ew}"



# function to get hints from sampled district (3 hints)

def get_hints(district):
    hints = [
        f"The district is located in the {get_direction(district['centroid_lat'], district['centroid_lon'])} of Riyadh.",
        f"The area of the district is approximately {district['area_km2']:.2f} square kilometers.",
        f"The district center is at ({district['centroid_lat']:.3f}, {district['centroid_lon']:.3f})."
    ]
    return hints



# function to create game loop

def play_game(gdf):
    sampled = gdf.sample(n=5).reset_index(drop=True)
    score = 0

    for index, district in sampled.iterrows():
        print(f"\nDistrict {index + 1} of {len(sampled)}")
        print("Guess the neighborhood name:")

        hints = get_hints(district)
        attempts = 3
        correct_answer = str(
            district.get("NEIGHBORHENAME", district.get("NEIGHBORHANAME", "Unknown"))
        ).strip().lower()

        while attempts > 0:
            print("\nHINT:")
            print(hints[3 - attempts])

            guess = input(f"Your guess (Attempts left: {attempts}): ").strip().lower()

            if guess == correct_answer:
                print("Correct!")
                score += 1
                break
            else:
                attempts -= 1
                if attempts == 0:
                    print(f"Wrong! The correct answer is: {correct_answer}")

        # show map after each district
        show_map_for_district(district, index + 1)

    print(f"\nGame Over! Your score: {score}/{len(sampled)}")



# function to show map with folium

def show_map_for_district(district, district_number):
    center = [district["centroid_lat"], district["centroid_lon"]]
    m = folium.Map(location=center, zoom_start=12)

    folium.GeoJson(
        district.geometry.__geo_interface__,
        style_function=lambda x: {
            "color": "red",
            "weight": 3,
            "fillColor": "yellow",
            "fillOpacity": 0.5,
        },
    ).add_to(m)

    name = district.get("NEIGHBORHENAME", district.get("NEIGHBORHANAME", "Unknown"))

    folium.Marker(
        location=center,
        popup=f"<b>{name}</b>",
        icon=folium.Icon(color="green", icon="home"),
    ).add_to(m)

    filename = f"district_{district_number}_{str(name).replace(' ', '_')}.html"
    m.save(filename)
    print(f"Map saved as: {filename}")



# start the game

play_game(gdf)
