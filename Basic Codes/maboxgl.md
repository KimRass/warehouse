# mapboxgl
## mapboxgl.viz
```python
from mapboxgl.viz import *
```
### CircleViz()
```python
viz = CircleViz(data=geo_data, access_token=token, center=[127.46,36.65], zoom=11, radius=2, stroke_color="black", stroke_width=0.5)
```
### GraduatedCircleViz()
```python
viz = GraduatedCircleViz(data=geo_data, access_token=token, height="600px", width="600px", center=(127.45, 36.62), zoom=11, scale=True, legend_gradient=True, add_snapshot_links=True, radius_default=4, color_default="black", stroke_color="black", stroke_width=1, opacity=0.7)
```
### viz.style
```python
viz.style = "mapbox://styles/mapbox/outdoors-v11"
```
- "mapbox://styles/mapbox/streets-v11"
- "mapbox://styles/mapbox/outdoors-v11"
- "mapbox://styles/mapbox/light-v10"
- "mapbox://styles/mapbox/dark-v10"
- "mapbox://styles/mapbox/satellite-v9"
- "mapbox://styles/mapbox/satellite-streets-v11"
- "mapbox://styles/mapbox/navigation-preview-day-v4"
- "mapbox://styles/mapbox/navigation-preview-night-v4"
- "mapbox://styles/mapbox/navigation-guidance-day-v4"
- "mapbox://styles/mapbox/navigation-guidance-night-v4"
### viz.show()
```python
viz.show()
```
### viz.create_html()
```python
with open("D:/☆디지털혁신팀/☆실거래가 분석/☆그래프/1km_store.html", "w") as f:
    f.write(viz.create_html())
```
## mapboxgl.utils
```python
from mapboxgl.utils import df_to_geojson, create_color_stops, create_radius_stops
```
### df.to_geojson()
```python
geo_data = df_to_geojson(df=df, lat="lat", lon="lon")
```
### viz.create_color_stops()
```python
viz.color_property = "error"
viz.color_stops = create_color_stops([0, 10, 20, 30, 40, 50], colors="RdYlBu")
```
### viz.create_radius_stops()
```python
viz.radius_property = "errorl"
viz.radius_stops = create_radius_stops([0, 1, 2], 4, 7)
```
