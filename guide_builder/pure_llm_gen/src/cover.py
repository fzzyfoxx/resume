
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import cv2 as cv
import seaborn as sns

class VoronoiCover:
    def __init__(self, height=1754, width=1240, n_points=1200, palette='crest', n_colors=6, ratios=[10,1,0], radius=20, buffer=2, border_color=100, darkening_ratio=90):
        self.height = height
        self.width = width
        self.n_points = n_points

        self.palette = palette
        self.n_colors = n_colors
        self.ratios = ratios
        self.radius = radius
        self.buffer = buffer
        self.border_color = border_color
        self.darkening_ratio = darkening_ratio

    def _get_polygons(self, vor):
            polygons = []
            for region in vor.regions:
                if (len(region) > 2) and (not -1 in region):
                    polygon = [vor.vertices[i] for i in region]
                    polygon.append(polygon[0])
                    polygons.append(np.array(polygon))
            return polygons
    
    def _rounded_corners(self, polygon):
            new_polygon = polygon.buffer(self.radius, join_style='bevel', cap_style='round').buffer(-self.radius, join_style='round', cap_style='round')
            return new_polygon

    def __call__(self, save_path):

        # Generate random points
        points = np.random.rand(self.n_points, 2) * [self.width, self.height]
        points = np.append(points, [[9999,9999], [-9999,9999], [9999,-9999], [-9999,-9999]], axis = 0)
        # Compute Voronoi diagram
        vor = Voronoi(points)  
        polygons = self._get_polygons(vor)

        # Define the bounding box
        bounding_box = box(0, 0, self.width, self.height)

        # Clip polygons to the bounding box
        clipped_polygons = [Polygon(polygon).intersection(bounding_box) for polygon in polygons]

        clipped_polygons = [self._rounded_corners(polygon).buffer(-3) for polygon in clipped_polygons]
        clipped_polygons = [polygon for polygon in clipped_polygons if not polygon.is_empty]
        centroids = [polygon.centroid.coords[0] for polygon in clipped_polygons]
        clipped_polygons = [np.stack(polygon.exterior.coords.xy, axis=-1) for polygon in clipped_polygons]

        img = np.ones((self.height, self.width, 3), np.uint8) * self.border_color

        palette = sns.color_palette(self.palette, self.n_colors)
        colors = np.array(palette) * 255

        positions_x = np.random.randint(0, self.width, (self.n_colors, 1))
        positions_y = np.random.randint(0, self.width, (self.n_colors, 1))
        positions = np.concatenate([positions_x, positions_y], axis=1)

        size = (self.width+self.height)/2
        ratios = np.array(self.ratios)
        ratios = ratios / sum(ratios)
        # Calculate color for each polygon based on distance to each color point
        for i, polygon in enumerate(clipped_polygons):
            centroid = centroids[i]
            distances = [1/(np.linalg.norm(np.array(centroid) - np.array(pos), ord=1)/size)**2 for pos in positions]
            total_distance = sum(distances)
            weights = [d / total_distance for d in distances]
            avg_color = np.average(colors, axis=0, weights=weights).astype(float)
            rnd_color = np.random.randint(0, 255, 3)
            max_color = colors[np.argmax(weights)]
            color = np.sum(np.array([avg_color, rnd_color, max_color])*ratios[:,np.newaxis], axis=0)
            cv.fillPoly(img, [polygon.astype(int)], color=tuple(color.astype(float)))

        dark_img = np.ones((self.height, self.width, 3), np.uint8) * self.darkening_ratio
        img = cv.subtract(img, dark_img)

        n_k = 1/16
        kernel = np.array([[n_k, n_k, n_k],[n_k,0.5,n_k],[n_k,n_k,n_k]],np.float32)
        img = cv.filter2D(img,-1,kernel)

        cv.imwrite(save_path, img[...,::-1], [cv.IMWRITE_JPEG_QUALITY, 100])