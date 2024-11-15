from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localisation

# grouping connected regions
label_image = measure.label(localisation.binary_car_image)  # measuring it against the bw picture

# dimensions that the license plate can be
# the plate height would be between 8% and 20%, and width would be 15% and 40%
# in this example, ive brute forced the adjustments of the values as this image does not conform to usual restraints -- possibility to optimise this
# it is now min height of 1% and max width of 50%
min_height, max_height, min_width, max_width = (0.01*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.5*label_image.shape[1])
plate_object_coords = []
plate_like_objects = []

fig, (ax1) = plt.subplots(1)
ax1.imshow(localisation.blue_car_image, cmap="gray");

for region in regionprops(label_image):
    if region.area < 50:
        # 50 is an arbitrary number. if numberplate is too small then it wont be detected
        continue

    # bounding where the numberplate could be with a red border
    min_row, min_col, max_row, max_col = region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col

    # refining to only license plates to have a border
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
        plate_like_objects.append(localisation.binary_car_image[min_row:max_row, min_col:max_col])
        plate_object_coords.append((min_row, min_col, max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)

plt.show()
