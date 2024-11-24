import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cca

# there may be cases where the license plate is not found and is mistaken for something else
# another identifying property of the license plate is the fact that it has characters

# first invert as it is easier to identify characters. chose the third one because i already knew that was where the license plate is
license_plate = np.invert(cca.plate_like_objects[2])
labelled_plate = measure.label(license_plate)

fig, ax1 = plt.subplots(1)
ax1.imshow(license_plate, cmap="gray")

# adjusted the min_width to be 2% to account for letters being split. 0 for y-axis and 1 for x-axis
min_height, max_height, min_width, max_width = (0.35*license_plate.shape[0], 0.60*license_plate.shape[0], 0.05*license_plate.shape[1], 0.15*license_plate.shape[1])

characters = []
counter = 0
column_list = []
for region in regionprops(labelled_plate):
    y0, x0, y1, x1 = region.bbox
    region_height = y1 - y0
    region_width = x1 - x0

    if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
        roi = license_plate[y0:y1, x0:x1]

        rect_border = patches.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rect_border)

        resized_char = resize(roi, (20, 20))
        characters.append(resized_char)

        column_list.append(x0)

plt.show()