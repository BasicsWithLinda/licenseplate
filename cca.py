from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localisation

# grouping connected regions
label_image = measure.label(localisation.binary_car_image)  # measuring it against the bw picture
fig, (ax1) = plt.subplots(1)
ax1.imshow(localisation.blue_car_image, cmap="gray")

for region in regionprops(label_image):
    if region.area < 50:
        # 50 is an arbitrary number. if numberplate is too small then it wont be detected
        continue

    # bounding where the numberplate could be with a red border
    minRow, minCol, maxRow, maxCol = region.bbox
    rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="red", linewidth=2, fill=False)
    ax1.add_patch(rectBorder)

plt.show()
