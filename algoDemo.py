import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.io import imread

# Load picture, convert to grayscale and detect edges
# image_rgb = data.coffee()[0:220, 160:420]

# coffee = data.coffee()
# plt.imshow(coffee)
# plt.show()

fname = input('Enter File Name\n')

image_rgb = imread(fname, as_gray=True)
plt.imshow(image_rgb)
plt.show()


print('Detecting Edges')
edges = canny(image_rgb)
# , sigma=2.0, low_threshold=0.55, high_threshold=0.8)
plt.imshow(edges)
plt.show()
# Perform a Hough Transform
# The accuracy corresponds to the bin size of a major axis.
# The value is chosen in order to get a single high accumulator.
# The threshold eliminates low accumulators
# plt.imshow(edges)
# plt.show()

print('Performing Hough Transform')
result = hough_ellipse(edges, min_size=70)
# accuracy=20, threshold=250,
# min_size=100, max_size=120)
print('Sorting result')
result.sort(order='accumulator')

# Estimated parameters for the ellipse
best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]
print('Found Ellipse Parameters')
# Draw the ellipse on the original image
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
print('Finding Ellipse Perimeter')
image_rgb[cy, cx] = (0, 0, 255)
# Draw the edge (white) and the resulting ellipse (red)
edges = color.gray2rgb(img_as_ubyte(edges))
edges[cy, cx] = (250, 0, 0)

fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                sharex=True, sharey=True)

ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)

plt.show()
