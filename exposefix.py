import cv2
import numpy as np
from skimage import exposure

origin = cv2.imread('generated/images/1-ground.png/diffused/1024tk_0_progress_00000.png')

fixed = np.zeros(origin.shape, dtype=origin.dtype)
fixed[:,:,0] = cv2.equalizeHist(origin[:,:,0])
fixed[:,:,1] = cv2.equalizeHist(origin[:,:,1])
fixed[:,:,2] = cv2.equalizeHist(origin[:,:,2])


p2, p98 = np.percentile(origin, (2, 98))
img_rescale = exposure.rescale_intensity(origin, in_range=(p2, p98))

# img_adapteq = exposure.equalize_adapthist(origin, clip_limit=0.03)

cv2.imwrite('generated/images/1-ground.png/diffused/exposefix.png', fixed)
# cv2.imwrite('generated/images/1-ground.png/diffused/exposefix_adapteq.png', img_adapteq)
cv2.imwrite('generated/images/1-ground.png/diffused/exposefix_rescale.png', img_rescale)