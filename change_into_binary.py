import cv2
path = 'E:/research/parkinson_detect/pd_net/otherdataset/data1/drawings/spiral/testing/0/V04HE01.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
thresh, new_img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY_INV)
# thresh_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)

cv2.imshow('img', img)
cv2.imshow('NEW_IMG', new_img)
cv2.imwrite('E:/research/parkinson_detect/pd_net/test_data/test_pic.png', new_img)
cv2.waitKey(0)