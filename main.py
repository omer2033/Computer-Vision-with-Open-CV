from PIL import Image
import numpy as np
import cv2

###  ADD NOISE TO SELECTED IMAGE ###

image = Image.open("car_image.jfif")
image_array = np.array(Image.open("car_image.jfif"))

for i in range(len(image_array)):
    for j in range(len(image_array[i])):
        blue = image_array[i, j, 0]
        green = image_array[i,j,1]
        red = image_array[i,j,2]
        grayscale_value = (blue * 0.114) + (green * 0.587) + (red * 0.299)
        image_array[i,j] = grayscale_value
    save_image = Image.fromarray(image_array)
    save_image.save("gray_car_image.jpg")


image_gray = Image.open("gray_car_image.jpg")
gray_resized_image = image_gray.resize((256, 256))
gray_resized_image.save('image_256.jpg')

noisy_image = cv2.imread('image_256.jpg')

gauss = np.random.normal(0, 1, noisy_image.size)
gauss = gauss.reshape(noisy_image.shape[0], noisy_image.shape[1], noisy_image.shape[2]).astype('uint8')
img_gauss = cv2.add(noisy_image, gauss)
image.show()
image_gray.show()
gray_resized_image.show()

cv2.imshow('Gray Noisy Image',img_gauss)
cv2.imwrite("noisy_image.jpg", img_gauss)

###  OBSERVE BLURRING EFFECT ON THE NOISY IMAGE ###
noisy_image = cv2.imread('noisy_image.jpg', 0)
row, col = noisy_image.shape
mask = np.ones([3, 3], dtype=int)
mask = mask / 9
last_image = np.zeros([row, col])

for i in range(1, row - 1):
    for j in range(1, col - 1):
        temp = noisy_image[i - 1, j - 1] * mask[0, 0] + noisy_image[i - 1, j] * mask[0, 1] + noisy_image[i - 1, j + 1] * mask[0, 2] + noisy_image[
            i, j - 1] * mask[1, 0] + noisy_image[i, j] * mask[1, 1] + noisy_image[i, j + 1] * mask[1, 2] + noisy_image[i + 1, j - 1] * mask[
                   2, 0] + noisy_image[i + 1, j] * mask[2, 1] + noisy_image[i + 1, j + 1] * mask[2, 2]
        last_image[i, j] = temp

last_image = last_image.astype(np.uint8)

cv2.imshow('Blurred Image', last_image)
cv2.imwrite('blurred.jpg', last_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

















