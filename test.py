# import cv2

# euroc_path = "./datasets/euroc/MH_02_easy/mav0/cam0/data/1403636858651666432.png"
# image = cv2.imread(euroc_path)
# cv2.imshow("original", image)
# image_trans1 = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
# cv2.imshow("trans1", image_trans1)
# image_trans2 = cv2.cvtColor(image_trans1 , cv2.COLOR_BGR2RGB)
# cv2.imshow("trans2", image_trans2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# tum_path = "./datasets/tum/rgbd_dataset_freiburg1_desk/rgb/1305031470.359624.png"
# image = cv2.imread(tum_path)
# cv2.imshow("original", image)
# image_trans1 = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
# cv2.imshow("trans1", image_trans1)
# image_trans2 = cv2.cvtColor(image_trans1 , cv2.COLOR_BGR2RGB)
# cv2.imshow("trans2", image_trans2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import torch
a = torch.tensor([0 , 1 , 0])
b = torch.tensor([1,0,0,1])
c = torch.logical_and(a , b)