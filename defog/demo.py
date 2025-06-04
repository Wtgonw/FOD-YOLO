import os
import cv2
from retinex import retinex_AMSR
import matplotlib.pyplot as plt




filepath = r'./fog/'
pathDir = os.listdir(filepath)


for i, allDir in enumerate(pathDir):

    imgpath = r'./fog/' + allDir
    img = cv2.imread(imgpath)
    retinex_img = retinex_AMSR(img, )

    # cv2.imwrite(r'./out_defog/' + allDir, retinex_img)
    cv2.imwrite(os.path.join(r'./out-defog/', allDir), retinex_img)
    print('Finished fogging:'+allDir+'(%g/%g)' % (i+1, len(pathDir)))