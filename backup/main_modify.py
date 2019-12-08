
'''
 This is the homework for SJTU IE308 Image Processing by Prof. Yi Xu
 Copy right by Sizhe Wei, Dec 2019;  ID: 517021910796
 HW No.2 BM3D Denoising Implement & False Color Transfer
 If you have any question, feel free to contact me at sizhewei@sjtu.edu.cn 
'''

import cv2
import numpy
import math
import numpy.matlib

cv2.setUseOptimized(True)

# Parameters initialization
sigma = 25
Threshold_Hard3D = 2.7 * sigma  # Threshold for Hard Thresholding

Step1_Blk_Size = 4  # block_Size
Step1_Blk_Step = 1  # Here sattle as 1
Step1_Search_Step = 1  # block search step
Match_threshold_1 = 125 * Step1_Blk_Size ** 2  # threshold for similarity
Step1_max_matched_cnt = 16  # Most amounts for group
Step1_Search_Window = 15  # Search for candidate matching blocks in a local neighborhood of restricted size NS*NS centered

Step2_Blk_Size = 4
Step2_Blk_Step = 1
Step2_Search_Step = 1
Match_threshold_2 = 220. / 16 * Step2_Blk_Size ** 2  # threshold for calculating the similarity
Step2_max_matched_cnt = 32
Step2_Search_Window = 25

Kaiser_Beta = 1.5


def init(img, _blk_size, _Kaiser_Beta):
    # Initialize the parameters
    m_shape = img.shape
    m_img = numpy.matrix(numpy.zeros(m_shape, dtype=float))
    m_wight = numpy.matrix(numpy.zeros(m_shape, dtype=float))

    # Window Function: 0 outside the the spercific area 
    K = numpy.matrix(numpy.kaiser(_blk_size, _Kaiser_Beta))
    m_Kaiser = numpy.array(K.T * K) 

    # print m_Kaiser, type(m_Kaiser), m_Kaiser.shape
    # cv2.imshow("Kaisser", m_Kaiser)
    # cv2.waitKey(0)
    # cv2.imwrite("Kaisser.jpg", m_Kaiser.astype(numpy.uint8))
    return m_img, m_wight, m_Kaiser


def Locate_blk(i, j, blk_step, block_Size, width, height):
    # To ensure the location is inside the image
    if i * blk_step + block_Size < width:
        point_x = i * blk_step
    else:
        point_x = width - block_Size

    if j * blk_step + block_Size < height:
        point_y = j * blk_step
    else:
        point_y = height - block_Size

    m_blockPoint = numpy.array((point_x, point_y), dtype=int)  # reference point

    return m_blockPoint


def Define_SearchWindow(_noisyImg, _BlockPoint, _WindowSize, Blk_Size):
    """
    return the coordinate of search window's reference point

    """
    point_x = _BlockPoint[0]  # current corrdinate
    point_y = _BlockPoint[1]  # 

    # get 4 coordinates
    LX = point_x + Blk_Size / 2 - _WindowSize / 2  # up-left x
    LY = point_y + Blk_Size / 2 - _WindowSize / 2  # up-left y
    RX = LX + _WindowSize  # down-right x
    RY = LY + _WindowSize  # down-right y

    # test if outside the image
    if LX < 0:
        LX = 0
    elif RX > _noisyImg.shape[0]:
        LX = _noisyImg.shape[0] - _WindowSize
    if LY < 0:
        LY = 0
    elif RY > _noisyImg.shape[0]:
        LY = _noisyImg.shape[0] - _WindowSize

    return numpy.array((LX, LY), dtype=int)


def Step1_fast_match(_noisyImg, _BlockPoint):
    """fast matching"""
    '''
    * return most similar block including itself
    *_noisyImg
    *_BlockPoint: the corrdinate of current point
    '''
    (present_x, present_y) = _BlockPoint  # current coordinate
    Blk_Size = Step1_Blk_Size
    Search_Step = Step1_Search_Step
    Threshold = Match_threshold_1
    max_matched = Step1_max_matched_cnt
    Window_size = Step1_Search_Window

    blk_positions = numpy.zeros((max_matched, 2), dtype=int)  # to record the location of similar blocks
    Final_similar_blocks = numpy.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)  # to record the result

    img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
    dct_img = cv2.dct(img.astype(numpy.float64))  # dct

    Final_similar_blocks[0, :, :] = dct_img  # sace the blocks
    blk_positions[0, :] = _BlockPoint

    Window_location = Define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size - Blk_Size) / Search_Step  # get the amount of blocks will be found
    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    similar_blocks = numpy.zeros((blk_num ** 2, Blk_Size, Blk_Size), dtype=float)
    m_Blkpositions = numpy.zeros((blk_num ** 2, 2), dtype=int)
    Distances = numpy.zeros(blk_num ** 2, dtype=float)  # record the similarity

    # begin the search in the search area 
    matched_cnt = 0
    for i in range(blk_num):
        for j in range(blk_num):
            tem_img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
            dct_Tem_img = cv2.dct(tem_img.astype(numpy.float64))
            # dct, then l2-norm
            m_Distance = numpy.linalg.norm((dct_img - dct_Tem_img)) ** 2 / (Blk_Size ** 2)

            # to record the blocks
            if m_Distance < Threshold and m_Distance > 0:  
                similar_blocks[matched_cnt, :, :] = dct_Tem_img
                m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                Distances[matched_cnt] = m_Distance
                matched_cnt += 1
            present_y += Search_Step
        present_x += Search_Step
        present_y = Window_location[1]  
    # first matched_cnt blocks
    Distances = Distances[:matched_cnt]
    # sort the block
    Sort = Distances.argsort()

    # count the number of blocks
    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched
    # matched_cnt->Final_similar_blocks, location saved in lk_positions
    if Count > 0:
        for i in range(1, Count):
            Final_similar_blocks[i, :, :] = similar_blocks[Sort[i - 1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i - 1], :]
    return Final_similar_blocks, blk_positions, Count


def Step1_3DFiltering(_similar_blocks):
    '''
    * 3D Filtering
    '''

    statis_nonzero = 0  # non-zero
    m_Shape = _similar_blocks.shape

    for i in range(m_Shape[1]):
        for j in range(m_Shape[2]):
            # print _similar_blocks[:, i, j], type(_similar_blocks[:, i, j])
            tem_Vct_Trans = cv2.dct(_similar_blocks[:, i, j])
            # hard threshold transfer
            tem_Vct_Trans[numpy.abs(tem_Vct_Trans[:]) < Threshold_Hard3D] = 0.
            statis_nonzero += tem_Vct_Trans.nonzero()[0].size
            _similar_blocks[:, i, j] = cv2.idct(tem_Vct_Trans)[0]
    return _similar_blocks, statis_nonzero


def Aggregation_hardthreshold(_similar_blocks, blk_positions, m_basic_img, m_wight_img, _nonzero_num, Count, Kaiser):
    '''
    * weight by number non-zero and put back the weighted blocks
    '''
    _shape = _similar_blocks.shape
    if _nonzero_num < 1:
        _nonzero_num = 1
    block_wight = (1. / (sigma ** 2 * _nonzero_num)) * Kaiser
    for i in range(Count):
        point = blk_positions[i, :]
        tem_img = block_wight * cv2.idct(_similar_blocks[i, :, :])
        m_basic_img[point[0]:point[0] + _shape[1], point[1]:point[1] + _shape[2]] += tem_img
        m_wight_img[point[0]:point[0] + _shape[1], point[1]:point[1] + _shape[2]] += block_wight


def BM3D_step1(_noisyImg):
    """Step 1: Basic"""
    # Initialization
    (width, height) = _noisyImg.shape  # width = row, height = col
    block_Size = Step1_Blk_Size  # block size
    blk_step = Step1_Blk_Step  # step
    Width_num = (width - block_Size) / blk_step
    Height_num = (height - block_Size) / blk_step

    # empty image, empty weight list, Kasier-window
    Basic_img, m_Wight, m_Kaiser = init(_noisyImg, Step1_Blk_Size, Kaiser_Beta)

    # Proceed
    for i in range(int(Width_num + 2)):
        for j in range(int(Height_num + 2)):
            # m_blockPoint reference point: up-left
            m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)  # ensure the point is inside the image
            Similar_Blks, Positions, Count = Step1_fast_match(_noisyImg, m_blockPoint)  # count means the number of similar blocks
            Similar_Blks, statis_nonzero = Step1_3DFiltering(Similar_Blks)  # Blocks group after filtering
            Aggregation_hardthreshold(Similar_Blks, Positions, Basic_img, m_Wight, statis_nonzero, Count, m_Kaiser)
    Basic_img[:, :] /= m_Wight[:, :]
    basic = numpy.matrix(Basic_img, dtype=int)
    basic.astype(numpy.uint8)

    return basic
'''
Color Image BM3D
def BM3D_step1(_noisyImg):
    yImg, uImg, vImg = cv2.split(_noisyImg)
    # yImg.astype(numpy.int)
    # uImg.astype(numpy.int)
    # vImg.astype(numpy.int)
    (width, height) = yImg.shape  # width = row, height = col
    block_Size = Size_Step1_blk  
    blk_step = Step1_Blk_Step  
    Width_num = (width - block_Size) / blk_step
    Height_num = (height - block_Size) / blk_step

    Basic_img, m_Wight, m_Kaiser = init(yImg, Size_Step1_blk, Beta_Kaiser)
    Basic_img_U, m_Wight_U, m_Kaiser_U = init(yImg, Size_Step1_blk, Beta_Kaiser)
    Basic_img_V, m_Wight_V, m_Kaiser_V = init(yImg, Size_Step1_blk, Beta_Kaiser)

    for i in range(int(Width_num + 2)):
        for j in range(int(Height_num + 2)):
            m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)  
            Similar_Blks, Positions, Count = Step1_fast_match(yImg, m_blockPoint) 
            Similar_Blks_U = numpy.zeros(Similar_Blks.shape)
            Similar_Blks_V = numpy.zeros(Similar_Blks.shape)
            for cnt in range(Count):
                Similar_Blks_U[cnt] = uImg[Positions[cnt, 0]: Positions[cnt, 0] + block_Size, Positions[cnt, 1]: Positions[cnt, 1] + block_Size]
                Similar_Blks_V[cnt] = vImg[Positions[cnt, 0]: Positions[cnt, 0] + block_Size, Positions[cnt, 1]: Positions[cnt, 1] + block_Size]
            Similar_Blks, statis_nonzero = Step1_3DFiltering(Similar_Blks) 
            Similar_Blks_U, statis_nonzero_U = Step1_3DFiltering(Similar_Blks_U)  
            Similar_Blks_V, statis_nonzero_V = Step1_3DFiltering(Similar_Blks_V)  
            Aggregation_hardthreshold(Similar_Blks, Positions, Basic_img, m_Wight, statis_nonzero, Count, m_Kaiser)
            Aggregation_hardthreshold(Similar_Blks_U, Positions, Basic_img_U, m_Wight_U, statis_nonzero_U, Count, m_Kaiser_U)
            Aggregation_hardthreshold(Similar_Blks_V, Positions, Basic_img_V, m_Wight_V, statis_nonzero_V, Count, m_Kaiser_V)
    Basic_img[:, :] /= m_Wight[:, :]
    Basic_img_U[:, :] /= m_Wight_U[:, :]
    Basic_img_V[:, :] /= m_Wight_V[:, :]
    basic_Y = numpy.matrix(Basic_img, dtype=int)
    basic_U = numpy.matrix(Basic_img_U, dtype=int)
    basic_V = numpy.matrix(Basic_img_V, dtype=int)
    basic_Y = basic_Y.astype(numpy.uint8)
    basic_U = basic_U.astype(numpy.uint8)
    basic_V = basic_V.astype(numpy.uint8)

    basic = cv2.merge((basic_Y, basic_U, basic_V))

    return basic
'''



def Step2_fast_match(_Basic_img, _noisyImg, _BlockPoint):
    '''
    * block match
    '''
    (present_x, present_y) = _BlockPoint  
    Blk_Size = Step2_Blk_Size
    Threshold = Match_threshold_2
    Search_Step = Step2_Search_Step
    max_matched = Step2_max_matched_cnt
    Window_size = Step2_Search_Window

    blk_positions = numpy.zeros((max_matched, 2), dtype=int)  
    Final_similar_blocks = numpy.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)
    Final_noisy_blocks = numpy.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)

    img = _Basic_img[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
    dct_img = cv2.dct(img.astype(numpy.float32))  
    Final_similar_blocks[0, :, :] = dct_img

    n_img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
    dct_n_img = cv2.dct(n_img.astype(numpy.float32)) 
    Final_noisy_blocks[0, :, :] = dct_n_img

    blk_positions[0, :] = _BlockPoint

    Window_location = Define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size - Blk_Size) / Search_Step  
    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    similar_blocks = numpy.zeros((blk_num ** 2, Blk_Size, Blk_Size), dtype=float)
    m_Blkpositions = numpy.zeros((blk_num ** 2, 2), dtype=int)
    Distances = numpy.zeros(blk_num ** 2, dtype=float)  

    matched_cnt = 0
    for i in range(blk_num):
        for j in range(blk_num):
            tem_img = _Basic_img[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
            # dct_Tem_img = cv2.dct(tem_img.astype(numpy.float32))
            # m_Distance = numpy.linalg.norm((dct_img - dct_Tem_img)) ** 2 / (Blk_Size ** 2)

            m_Distance = numpy.linalg.norm((img - tem_img)) ** 2 / (Blk_Size ** 2)
            
            if m_Distance < Threshold and m_Distance > 0:
                dct_Tem_img = cv2.dct(tem_img.astype(numpy.float32))
                similar_blocks[matched_cnt, :, :] = dct_Tem_img
                m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                Distances[matched_cnt] = m_Distance
                matched_cnt += 1
            present_y += Search_Step
        present_x += Search_Step
        present_y = Window_location[1]
    Distances = Distances[:matched_cnt]
    Sort = Distances.argsort()

    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            Final_similar_blocks[i, :, :] = similar_blocks[Sort[i - 1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i - 1], :]

            (present_x, present_y) = m_Blkpositions[Sort[i - 1], :]
            n_img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
            Final_noisy_blocks[i, :, :] = cv2.dct(n_img.astype(numpy.float64))

    return Final_similar_blocks, Final_noisy_blocks, blk_positions, Count


def Step2_3DFiltering(_Similar_Bscs, _Similar_Imgs):
    '''
    * 3D filtering
    '''
    m_Shape = _Similar_Bscs.shape
    Wiener_wight = numpy.zeros((m_Shape[1], m_Shape[2]), dtype=float)

    for i in range(m_Shape[1]):
        for j in range(m_Shape[2]):
            tem_vector = _Similar_Bscs[:, i, j]
            tem_Vct_Trans = numpy.matrix(cv2.dct(tem_vector))

            Norm_2 = numpy.float64(tem_Vct_Trans.T * tem_Vct_Trans)
            m_weight = Norm_2 / (Norm_2 + sigma ** 2)
            Wiener_wight[i, j] = m_weight

            #if m_weight != 0: Wiener_wight[i, j] = 1. / (m_weight ** 2 * sigma ** 2)
            # else:
            #     Wiener_wight[i, j] = 10000

            # RES=IDCT(WEIGHT(DCT(NOISE_BLOCK)))
            tem_vector = _Similar_Imgs[:, i, j]
            tem_Vct_Trans = m_weight * cv2.dct(tem_vector)
            _Similar_Bscs[:, i, j] = cv2.idct(tem_Vct_Trans)[0]

    return _Similar_Bscs, Wiener_wight


def Aggregation_Wiener(_Similar_Blks, _Wiener_wight, blk_positions, m_basic_img, m_wight_img, Count, Kaiser):
    '''
    * Aggregation proceed
    '''
    _shape = _Similar_Blks.shape
    block_wight = _Wiener_wight * Kaiser

    for i in range(Count):
        point = blk_positions[i, :]
        tem_img = _Wiener_wight * cv2.idct(_Similar_Blks[i, :, :]) * Kaiser
        m_basic_img[point[0]:point[0] + _shape[1], point[1]:point[1] + _shape[2]] += tem_img
        m_wight_img[point[0]:point[0] + _shape[1], point[1]:point[1] + _shape[2]] += block_wight


def BM3D_2nd_step(_basicImg, _noisyImg):
    '''Step 2. Final estimate '''
    
    (width, height) = _noisyImg.shape
    block_Size = Step2_Blk_Size
    blk_step = Step2_Blk_Step
    Width_num = (width - block_Size) / blk_step
    Height_num = (height - block_Size) / blk_step

    m_img, m_Wight, m_Kaiser = init(_noisyImg, block_Size, Kaiser_Beta)

    for i in range(int(Width_num + 2)):
        for j in range(int(Height_num + 2)):
            m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)
            Similar_Blks, Similar_Imgs, Positions, Count = Step2_fast_match(_basicImg, _noisyImg, m_blockPoint)
            Similar_Blks, Wiener_wight = Step2_3DFiltering(Similar_Blks, Similar_Imgs)
            Aggregation_Wiener(Similar_Blks, Wiener_wight, Positions, m_img, m_Wight, Count, m_Kaiser)
    m_img[:, :] /= m_Wight[:, :]
    Final = numpy.matrix(m_img, dtype=int)
    Final.astype(numpy.uint8)

    return Final

# add noise:
def Gauss_noise(img, sigma=25):
    noise = numpy.matlib.randn(img.shape) * sigma
    res = img + noise
    return res


# psnr
def PSNR(img1, img2):
    D = numpy.array(img1 - img2, dtype=numpy.int64)
    D[:, :] = D[:, :] ** 2
    RMSE = D.sum() / img1.size
    psnr = 10 * math.log10(float(255. ** 2) / RMSE)
    return psnr

'''
# Color Image BM3D
def Gauss_noise_color(img, sigma=0.001):
    mean = 0
    image = numpy.array(img / 255, dtype=float)
    noise = numpy.random.normal(mean, sigma ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = numpy.clip(out, low_clip, 1.0)
    out = numpy.uint8(out * 255)
    return out

def PSNR(img1, img2):
    D = numpy.array(numpy.int64(img1) - numpy.int64(img2), dtype=numpy.int64)
    print(D)
    D[:, :, :] = D[:, :, :] ** 2
    RMSE = D.sum() / img1.size
    psnr = 10 * math.log10(float(255. ** 2) / RMSE)
    return psnr
'''

if __name__ == '__main__':
    cv2.setUseOptimized(True)  
    img_name = "images/eGrass_xs.jpg"
    ori = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE) 
    cv2.imwrite("results/ori.jpg", ori)
    img = Gauss_noise(ori)
    cv2.imwrite("results/noise.jpg", img)

    print('The PSNR After adding noise %f' % PSNR(ori, img))

    e1 = cv2.getTickCount()  

    Basic_img = BM3D_step1(img)
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()  
    print ("The Processing time of the Step 1 is %f s" % time)
    cv2.imwrite("results/Basic3.jpg", Basic_img)

    print ("The PSNR between the two img of the First step is %f" % PSNR(ori, Basic_img))

    Final_img = BM3D_2nd_step(Basic_img, img)
    e3 = cv2.getTickCount()
    time = (e3 - e2) / cv2.getTickFrequency()
    print ("The Processing time of the Step 2 is %f s" % time)
    cv2.imwrite("results/Final3.jpg", Final_img)

    print ("The PSNR between the two img of the Second step is %f" % PSNR(ori, Final_img))
    time = (e3 - e1) / cv2.getTickFrequency()
    print ("The total Processing time is %f s" % time)

'''
Color Image BM3D
'''
# if __name__ == '__main__':
#     cv2.setUseOptimized(True)  
#     img_name = "images/eGrass.jpg"  
#     ori = cv2.imread(img_name,1) 
#     cv2.imwrite("results/ori.jpg", ori)
#     # print(ori[:,:,0])
#     # print("-----------------------")
#     oriYUV = cv2.cvtColor(ori, cv2.COLOR_BGR2YUV)
#     yImg, uImg, vImg = cv2.split(oriYUV)

#     noiBGR = Gauss_noise_color(ori)
#     # print(noiBGR[:,:,0])
#     noiYUV = cv2.cvtColor(noiBGR,cv2.COLOR_BGR2YUV)
#     cv2.imwrite("results/noise.jpg", noiBGR)
#     # cv2.imshow('noiYUV-Y',noiYUV[:,:,0])
#     # cv2.waitKey(0)
#     # print("-----------------------")
#     # print(noiBGR[:,:,0]-ori[:,:,0])
#     # print(noiYUV[:,:,0]-oriYUV[:,:,0])

#     print('The PSNR After add noise %f' % PSNR(ori, noiBGR))
#     e1 = cv2.getTickCount() 
#     Basic_img = BM3D_step1(noiYUV)
#     e2 = cv2.getTickCount()
#     time = (e2 - e1) / cv2.getTickFrequency() 
#     Basic_img_test = cv2.cvtColor(Basic_img, cv2.COLOR_YUV2BGR)
#     cv2.imwrite("results/Basic_test.jpg", Basic_img_test)