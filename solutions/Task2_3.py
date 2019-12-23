# linear algebra
import numpy as np
# data processing, CSV file I/O
import pandas as pd
from imageio import imread

# Input data files are available in the "../input/" directory.
import matplotlib.pyplot as plt
import os

# Here you should specify your input path. For me it was the following:
INPUT_PATH = r'C:/Users/yarik/Downloads/input'
DATA_PATH = INPUT_PATH
# Here's specified train data path
TRAIN_DATA = os.path.join(DATA_PATH, "train_v2")
# Here you can find some test data
TEST_DATA = os.path.join(DATA_PATH, "test_v2")
path_train = r'C:/Users/yarik/Downloads/input/train_v2/'
path_test = r'C:/Users/yarik/Downloads/input/test_v2/'
path_submission = r'C:/Users/yarik/Downloads/input/sample_submission_v2.csv'
# number of images to choose
IMG_NUM = 10
# Printing what we have in our input folder:
print(os.listdir(INPUT_PATH))
# Any results you write to the current directory are saved as output.
'''
train = os.listdir(path_train)
print('train data', len(train), 'images')
test = os.listdir(path_test)
print('test data', len(test), 'images')
'''
# Our submission file will reside in 'submission' variable
submission = pd.read_csv(path_submission)
# After choosing image, we will make plot
def make_plt(ImageId):
    img = imread(path_train + ImageId)
    img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768))
    for mask in img_masks:
        try:
            # decoding that masks with special function
            all_masks += rle_decode(mask)
            # if there are no ships on a mask Exception will be thrown and handled
        except Exception:
            print('No ships on that picture! Choose another one')
            print('Please, choose picture:')
    # building plots with image, mask and image with mask
    fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(all_masks)
    axarr[2].imshow(img)
    # here we're adding mask to the trird image
    axarr[2].imshow(all_masks, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.show()
# here we decode our mask on an image
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    # if there are no ships 'NoShipsException' will be thrown
    if(str(mask_rle) == 'nan'):
        raise Exception('No ships on that picture! Choose another one')
    # Splitting this mask so we can fulfill it with another color
    s = mask_rle.split()
    # defining whether it's a start of pixels or length
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    # defining ends of each such pixel stick
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    # now starting from start and ending in
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

masks = pd.read_csv(r'C:/Users/yarik/Downloads/input/train_ship_segmentations_v2.csv')

print('-------------- Program made by Yaroslav Oliinyk --------------')
print('inspired by user @inversion at Kaggle')
print('Image classifier. The main task of the program is to classify the ships')
# listing all possible images
img_lst = os.listdir(path_train)
while(True):
    print('Choose image:')
    index = 1
    # We chose IMG_NUM first images
    for i in range(IMG_NUM):
        # Listing them
        print('\t', index, '.', img_lst[i])
        index += 1
    # Putting button for quit
    print(IMG_NUM+1, '.', 'Quit')
    chs = input()
    try:
        chs = int(chs)
        chs -= 1
        # If the number you entered not in required bounds, you'll get an exception
        if(chs not in range(IMG_NUM+2)):
            raise Exception
    except Exception:
        print('Enter valid number')
        continue
    if(chs == IMG_NUM):
        break
    ImageId = img_lst[chs]
    make_plt(ImageId)


