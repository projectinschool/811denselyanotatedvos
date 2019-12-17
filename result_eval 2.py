import os
import numpy as np
import cv2
import PIL.Image
import pickle
import matplotlib.pyplot
import pdb
import argparse
import textwrap

from eval_segm import pixel_accuracy
from eval_segm import mean_IU
from eval_segm import mean_accuracy
from eval_segm import frequency_weighted_IU

def db_eval_iou(annotation,segmentation):

	""" Compute region similarity as the Jaccard Index.
	Arguments:
		annotation   (ndarray): binary annotation   map.
		segmentation (ndarray): binary segmentation map.
	Return:
		jaccard (float): region similarity
 """

	annotation   = annotation.astype(np.bool)
	segmentation = segmentation.astype(np.bool)

	if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
		return 1
	else:
		return np.sum((annotation & segmentation)) / \
				np.sum((annotation | segmentation),dtype=np.float32)

class ImageConverter(object):
    image_base_path = "../image/"
    data_base_path = "../data/"

    def image_to_array(self, image_folder_path):
        """
        from image to binary file
        """
        img = cv2.imread(image_folder_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY)[1]
        return th
        # test = np.array([[0,0,0,0,0], [0,0,0,0,0]])
        # return test



def main():
    parser = \
        argparse.ArgumentParser(
            prog='parser',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''
    **************************************************************
                       Result Evaluation
                    -----------------------
    The path to Result and Reference Pictures is user-defined path.
    **************************************************************
                                     ''')
                                     )

    parser.add_argument('--result_image_path', type=str,
                        help='the path to the result images')
    parser.add_argument('--ref_image_path', type=str,
                        help='the path to the davis reference path')

    args = parser.parse_args()

    result_image_path = args.result_image_path
    if not  result_image_path:
        raise Exception("No result image path is given neither in "
                        "command line nor environment arguements")

    ref_image_path = args.ref_image_path
    if not ref_image_path:
        raise Exception("No reference images path is given neither in "
                        "command line nor environment arguements")

    result_imgs = []
    ref_imgs = []
    for a, b, result_files in os.walk(result_image_path):
        result_imgs = result_files

    for a, b, ref_files in os.walk(ref_image_path):
        ref_imgs = ref_files


   # if len(result_imgs) != len(ref_imgs):
    #    raise Exception("The number of image in result and ref is not match. Cannot evaluate.")

    print("+", "-".center(85, '-'), "+")
    print("|",  "pixel_accuracy".center(20), "mean_IU".center(20),
          "mean_accuracy".center(20), "frequency_weighted_IU".center(20), "|".rjust(2))
    result_image_filenames = os.listdir(result_image_path)
    ref_image_filenames = os.listdir(ref_image_path)

    image_converter = ImageConverter()
    each_pixel_accuracy = []
    each_mean_IU = []
    each_mean_accuracy = []
    each_frequency_weighted_IU = []
    for i in range(len(result_imgs)):
        print("filename:", result_image_filenames[i])
        result_imgs[i] = image_converter.image_to_array(os.path.join(result_image_path, result_image_filenames[i]))
        ref_imgs[i] = image_converter.image_to_array(os.path.join(ref_image_path, ref_image_filenames[i]))

        i_pixel_accuracy = pixel_accuracy(result_imgs[i], ref_imgs[i])
        //i_mean_IU = mean_IU(result_imgs[i], ref_imgs[i])
        i_mean_IU = db_eval_iou(ref_imgs[i], result_imgs[i])
        i_mean_accuracy = mean_accuracy(result_imgs[i], ref_imgs[i])
        i_frequency_weighted_IU = frequency_weighted_IU(result_imgs[i], ref_imgs[i])

        each_pixel_accuracy.append(i_pixel_accuracy)
        each_mean_IU.append(i_mean_IU)
        each_mean_accuracy.append(i_mean_accuracy)
        each_frequency_weighted_IU.append(i_frequency_weighted_IU)

        print("|",
        str(i_pixel_accuracy).center(20), str(i_mean_IU).center(20),
              str(i_mean_accuracy).center(20), str(i_frequency_weighted_IU).center(20), "|".rjust(2))

    print("| mean value:",
          str(np.mean(each_pixel_accuracy)).center(12),
          str(np.mean(each_mean_IU)).center(20),
          str(np.mean(each_mean_accuracy)).center(20),
          str(np.mean(each_frequency_weighted_IU)).center(20),"|".rjust(2))
    print("+", '-'.center(85, '-'), "+")


if __name__ == "__main__":
    main()
