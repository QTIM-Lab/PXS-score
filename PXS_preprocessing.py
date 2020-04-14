'''
Processing DICOMs
'''

import os
import pydicom
import numpy as np
import cv2

working_path = '/home/PXS_score/'
os.chdir(working_path)

### Code from Jeremy Irvin, author of CheXpert ###

def dcm2img(dcm_file_path):
    """Extract the image from a path to a DICOM file."""

    # Read the DICOM and extract the image.
    dcm_file = pydicom.dcmread(dcm_file_path)
    raw_image = dcm_file.pixel_array

    assert len(raw_image.shape) == 2,\
        "Expecting single channel (grayscale) image."

    # The DICOM standard specifies that you cannot make assumptions about
    # unused bits in the representation of images, see Chapter 8, 8.1.1, Note 4:
    # http://dicom.nema.org/medical/dicom/current/output/html/part05.html#chapter_8
    # pydicom doesn’t exclude those unused bits by default, so we need to mask them
    raw_image = np.bitwise_and(raw_image, (2 ** (dcm_file.HighBit + 1) -
                                           2 ** (dcm_file.HighBit -
                                                 dcm_file.BitsStored + 1)))

    # Normalize pixels to be in [0, 255].
    raw_image = raw_image - raw_image.min()
    normalized_image = raw_image / raw_image.max()
    rescaled_image = (normalized_image * 255).astype(np.uint8)

    # Correct image inversion.
    if dcm_file.PhotometricInterpretation == "MONOCHROME1":
       rescaled_image = cv2.bitwise_not(rescaled_image)

    # Perform histogram equalization.
    adjusted_image = cv2.equalizeHist(rescaled_image)

    return adjusted_image

image_path = '' # DICOM image path
img_file = dcm2img(image_path)
cv2.imwrite(image_path, img_file, [cv2.IMWRITE_JPEG_QUALITY, 95])
