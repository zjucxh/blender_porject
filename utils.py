import cv2 as cv
import numpy as np
import h5py

def h5py2img(h5file, img_name):
    """
    Convert an HDF5 dataset to an image.
    
    Parameters:
    h5file (str): Path to the HDF5 file.
    dataset_name (str): Name of the dataset within the HDF5 file.
    
    Returns:
    np.ndarray: The image data as a NumPy array.
    """
    with h5py.File(h5file, 'r') as f:
        img_data = f[img_name][:]
        print(' img shape : {0}'.format(img_data.shape))
        return img_data
    

if __name__ == "__main__":
    # Example usage
    h5file = "output/output.hdf5/0.hdf5"
    img_name = "colors"
    
    img_data = h5py2img(h5file, img_name)

    # visualize the image
    cv.imshow('Image', img_data)
    cv.waitKey(0)
    cv.destroyAllWindows()