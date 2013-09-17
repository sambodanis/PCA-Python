import numpy, Image, random
from scipy import linalg, ndimage
from scipy.misc import toimage, imsave
from matplotlib import pyplot
import PCA

image_edge_length = 512
num_images = 1
subsample_edge_length = 128


def load_images(filename):
    binary_all_images = numpy.fromfile(filename, dtype=float)
    # image_edge_length = 512
    # num_images = 10
    all_numbers = []
    array_position = 0
    for k in range(0, num_images):
        matrix = numpy.zeros((image_edge_length, image_edge_length), numpy.float64)    
        for i in range(0, image_edge_length):
            for j in range(0, image_edge_length):
                matrix[i][j] = binary_all_images[array_position]
                array_position += 1
        all_numbers.append(matrix)
    toimage(all_numbers[0]).show()

    return all_numbers

def get_subsamples(image_array):
    result = []
    for i in range(0, 30):
        k = random.randint(0, num_images - 1)
        curr_matrix = image_array[k]
        m, n = curr_matrix.shape
        x = random.randint(0, m - subsample_edge_length)
        y = random.randint(0, n - subsample_edge_length)
        submatrix = curr_matrix[x : x + subsample_edge_length, y : y + subsample_edge_length]
        result.append(submatrix)
    return result

def main():
    # image_array = load_images('olsh.dat')
    # im = Image.open("test_image_1.png")
    gray = ndimage.imread("test_image_1.png", flatten=True)
    # toimage(gray).show()

    subsamples = get_subsamples([gray])    

    for i, sample in enumerate(subsamples):
        pca = PCA.PCA(sample, 90)
        compressed = toimage(pca.get_compressed_matrix())
        filename = 'pca_images/im_ex' + str(i) + '.png'
        print filename
        imsave(filename, compressed)
        toimage(compressed).show()

    # pca = PCA.PCA(gray, 98)
    # image_compressed = toimage(pca.get_compressed_matrix())
    # imsave('pca_images/im_ex1.png', image_compressed)
    # toimage(image_compressed).show()



    # training_examples = get_subsamples(image_array)
    # compressed_arr = []
    # for example in training_examples:
    #     pca = PCA.PCA(example)
    #     compressed_arr.append(pca.get_compressed_matrix())
    # toimage(compressed_arr[0]).show()
    
    
    # matrix = numpy.array([[1, 2, 3], [4, 5, 6]])
    # pca = PCA.PCA(matrix)
    # print pca.get_compressed_matrix()[0:1][0:2]
    # print pca.get_principal_components()
    


if __name__ == '__main__':
    main()
    