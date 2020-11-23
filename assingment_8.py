import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Part (i)
#Part A
def single_convolution(input, kernel, size):
    result = 0
    for a in range(0, size[0]):
        for  b in range(0, size[1]):
            result = result + input[a][b] * kernel[a][b]
    return result

def return_array_section(array, starting_position, size):
    temp_array = []
    for a in range(starting_position[0], starting_position[0]+size[0]):
        temp_row = []
        for b in range(starting_position[1], starting_position[1]+size[1]):
            temp_row.append(array[a][b])

        temp_array.append(temp_row)
    return temp_array

def convolve(input_array, kernel):
    input_array_size = [len(input_array), len(input_array[0])]
    kernel_size = [len(kernel), len(kernel[0])]
    output_array = []
    current_position = [0, 0]
    for a in range(0, (input_array_size[0]-kernel_size[0])+1):
        new_row = []
        for b in range(0, (input_array_size[1]-kernel_size[1]) + 1):
            temp_input = return_array_section(input_array, current_position, kernel_size)
            new_value = single_convolution(temp_input, kernel, kernel_size)
            new_row.append(new_value)
            current_position[1] = current_position[1] + 1 

        output_array.append(new_row)
        current_position[0] = current_position[0] + 1
        current_position[1] = 0 
    return output_array

# test_array_1 = [[1,2,3,4,1,4,3], [5,6,7,8,4,5,6], [4,5,6,7,7,8,5], [0,0,1,1,4,5,6], [1,4,3,67,4,2,7]]
# test_kernel_1 = [[1,1,1], [0,0,0], [-1,-1,-1]]
# result = convolve(test_array_1, test_kernel_1)
# print("convolution result", result)

#Part B
from PIL import Image
im = Image.open('triangle.png')
rgb = np.array(im.convert('RGB'))
test = rgb[:, :, 0]
Image.fromarray(np.uint8(test)).show()

kernel1 = [[-1,-1,-1], [-1,8,-1], [-1,-1,-1]]
kernel2 = [[0,-1,0], [-1,8,-1], [0,-1,0]]

kernel1_result = convolve(test, kernel1)
kernel2_result = convolve(test, kernel2)

Image.fromarray(np.uint8(kernel1_result)).show()
Image.fromarray(np.uint8(kernel2_result)).show()