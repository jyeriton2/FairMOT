# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

import torch
from DCN import DCN


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    test_input = torch.rand(2,64,128,128).cuda() # # of data, channels, w, h
    model = DCN(64,64,kernel_size=3,stride=1,padding=1).cuda()    #input channels , output channels, kw & kh, padding
    out = model(test_input)
    print(out.shape)
    



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
