# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Use a breakpoint in the code line below to debug your script.
# Press Ctrl+F8 to toggle the breakpoint.


from CropUtil import CropUtil

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cropper = CropUtil(3)
    # Python list of images
    image_list = cropper.capture()
    print(len(image_list))
