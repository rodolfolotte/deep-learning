import cv2, os, sys
import numpy as np

from os.path import basename

# USAGE:
# python unifyColor.py PATH_TO_ANNOTATION PATH_TO_OUTPUT_FOLDER
#
# EXAMPLE:
# python unifyColor.py /home/lotte/Desktop/encp/labels/ /home/lotte/Desktop/encp/labels-unified/

def main():
    classes_colors =  [[0, 0, 0], #background
                      [0, 0, 255], #roof
                      [128, 255, 255], #sky
                      [255, 255, 0], #wall
                      [128, 0, 255], #balcony
                      [255, 0, 0], #window
                      [255, 128, 0], #door
                      [0, 255, 0]] #shop

    # "balcony" : {[85,255,170], [170,255,85]},
    # "window" : {[0, 85, 255], [255, 255, 0]},
    # "wall" : {[0, 0, 255], [0, 255, 255]},
    # "door" : {[0, 170, 255], [170, 0, 0]},

    files = os.listdir(sys.argv[1])

    for file in files:

        path = sys.argv[1] + file
        name, file_extension = os.path.splitext(file)
        output = sys.argv[2] + name + ".png"

        if(os.path.isfile(path)):

          print("Unifying facade color classes...")
          image = cv2.imread(path)
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

          rows, cols, bands = image.shape
          im = np.zeros((rows, cols, bands), dtype=np.uint8)

          for i in range(rows):
            for j in range(cols):
              color = image[i, j]
              color = map(int, color)

              # window
              if (color == [0, 0, 128]):
                color_code = classes_colors[5]
              # door
              elif (color == [128, 128, 0]):
                color_code = classes_colors[6]
              # wall
              elif (color == [128, 0, 0]):
                color_code = classes_colors[3]
              # something else
              else:
                color_code = classes_colors[0]


              im[i][j] = color_code

        else:
            print(file + " is not a valid file. Check and try again!")

        cv2.imwrite(output, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        print("Annotated image with unified classes saved as " + output + "\n")


if __name__ == '__main__':
    main()