import cv2, os, sys
import numpy as np

from os.path import basename

# USAGE:
# python txt2img.py PATH_TO_TXT_FOLDER PATH_TO_OUTPUT_FOLDER
#
# EXAMPLE:
# python txt2img.py /home/lotte/Desktop/encp/labels/ /home/lotte/Desktop/encp/annotations/

def nr_of_lines(full_path):
  """ Count number of lines in a file."""
  f = open(full_path)
  lines = sum(1 for line in f)
  f.close()
  return lines


def main():
    classes_colors =  [[255, 128, 0],
                   [0, 255, 0],
                   [128, 0, 255],
                   [255, 0, 0],
                   [255, 255, 0],
                   [128, 255, 255],
                   [0, 0, 255]]

    files = os.listdir(sys.argv[1])

    for file in files:

        path = sys.argv[1] + file
        name, file_extension = os.path.splitext(file)
        output = sys.argv[2] + name + ".png"

        if(os.path.isfile(path)):

            print("Traslating txt file " + file + " in a label png raster...")
            i = 0
            j = 0
            n_lines = nr_of_lines(path)

            f = open(path, "r")

            l = f.readline();
            fields = l.split(" ");
            im = np.zeros((n_lines, len(fields), 3), dtype=np.uint8)

            for line in f:
                if not line:
                    continue

                fields = line.split(" ");

                for field in fields:
                    value = int(field)

                    if value is -1:
                        color_code = [0, 0, 0]
                    else:
                        color_code = classes_colors[value]

                    im[i][j] = color_code
                    j = j + 1

                j = 0
                i = i + 1

            f.close()
        else:
            print(file + " is not a valid file. Check and try again!")

        cv2.imwrite(output, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        print("Labelled raster saved as " + output + "\n")


if __name__ == '__main__':
    main()