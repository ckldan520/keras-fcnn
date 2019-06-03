import cv2
import numpy as np


def get_data(input_path):
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    # 1li  0oO Ww cC Zz pP Ss Vv Uu Xx
    char_set = dict(
        [('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8), ('9', 9),
         ('a', 10), ('b', 11), ('c', 12), ('d', 13), ('e', 14), ('f', 15), ('g', 16), ('h', 17), ('j', 18),
         ('k', 19), ('m', 20), ('n', 21), ('p', 22), ('q', 23), ('r', 24), ('s', 25), ('t', 26), ('u', 27),
         ('v', 28), ('w', 29), ('x', 30), ('y', 31), ('z', 32), ('A', 33), ('B', 34), ('D', 35), ('E', 36),
         ('F', 37), ('G', 38), ('H', 39), ('I', 40), ('J', 41), ('K', 42), ('L', 43), ('M', 44), ('N', 45),
         ('Q', 46), ('R', 47), ('T', 48), ('Y', 49)]
    )

    visualise = True

    with open(input_path,'r') as f:

        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            (filename,x1,y1,x2,y2,class_name) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = char_set[class_name]

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = cv2.imread(filename)
                (rows,cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0,6) > 0:
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)), 'y2': int(float(y2))})


        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping


