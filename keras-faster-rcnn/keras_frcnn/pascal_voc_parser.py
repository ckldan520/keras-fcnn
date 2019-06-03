import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
def get_data(input_path):
	all_imgs = []

	classes_count = {}

	class_mapping = {}

	visualise = False

	# 1li  0oO Ww cC Zz pP Ss Vv Uu Xx
	char_set = dict(
		[('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8), ('9', 9)]
		 # ('a', 10), ('b', 11), ('c', 12), ('d', 13), ('e', 14), ('f', 15), ('g', 16), ('h', 17), ('j', 18),
		 # ('k', 19), ('m', 20), ('n', 21), ('p', 22), ('q', 23), ('r', 24), ('s', 25), ('t', 26), ('u', 27),
		 # ('v', 28), ('w', 29), ('x', 30), ('y', 31), ('z', 32), ('A', 33), ('B', 34), ('D', 35), ('E', 36),
		 # ('F', 37), ('G', 38), ('H', 39), ('I', 40), ('J', 41), ('K', 42), ('L', 43), ('M', 44), ('N', 45),
		 # ('Q', 46), ('R', 47), ('T', 48), ('Y', 49)]
	)

	for idx in char_set:
		class_mapping[idx] = char_set[idx]

	#	data_paths = [os.path.join(input_path,s) for s in ['VOC2007', 'VOC2012']]
	data_paths = [os.path.join(input_path, s) for s in ['VOC2012']]

	print('Parsing annotation files')

	for data_path in data_paths:

		annot_path = os.path.join(data_path, 'Annotations')
		#imgs_path = os.path.join(data_path, 'JPEGImages')
		imgs_path = os.path.join(data_path, 'captcha_img')
		imgsets_path_trainval = os.path.join(data_path, 'ImageSets','Main','trainval.txt')
		imgsets_path_test = os.path.join(data_path, 'ImageSets','Main','test.txt')

		trainval_files = []
		test_files = []
		try:
			with open(imgsets_path_trainval) as f:
				for line in f:
					trainval_files.append(line.strip() + '.jpg')
		except Exception as e:
			print(e)

		try:
			with open(imgsets_path_test) as f:
				for line in f:
					test_files.append(line.strip() + '.jpg')
		except Exception as e:
			if data_path[-7:] == 'VOC2012':
				# this is expected, most pascal voc distibutions dont have the test.txt file
				pass
			else:
				print(e)

		annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
		idx = 0
		for annot in annots:
			try:
				idx += 1

				et = ET.parse(annot)
				element = et.getroot()

				element_objs = element.findall('object')
				element_filename = element.find('filename').text
	#tiansheng
				element_filename =  element_filename  #
				element_width = int(element.find('size').find('width').text)
				element_height = int(element.find('size').find('height').text)

				if len(element_objs) > 0:
					annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,
									   'height': element_height, 'bboxes': []}

					if element_filename in trainval_files:
						annotation_data['imageset'] = 'trainval'
					elif element_filename in test_files:
						annotation_data['imageset'] = 'test'
					else:
						continue
#						annotation_data['imageset'] = 'trainval'

				for element_obj in element_objs:
					class_name = element_obj.find('name').text

					# 1li  0oO Ww cC Zz pP Ss Vv Uu Xx
					if class_name == 'l' or class_name == 'i':
						class_name = '1'
					elif class_name == 'o' or class_name == 'O':
						class_name = '0'
					elif class_name == 'W':
						class_name = 'w'
					elif class_name == 'C':
						class_name = 'c'
					elif class_name == 'Z':
						class_name = 'z'
					elif class_name == 'P':
						class_name = 'p'
					elif class_name == 'S':
						class_name = 's'
					elif class_name == 'V':
						class_name = 'v'
					elif class_name == 'U':
						class_name = 'u'
					elif class_name == 'X':
						class_name = 'x'

					if class_name not in classes_count:
						classes_count[class_name] = 1
					else:
						classes_count[class_name] += 1

					if class_name not in class_mapping:
						class_mapping[class_name] = char_set[class_name]

					obj_bbox = element_obj.find('bndbox')
					x1 = int(round(float(obj_bbox.find('xmin').text)))
					y1 = int(round(float(obj_bbox.find('ymin').text)))
					x2 = int(round(float(obj_bbox.find('xmax').text)))
					y2 = int(round(float(obj_bbox.find('ymax').text)))
					difficulty = int(element_obj.find('difficult').text) == 1
					annotation_data['bboxes'].append(
						{'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
				all_imgs.append(annotation_data)

				if visualise:
					img = cv2.imread(annotation_data['filepath'])
					for bbox in annotation_data['bboxes']:
						cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
									  'x2'], bbox['y2']), (0, 0, 255))
					cv2.imshow('img', img)
					cv2.waitKey(0)

			except Exception as e:
				print(e)
				continue
	return all_imgs, classes_count, class_mapping
