from contextlib import closing
from glob import glob
from multiprocessing import cpu_count, Pool
from os.path import join
from pathlib import Path
from typing import List, Tuple
import math

import numpy as np
import cv2
import click


def simple_threshold(image: np.ndarray, threshold: int = 90, invert: bool = False) -> np.ndarray:
	if threshold < 0 or threshold > 255:
		raise ValueError(f"Threshold must be between 0 and 255")

	_, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY)
	return image


def otsu_threshold(image: np.ndarray) -> np.ndarray:
	# image = cv2.createCLAHE(clipLimit=3).apply(image)
	_, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return image


def find_contours(threshold_image: np.ndarray, color_image: np.ndarray) -> (np.ndarray, List[Tuple]):
	contours, hierarchy = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours_image = color_image.copy()
	i = 0
	min_area = 10000
	max_area = 200000
	
	for contour in contours:
		i += 1
		cnt = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True)
		bounding_rect = cv2.boundingRect(cnt)
		(x, y, w, h) = bounding_rect
		min_rect = cv2.minAreaRect(cnt)
		area = cv2.contourArea(contour)
		rect_area = w * h

		# if contour is the right size, draw and label it
		if max_area > area > min_area:
			cv2.drawContours(contours_image, [contour], 0, (0, 255, 0), 3)
			cv2.putText(contours_image, str(i), (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

	return contours_image


@click.group()
def cli():
	pass


@cli.command()
@click.argument('input_file')
@click.option('-o', '--output_directory', required=False, type=str, default='')
def segment(input_file, output_directory):
	Path(output_directory).mkdir(parents=True, exist_ok=True)
	color = cv2.imread(input_file)
	gray = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
	
	# apply threshold
	thresh = otsu_threshold(gray)
	cv2.imwrite(join(output_directory, "thresh.png"), thresh)

	# close image
	kernel = np.ones((7, 7), np.uint8)
	dilated = cv2.dilate(thresh, kernel, iterations=1)
	eroded = cv2.erode(dilated, kernel, iterations=1)
	closed = cv2.morphologyEx(dilated.copy(), cv2.MORPH_CLOSE, kernel)
	cv2.imwrite(join(output_directory, "dilated.png"), dilated)
	cv2.imwrite(join(output_directory, "eroded.png"), eroded)
	cv2.imwrite(join(output_directory, "closed.png"), closed)
	
	# find contours
	contours = find_contours(closed, color)
	cv2.imwrite(join(output_directory, "contours.png"), contours)


if __name__ == '__main__':
	cli()
