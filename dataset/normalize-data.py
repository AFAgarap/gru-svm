import csv
import os

def main():
	#
	

def linear_scale(min, max, x):
	'''Scales integer values [min, max] -> [0.0, 1.0]'''
	if (x >= min and x <= max):
		scaled_value = (((1.0 - 0.0) * (x - min)) / (max - min)) + 0.0
	else:
		scaled_value = -1
	return scaled_value

if __name__ == '__main__':
	main()