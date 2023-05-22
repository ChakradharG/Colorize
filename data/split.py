from os import listdir
from sklearn.model_selection import train_test_split


def main(root):
	files = listdir(root)
	train_files, test_files = train_test_split(files, test_size=0.2)

	with open('data/train.txt', 'w') as f:
		for file in train_files:
			f.write(file + '\n')

	with open('data/test.txt', 'w') as f:
		for file in test_files:
			f.write(file + '\n')


main(root='./data/images/')
