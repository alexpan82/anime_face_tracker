import argparse
import cv2
import glob
import socket
import sys
import random
import os
from shutil import copy2

# Returns a numpy file of user inputted numbers associated with each jpg image feature vector.
# Addition returns a csv file of user inputted number associated with each jpg file name
# Also marks rectangles around faces in the jpg images
# filelist is a list of jpg images in a directory


def findFace(flist, counter, path):
	# Write annotations to csv file
	out_path = os.path.join(path, 'data_out.csv')
	outfilecsv = open(out_path, 'w')

	# counter: a counter to tell server how many connections left to send

	for jpgfile in flist:
		img = cv2.imread(jpgfile)

		# Connect to TCP/IP socket
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		
		# Bind the socket to the port
		server_address = ('localhost', 10000)
		print(sys.stderr, 'starting up on %s port %s' % server_address)
		sock.connect(server_address)

		try:
			# Send data to server
			# Send the classification history and counter as strings
			decodedmessage = str(counter)
			counter -= 1
			message = decodedmessage.encode()
			# 1 send message
			sock.sendall(message)

			# Make sure the message got thru!
			while True:
				print('confirming')
				# 4 Receive message from server and check whether is matches up
				confirm = sock.recv(16)
				if confirm != message:
					print(confirm.decode(), message.decode())
					sock.sendall(message)
				else:
					print('saved')
					sock.sendall('True'.encode())
					break
			
			# Show image
			cv2.imshow('hi', img)
			cv2.waitKey(20)

			# Receive user input on server and go to the next image
			data = sock.recv(16).decode()

			# write to file
			outfilecsv.write('%s,%s\n' % (jpgfile, data))

		finally:
			print(sys.stderr, 'closing socket')
			sock.close()


# Make this directory structure
'''
main_directory/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
.....b_image_1.jpg
......b_image_2.jpg
'''


def organize(path):
	# Read from csv file
	csv_path = os.path.join(path, 'data_out.csv')
	out_path = os.path.join(path, 'annotated')
	if os.path.exists(out_path) is False:
		os.mkdir(out_path)

	print('Reading from ', csv_path)
	print('Outputting to ', out_path)
	with open(csv_path, 'r') as f:
		for line in f:
			line = line.strip().split(',')
			category = line[1]
			png = line[0].split('/')[-1]
			category_path = os.path.join(path, 'annotated', category)
			png_path = os.path.join(path, 'cropped_faces', png)
			# mkdir
			if os.path.exists(category_path) is False:
				os.mkdir(category_path)
			# copy png file to subdirectory
			copy2(png_path, os.path.join(category_path, png))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Annotates pics in the /path/to/output/cropped_faces directory')
	parser.add_argument("command",
						metavar="<command>",
						help="Please choose 'annotate' or 'organize'")
	parser.add_argument('--output', required=True,
						metavar="/path/to/output",
						help='Directory of all output data. I recommend to create a new directory for each anime')
	args = parser.parse_args()

	path = args.output
	if args.command == 'annotate':
		# Import picture paths in folder to list in a random order
		cropped_path = os.path.join(path, 'cropped_faces')
		print('Reading from ', cropped_path)
		filelist = glob.glob(os.path.join(cropped_path, '*png'))
		random.shuffle(filelist)
		findFace(filelist, len(filelist), path)

	elif args.command == 'organize':
		# Now organize cropped pngs
		organize(path)

	else:
		print("Please choose 'annotate' or 'organize'")
		quit()
