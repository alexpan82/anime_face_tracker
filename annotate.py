from sys import argv
import cv2
import glob
import socket
import sys
import random
import os

# Returns a numpy file of user inputed numbers associated with each jpg image feature vector.
# Addition returns a csv file of user inputed number associated with each jpg file name
# Also marks rectangles around faces in the jpg images
# filelist is a list of jpg images in a directory


def findFace(flist, counter):
	# Write annotations to csv file
	outfilecsv = open('data_out.csv', 'w')

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


# Make this directory struc
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
	with open('data_out.csv', 'r') as f:
		for line in f:
			line = line.strip().split(',')
			category = line[1]
			png = line[0].split('/')[-1]
			category_path = os.path.join(path, category)
			png_path = os.path.join(path, png)
			# mkdir
			if os.path.exists(category_path) is False:
				os.mkdir(category_path)
			# mv png file to subdirectory
			os.rename(png_path, os.path.join(category_path, png))

# Import picture paths in folder to list in a random order
path = argv[1]

filelist = glob.glob(os.path.join(path,'*png'))
random.shuffle(filelist)
findFace(filelist, len(filelist))
# findFace(filelist, 120)
# Now organize cropped pngs
organize(path)