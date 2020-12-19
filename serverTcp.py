import socket
import sys

# Stores a history of user annotated classifiers for ease
history = set()

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Bind the socket to the port
server_address = ('localhost', 10000)
print(sys.stderr, 'starting up on %s port %s' % server_address)
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

while True:
    # Wait for a connection
    print(sys.stderr, 'waiting for a connection')
    connection, client_address = sock.accept()

    try:
        print(sys.stderr, 'connection from', client_address)

        # Receive the data in small chunks and retransmit it
        while True:
            # 2 receive 1st message
            data = connection.recv(16)

            if data:
                # Check if data matches up
                # 3 send back 1st message for confirmation
                connection.sendall(data)
                while True:
                    print('confirming')
                    # 5 receive either a confirmation or another message
                    confirm = connection.recv(16).decode()
                    if confirm == "True":
                        print('saved')
                        break
                    else:
                        print(confirm)
                        data = confirm
                        connection.sendall(data.encode())

                print(sys.stderr, 'received "%s"' % data.decode())
                counter = int(data.decode())
                userinput = input("Classification history: %s\nPlease classify: " % ', '.join(list(history)))
                userinput = userinput.strip().lower()
                history.add(userinput)
                userinput = userinput.encode()

                # print(sys.stderr, 'sending data back to the client')
                connection.sendall(userinput)
            else:
                print(sys.stderr, 'no more data from', client_address)
                break

    finally:
        # Clean up the connection
        print('Closing connection')
        connection.close()
        # Kill the script if last counter    
        if counter == 1:
            quit()
