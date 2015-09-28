import socket
import threading
import SocketServer
import json
import time
import sys
import BINoculars.util, BINoculars.main
import traceback
import re

queue = list()

class MyTCPHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        input = self.request.recv(1024)
        try:
            job = json.loads(input)
            queue.append(job)
            print 'Recieved command: {0}. Job is added to queue.\nJobs left in queue: {1}'.format(job['command'], ','.join(tuple(j['command'] for j in queue)))
            response = 'Job added to queue'
        except:
            print 'Could not parse the job: {0}'.format(input)
            response = 'Error: Job could not be added to queue'
        finally:
            self.request.sendall(response)

def process(run_event):
    while run_event.is_set():
        if len(queue) == 0:
            time.sleep(1)
        else:
            job = queue.pop()
            command = job['command']
            print 'Start processing: {0}'.format(command)
            try:
                if 'overrides' in job:
                    overrides = tuple(re.split('[:=]', ovr) for ovr in job['overrides'])
                else:
                    overrides = []
                configobj = BINoculars.util.ConfigFile.fromtxtfile(job['configfilename'], overrides = overrides)
                BINoculars.main.Main(configobj, [command])
                print 'Succesfully finished processing: {0}.'.format(command)
            except Exception as exc:
                errorfilename = 'error_{0}.txt'.format(command)
                print 'An error occured for scan {0}. For more information see {1}'.format(command, errorfilename)
                with open(errorfilename, 'w') as fp:
                    traceback.print_exc(file = fp)
            finally:
                print 'Jobs left in queue: {0}'.format(','.join(tuple(j['command'] for j in queue)))

if __name__ == "__main__":
    HOST, PORT = socket.gethostbyname(socket.gethostname()), 0

    run_event = threading.Event()
    run_event.set()

    process_thread = threading.Thread(target=process, args = (run_event,))
    process_thread.start()

    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)
    ip, port = server.server_address

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    print 'Server started running at ip {0} and port {1}. Interrupt server with Ctrl-C'.format(ip,port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        run_event.clear()
        process_thread.join()
