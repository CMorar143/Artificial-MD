import xport
with open('example.xpt', 'rb') as f:
    for row in xport.reader(f):
        print row