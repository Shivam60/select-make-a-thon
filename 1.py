from threading import Thread
import queue

def do_stuff(q):
  while True:
    print (q.get())
    q.task_done()

q = queue.Queue(maxsize=0)
num_threads = 1

for i in range(num_threads):
  worker = Thread(target=do_stuff, args=(q,))
  worker.setDaemon(True)
  worker.start()

for y in range (10):
  for x in range(100):
    q.put(x + y * 100)
  q.join()
  print ("Batch " + str(y) + " Done")
