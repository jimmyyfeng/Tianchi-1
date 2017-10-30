import psutil
import subprocess
import time

def sysinfo():
	subprocess.Popen('top')
	while True:
		print psutil.cpu_percent(percpu=True)
		print psutil.virtual_memory()
		time.sleep(10)		
if __name__=='__main__':
	sysinfo()
