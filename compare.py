import subprocess
import os

patients = [28, 30, 51, 100, 151, 177, 195, 198, 239, 244, 248, 254, 266, 272, 287, 332, 358, 395, 397, 398, 432, 434, 446, 457, 461]
done = 9
es = [[16, 17, 18], [10, 11, 12], [11, 12, 13], [13, 14, 15], [11, 12, 13], [13, 14, 15], [13, 14, 15], [10, 11, 12], [10, 11, 12]]

findPath = '../!Data/train/'
saxList = [[] for _ in range(done)]


for i in range(done):
	patient = patients[i]
	path = findPath + str(patient) + '/study/'
	cmd = "ls -d " + path + 'sax_*'
	saxList[i] = map(lambda x: int(x[x.find('_')+1:]), subprocess.check_output(cmd, shell=True).split())
	saxList[i].sort()
	saxList

print saxList