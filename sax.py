import subprocess
patients = [28, 30, 51, 100, 151, 177, 195, 198, 239, 244, 248, 254, 266, 272, 287, 332, 358, 395, 397, 398, 432, 434, 446, 457, 461]
done = 7
findPath = '../!Data/train/'

for i in range(done):
	patient = patients[i]
	path = findPath + str(patient) + '/study/'
	cmd = "ls -d " + path + 'sax_*'
	saxList = subprocess.check_output(cmd, shell=True).split()
	print saxList