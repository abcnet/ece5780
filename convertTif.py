import subprocess
import os
tifPath = '/Users/zr/Library/Mobile Documents/com~apple~CloudDocs/!Cornell/!Courses/ECE 5780 Computer Analysis of Biomed Images/Project/Kaggle/9'

patients = [28, 30, 51, 100, 151, 177, 195, 198, 239, 244, 248, 254, 266, 272, 287, 332, 358, 395, 397, 398, 432, 434, 446, 457, 461]
done = 9
es = [[16, 17, 18], [10, 11, 12], [11, 12, 13], [13, 14, 15], [11, 12, 13], [13, 14, 15], [13, 14, 15], [10, 11, 12], [10, 11, 12]]
seeds_ed_inner = [[],[],[],[],[],[],[],[]]
seeds_ed_outer = [[],[],[],[],[],[],[],[]]
seeds_es_inner = [[],[],[],[],[],[],[],[]]
seeds_es_outer = [[],[],[],[],[],[],[],[]]



for i in range(done):
	patient = patients[i]
	print 'running patient', patient 
	patientPath = tifPath + '/' + ('CM0' if patient >= 100 else 'CM00') + str(patient)
	cmd = 'ls "' + patientPath + '"'
	files = subprocess.check_output(cmd, shell=True).split()
	# print files
	for file in files:
		if file[-4:] == '.tif':
			print 'converting file ', file
			single = file[:-4] + '-single.png'
			cmd = 'mkdir tiftest/' + str(patient)
			# print cmd
			os.system(cmd)
			os.system('convert "' + patientPath + '/' + file + '" tiftest/' + str(patient) + '/' + single)