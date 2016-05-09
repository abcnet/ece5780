# import subprocess
import commands
#vxtotiffs if=sax_9.1_boundary.vxb of=sax_9.1_boundary.tif

cm = ['CM0028', 'CM0030', 'CM0051', 'CM0100', 'CM0151', 
'CM0171', 'CM0195', 'CM0198', 'CM0239', 'CM0244', 
'CM0248', 'CM0254', 'CM0266', 'CM0272', 'CM0287', 
'CM0332', 'CM0357', 'CM0358', 'CM0395', 'CM0397', 
'CM0398', 'CM0432', 'CM0434', 'CM0446', 'CM0461']
done = 1
# f = open('run.sh', 'w')
for i in range(done):
	commands.getstatusoutput("imglnk sc=" + cm[i] + '\n')
	files = commands.getstatusoutput('ls ' + cm[i])[1].split()
	# print files
	for file in files:
		if file[-4:] == '.vxa':
			# print file
			commands.getstatusoutput("varend -f " + cm[i] + '/' + file + " | vdim -c of=" + cm[i] + '/' + file[:-4] + '.vxb')
	# subprocess.check_output(cmd, shell=True)
# f.close()