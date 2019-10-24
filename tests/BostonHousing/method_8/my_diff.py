fp_consant=open("model_constant.h5","rb")
fp_adaptive=open("model_adaptive.h5","rb")

lines_constant=fp_consant.readlines()
lines_adaptive=fp_adaptive.readlines()

for i in range(len(lines_adaptive)):
	if(lines_adaptive[i]!=lines_constant[i]):
		print("Line number ",i+1)
		print("lines_adaptive",hex(int(lines_adaptive[i],2)))
		print("lines_constant",hex(int(lines_constant[i],2)))
		break

fp_consant.close()
fp_adaptive.close()