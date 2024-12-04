import numpy as np
from transport import transport_torch as transport_pure_gpu
def black_box(DA, SB, C, delta, device, ymax):
	rows = len(C) 
	cols = len(C[0])
	C_comp = [[0 for i in range(cols)] for j in range(rows)]
	for i in range(rows):
		for j in range(cols): 
			if C[i][j]>=ymax or C[i][j]==0:
        C_comp[i][j] = ymax
			else:
				C_comp[i][j]=C[i][j]
	Mb, yA, yB, ot_pyt_loss, iteration = transport_pure_gpu(DA, SB, C_comp, delta, device=device)
	for i in range(rows):
		for j in range(cols): 
			if C[i][j]>=ymax :
				ot_pyt_loss -= Mb[i][j]*ymax
				Mb[i][j]=0
	return Mb, yA, yB, ot_pyt_loss, iteration
def RPW_approx(DA, SB, C, delta, device):
	
