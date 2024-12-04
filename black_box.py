import numpy as np
import torch
from transport import transport_torch as transport_pure_gpu
def black_box(DA, SB, C, delta, device, ymax):
	"""rows = len(C) 
	cols = len(C[0])
	C_comp = [[0 for i in range(cols)] for j in range(rows)]
	for i in range(rows):
		for j in range(cols): 
			if C[i][j]>=ymax or C[i][j]==0:
				C_comp[i][j] = ymax
				print("bad")
			else:
				C_comp[i][j]=C[i][j]
				print("good",i," ",j)"""

	
	#mask = C < ymax
	C[C>=ymax]=ymax
	#mask2 = C_temp>= ymax
	#size=C.size()
	
	C_comp = C[mask]
	print("test", C)
	#C_temp=C_temp[mask2]
	C_comp=C_comp.add(C_temp)
	
	#C_tensor = torch.tensor(C_comp, device=device, requires_grad=False)
	print("success")
	Mb, yA, yB, ot_pyt_loss, iteration = transport_pure_gpu(DA, SB, C, delta, device=device)
	for i in range(rows):
		for j in range(cols): 
			if C[i][j]>=ymax :
				ot_pyt_loss -= Mb[i][j]*ymax
				Mb[i][j]=0
	return Mb, yA, yB, ot_pyt_loss, iteration
def RPW_approx(DA, SB, C, delta, p, device):
	Fi_est=1
	rows = len(C) 
	cols = len(C[0])
	C_scale = [[0 for i in range(cols)] for j in range(rows)]
	for i in range(rows):
		for j in range(cols): 
			C_scale=C[i][j]**p
			C_scale=C[i][j]*Fi_est
	Mb, yA, yB, ot_pyt_loss, iteration=black_box(DA, SB, C_scale, delta, device, Fi_est)
	p_root=1/p
	ot_pyt_loss=ot_pyt_loss**p_root
	return Mb, yA, yB, ot_pyt_loss, iteration
