import numpy as np
import torch
from transport import transport_torch as transport_pure_gpu
def black_box(DA, SB, C, delta, device, ymax):
	rows = len(C) 
	cols = len(C[0])

	C_temp=C.clone()
	C_temp[C_temp>=ymax]=ymax
	print("test", C_temp)
	print("success")
	Mb, yA, yB, ot_pyt_loss, iteration = transport_pure_gpu(DA, SB, C_temp, delta, device=device)
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
	'''C_scale = [[0 for i in range(cols)] for j in range(rows)]
	for i in range(rows):
		for j in range(cols): 
			C_scale=C[i][j]**p
			C_scale=C[i][j]*Fi_est'''
	C_scale=C.clone()
	C_scale=C_scale.pow(p)
	C_scale=C_scale*Fi_est
	ymax=min(1,(Fi_est**p)/delta
	Mb, yA, yB, ot_pyt_loss, iteration=black_box(DA, SB, C_scale, delta, device=device, ymax)
	p_root=1/p
	ot_pyt_loss=ot_pyt_loss**p_root
	return Mb, yA, yB, ot_pyt_loss, iteration
