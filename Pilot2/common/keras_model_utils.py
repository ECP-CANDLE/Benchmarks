import keras
import numpy as np

def get_conv_macs_size(l):
	Data={}
	count=0
	if 'CONVOLUTION2D' in str(l.__class__).upper():
		Data['layer_name']='convolution'
		Data['channels']=l.input_shape[1]
		Data['filters']=l.output_shape[1]
		Data['kernel_x']=l.nb_row
		Data['kernel_y']=l.nb_col
		Data['img_y']=l.output_shape[3]
		Data['img_x']=l.output_shape[2]

		Data['layer_macs']=Data['channels']*Data['filters']*Data['kernel_x']*Data['kernel_y']*Data['img_x']*Data['img_y']
		Data['layer_size']=Data['channels']*Data['filters']*Data['kernel_x']*Data['kernel_y']
		return Data
	else:
		return None
	

def get_dense_mac_size(l):
	Data={}
	if 'DENSE' in str(l.__class__).upper():
		Data['layer_name']='dense'
		Data['num_hidden']=l.output_shape[1]
		Data['layer_size']=np.prod(l.input_shape[1:])*l.output_shape[1]
		Data['layer_macs']=np.prod(l.input_shape[1:])*l.output_shape[1]
		return Data
	else:
		return None

def Model_Info(net):
	print_str= "|%15s | %10s | %5s  | %8s |%31s |%23s |" % ('type', 'feature size', '#Filters','FilterSize', '#params', '#MACs')
	print (len(print_str)*'-')
	print (print_str)
	print (len(print_str)*'-')

	## Get Total Macs and Total Params
	Macs=0; Size=0
	layers=net.layers
	#layers=lasagne.layers.get_all_layers(net[net.keys()[-1]])

	for l in layers:
		Data_C=get_conv_macs_size(l)
		if Data_C!=None:
			Macs+=Data_C['layer_macs']
			Size+=Data_C['layer_size']
		Data_L=get_dense_mac_size(l)
		if Data_L!=None:
			Macs+=Data_L['layer_macs']
			Size+=Data_L['layer_size']

	
	cp =Size
	cm=Macs
	for l in layers:
		#print l.output_shape,l.__module__
		Data=get_conv_macs_size(l)
		if Data!=None:
			lt=Data['layer_name']
			num_modules_x=Data['img_x']
			num_modules_y=Data['img_y']
			num_filters=Data['filters']
			num_params=Data['layer_size']
			num_macs=Data['layer_macs']
			filter_size_x=Data['kernel_x']
			filter_size_y=Data['kernel_y']

			stats_str= "|%15s |%3dx%-6d    |%8d   | %02dx%02d     |%6.2f Mill,%6.2f MB (%6.1f%%) |%8.2f Mill (%6.1f%%) |" \
			% (lt, num_modules_x, num_modules_y, num_filters , filter_size_x, filter_size_y,num_params/1.0e6, num_params*4.0/1024/1024,\
			 float(num_params)/float(cp)*100, num_macs/1e6, float(num_macs)/float(cm)*100 )
			print (stats_str)
		
		Data=get_dense_mac_size(l)
		if Data!=None:
			lt=Data['layer_name']		
			num_params=Data['layer_size']
			num_macs=Data['layer_macs']
			num_hidden=Data['num_hidden']
			stats_str= "|%15s |%3dx%-6d    |%8d   | %02dx%02d     |%6.2f Mill,%6.2f MB (%6.1f%%) |%8.2f Mill (%6.1f%%) |" \
			% (lt, 0, 0, num_hidden , 1, 1,num_params/1.0e6, num_params*4.0/1024/1024,\
			 float(num_params)/float(cp)*100, num_macs/1e6, float(num_macs)/float(cm)*100 )
			print (stats_str)				
			#print (k,':',(10-len(k))*' ',model.nodes[k].nb_filter,(5-len(str(model.nodes[k].nb_filter)))*' ',model.nodes[k].nb_row,5*' ',model.nodes[k].input_shape[2],(8-len(str(model.nodes[k].input_shape[2])))*' ',\
			# model.nodes[k].input_shape[3],(8-len(str(model.nodes[k].input_shape[3])))*' ',model.nodes[k].output_shape[2],(8-len(str(model.nodes[k].output_shape[2])))*' ',model.nodes[k].output_shape[3],\
			# (7-len(str(model.nodes[k].output_shape[3])))*' ',layer_macs,(10-len(str(layer_macs)))*' ',layer_size)
	final_stats_str= "|%*s| %6.2f Mill,%6.2f MB (%6.1f%%)| %7.2f Mill (%6.1f%%) | %9s |" % (40+8, '+'*(40+8) ,Size/1.0e6 , Size*4.0/1024/1024 , float(Size)/float(cp)*100, Macs/1e6 , float(Macs)/float(cm)*100, '')

	print (len(print_str)*'-')
	print (final_stats_str)
	print (len(print_str)*'-')

	#print ('Total MACS:',Macs/1e9, 'BILLION')
	#print ('Total Size:', 4*Size/1024./1024., 'MB') 

	#return Macs,Size
	
