import torch
import pickle
import argparse
import progressbar
import numpy as np
import random as rd
from Model import Model
import torch.optim as optim
import torch.nn.functional as F
from Data_process import Data_process
from DataLoader import DataLoader

class Main:
	
	def __init__(self, args):
		self.lr = args.lr
		self.ns = args.ns
		self.hidden = args.hidden
		self.dataset = args.dataset
		self.init_type = args.init_type
		self.numOfEpoch = args.numOfEpoch
		self.his_length = args.his_length
		self.graph_sample = args.graph_sample
		self.minibatch = 1

		Data = Data_process(self.dataset)
		self.data_dict = Data.data_dict
		self.DataLoader = DataLoader(self.data_dict, args)

		self.train_time = self.data_dict["train_time"]
		self.all_time = self.data_dict["all_time"]
		self.batch_size = self.data_dict["data_size"] // self.train_time

		#self.device = torch.device("cuda:-1")

		raw_ent = []
		raw_rel = []
		self.model = Model(self.data_dict["entity_num"], self.data_dict["relation_num"], self.hidden, self.dataset, raw_ent, raw_rel, args.init_type)
		self.model.cuda()
		self.best = [0,0,0,0,0]

		self.Train()

	def Train(self):
		optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = 0)
		#quick_params = list(map(id, self.model.relation_embeddings_1.parameters())) # 返回的是parameters的 内存地址
		#base_params = filter(lambda p: id(p) not in quick_params, self.model.parameters()) 
		#optimizer = optim.Adam([
		#{'params': base_params},
		#{'params': self.model.relation_embeddings_1.parameters(), 'lr': 0.01}], self.lr)
		for epoch in range(self.numOfEpoch):
			epochLoss = 0
			p = progressbar.ProgressBar(widgets = ["Epoch", str(epoch),":[", progressbar.Percentage(),"]", progressbar.Timer()], maxval = self.train_time-1)
			p.start()
			self.DataLoader.reset()
			self.model.reset()
			self.model.train()
			t=0
			#print(self.model.entity_weights.weight.data[0][0:10])
			while t < self.train_time-1:
				new_nodes = []
				appeared_nodes = []
				related_nodes = []
				active_nodes = []
				interact_message = []
				prop_nodes = []
				prop_message = []
				pos_sample = []
				neg_sample = []
				steps = []
				p.update(t)
				for i in range(self.minibatch):
					if t+i < self.train_time-1:
						batch_data, triples = self.DataLoader.generateBatches(t+i, "Train")

						new_nodes.append(batch_data["new_nodes"])
						active_nodes.append(batch_data["active_nodes"])
						appeared_nodes.append(batch_data["appeared_nodes"])
						interact_message.append(batch_data["interact_message"])
						prop_nodes.append(batch_data["prop_nodes"])
						prop_message.append(batch_data["prop_message"])
						related_nodes.append(batch_data["related_nodes"])
						
						pos_sample.append(triples)
						neg_sample.append(self.DataLoader.negativeSampling(triples, batch_data["active_nodes"], batch_data["prop_nodes"], t))
						steps.append(t+i)
					else:
						break

				batchLoss = self.model.forward(t, new_nodes, active_nodes, interact_message, \
												prop_nodes, prop_message, \
												pos_sample, neg_sample, steps, appeared_nodes, related_nodes)
				t += 1
				if batchLoss != 0:
					optimizer.zero_grad()
					batchLoss.backward()
					optimizer.step()

					epochLoss += batchLoss
				
			p.finish()
			print("loss: " + str(float(epochLoss)))

			if epoch % 3 == 0:
				self.model.eval()
				with torch.no_grad():
					print("evaluating...")
					self.eval(self.model.entity_weights.weight.data)
		print(self.best)
				
	def adjust_learning_rate(self, optimizer, batch_len):
		lr = self.lr * (batch_len/self.batch_size)
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	def save_embedding(self):
		base_path = "../dataset/" + self.dataset + "/"
		ent_path = base_path+"ent_raw_B_50.pt"
		rel_path = base_path+"rel_raw_B_50.pt"
		torch.save(self.model.entity_weights.weight.data, ent_path)
		torch.save(self.model.relation_embeddings.weight.data, rel_path)
		print("saved")

	def get_embdding(self):
		base_path = "../dataset/" + self.dataset + "/"
		ent_path = base_path+"ent_raw_B_50.pt"
		rel_path = base_path+"rel_raw_B_50.pt"
		ent_emb = torch.load(ent_path)
		rel_emb = torch.load(rel_path)

		return ent_emb, rel_emb


	def eval(self, pre_weights):
		MRR = [0,0]
		H1 = [0,0]
		H3 = [0,0]
		H5 = [0,0]
		H10 = [0,0]
		start_time = self.data_dict["train_time"]
		p = progressbar.ProgressBar(widgets = ["Forecast:", progressbar.Bar('*'), progressbar.Percentage(), "|", progressbar.Timer()], maxval = self.all_time-start_time-1)
		p.start()
		numOftriples = 0
		tmp_weights = pre_weights.clone().detach()
		for batch in range(self.all_time-start_time-1):
			#tmp_weights = pre_weights.clone().detach()
			batch_time = batch + start_time
			p.update(batch)
			batch_data, triples = self.DataLoader.generateBatches(batch_time, "Forecast")
			new_nodes = batch_data["new_nodes"].cuda()
			appeared_nodes = batch_data["appeared_nodes"].cuda()
			active_nodes = batch_data["active_nodes"].cuda()
			interact_message = batch_data["interact_message"].cuda()
			prop_nodes = batch_data["prop_nodes"].cuda()
			prop_message = batch_data["prop_message"].cuda()
			related_nodes = batch_data["related_nodes"].cuda()
			new_weights, norm1 , norm2= self.model.update_weights(tmp_weights, new_nodes, active_nodes, interact_message, \
													prop_nodes, prop_message, batch_time, active_nodes, related_nodes)
			
			#print(new_weights[0][0:10])
			if batch_time in self.data_dict["test_triples"].keys(): 
				numOftriples += len(triples[1])
				result = self.model.reconstruct(new_weights, appeared_nodes.cuda(), triples[0].cuda(), batch_time, appeared_nodes)
				MRR[0] += result[0]
				H1[0] += result[1]
				H3[0] += result[2]
				H5[0] += result[3]
				H10[0] += result[4]

				result = self.model.forecast(new_weights, appeared_nodes.cuda(), triples[1].cuda(), batch_time+1, active_nodes, new_nodes, self.data_dict["T-e-r-e"][batch_time+1])
				MRR[1] += result[0]
				H1[1] += result[1]
				H3[1] += result[2]
				H5[1] += result[3]
				H10[1] += result[4]

		p.finish()
		#print(numOftriples)
		MRR[0] = MRR[0]/(2*numOftriples)
		H1[0] = H1[0]/(2*numOftriples)
		H3[0] = H3[0]/(2*numOftriples)
		H5[0] = H5[0]/(2*numOftriples)
		H10[0] = H10[0]/(2*numOftriples)

		MRR[1] = MRR[1]/(2*numOftriples)
		H1[1] = H1[1]/(2*numOftriples)
		H3[1] = H3[1]/(2*numOftriples)
		H5[1] = H5[1]/(2*numOftriples)
		H10[1] = H10[1]/(2*numOftriples)

		'''
		print("MRR_F: " + str(MRR[1]) + " | MRR_L: " + str(MRR[0]))
		print("H1_F: " + str(H1[1]) + " | H1_L: " + str(H1[0]))
		print("H3_F: " + str(H3[1]) + " | H3_L: " + str(H3[0]))
		print("H5_F: " + str(H5[1]) + " | H5_L: " + str(H5[0]))
		print("H10_F: " + str(H10[1]) + " | H10_L: " + str(H10[0]))
		'''

		if MRR[1] > self.best[0]:
			self.best[0] = MRR[1]
			self.best[1] = H1[1]
			self.best[2] = H3[1]
			self.best[3] = H5[1]
			self.best[4] = H10[1]

		return 0

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="model")
	parser.add_argument("--hidden",dest="hidden",type=int,default=200)
	parser.add_argument("--lr", dest="lr", type=float, default=1e-3)
	parser.add_argument("--ns", dest="ns", type=int, default=20)
	parser.add_argument("-init_type", dest="init_type", type=str, default="average")
	parser.add_argument("--dataset",dest="dataset",type=str,default="ICEWS18")
	parser.add_argument("--numOfEpoch",dest="numOfEpoch",type=int,default=100)
	parser.add_argument("--his_length", dest="his_length",type=int,default=2) 
	parser.add_argument("--graph_sample", dest="graph_sample",type=int,default=10)

	args=parser.parse_args()
	Main(args)








		