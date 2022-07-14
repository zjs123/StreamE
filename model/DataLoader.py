import torch
import numpy as np
import random

class DataLoader:

	def __init__(self, data_dict, args):
		self.numOfEntity = data_dict["entity_num"]
		self.numOfRelation = data_dict["relation_num"]
		self.train_time = data_dict["train_time"]
		self.all_time = data_dict["all_time"]

		self.data_dict = data_dict
		self.his_length = args.his_length
		self.graph_sample = args.graph_sample
		self.negative_sample = args.ns

		self.all_ent = set([i for i in range(self.numOfEntity)])
		self.all_rel = [i for i in range(self.numOfRelation)]

		self.appeared_nodes = set()
		self.seen_nodes = set()
		random.seed(2021)

	def reset(self):
		self.appeared_nodes.clear()

	def filter(self, batch):
		new_batch = []
		for triple in batch:
			if triple[0] not in self.appeared_nodes or triple[2] not in self.appeared_nodes:
				continue
			else:
				new_batch.append(triple)
		return new_batch

	def filter_unseen(self, batch):
		new_batch = []
		for triple in batch:
			if triple[0] not in self.appeared_nodes or triple[2] not in self.appeared_nodes:
				continue
			
			if triple[0] in self.seen_nodes and triple[2] in self.seen_nodes:
				continue
			new_batch.append(triple)
		return new_batch

	def generateBatches(self, batch, Type):

		batch_data = {}

		#find new nodes
		if len(self.appeared_nodes) == 0:
			new_nodes = set(self.data_dict["T-e-r-e"][batch].keys())
			self.appeared_nodes = set(self.data_dict["T-e-r-e"][batch].keys())
		else:
			new_nodes = set(self.data_dict["T-e-r-e"][batch].keys()) - self.appeared_nodes
			self.appeared_nodes = set(self.data_dict["T-e-r-e"][batch].keys()) | self.appeared_nodes
		#find related entity for new_nodes
		related_nodes = []
		for node in new_nodes:
			tmp_related = []
			tmp_list = self.data_dict["e-[r,e]"][node][batch]
			#print(self.data_dict["[r,e]-e"].keys())
			for r_e in tmp_list:
				r_e_index = " ".join('%s' %id for id in r_e)
				tmp_related.extend(list(self.data_dict["[r,e]-e"][r_e_index]))
			tmp_related = list(set(tmp_related)&self.appeared_nodes)
			if len(tmp_related) < self.graph_sample:
				related_nodes.append(tmp_related+[self.numOfEntity]*(self.graph_sample - len(tmp_related)))
			else:
				related_nodes.append(tmp_related[0:self.graph_sample])

		batch_data["related_nodes"] = torch.LongTensor(list(related_nodes))

		batch_data["new_nodes"] = torch.LongTensor(list(new_nodes))
		#find active nodes
		batch_data["active_nodes"] = torch.LongTensor(list(self.data_dict["T-e-r-e"][batch].keys()))

		#generate interact message
		batch_data["interact_message"] = []
		prop_nodes_2_message = {}
		for node in self.data_dict["T-e-r-e"][batch].keys():
			rel = []
			inter_e = []
			triples = self.data_dict["e-[r,e]"][node][batch]
			if len(triples) < self.graph_sample:
				tmp_list = list(triples) + [[self.numOfRelation, self.numOfEntity]]*(self.graph_sample-len(triples))
			else:
				
				if batch <= self.train_time:
					tmp_list = random.sample(list(triples), self.graph_sample)
				else:
					tmp_list = triples[0:self.graph_sample]
				
				#tmp_list = triples[0:self.graph_sample]
			for inter in tmp_list:
				rel.append(inter[0])
				inter_e.append(inter[1])

				pre_inter = sum(self.data_dict["e-[r,e]"][node][max(batch-self.his_length,0):batch],[])
				if len(pre_inter) > self.graph_sample:
					
					if batch <= self.train_time:
						pre_inter = random.sample(pre_inter, self.graph_sample)
					else:
						pre_inter = pre_inter[0:self.graph_sample]
					
					#pre_inter = pre_inter[0:self.graph_sample]
				for inter in pre_inter:
					rel_p = inter[0]
					ent_p = inter[1]
					if ent_p == inter_e[-1]:
						continue
					if ent_p in prop_nodes_2_message.keys():
						prop_nodes_2_message[ent_p].append([rel_p,rel[-1],inter_e[-1]])
					else:
						prop_nodes_2_message[ent_p] = []
						prop_nodes_2_message[ent_p].append([rel_p,rel[-1],inter_e[-1]])

			batch_data["interact_message"].append([rel, inter_e])
		batch_data["interact_message"] = torch.LongTensor(batch_data["interact_message"])#act_node*3*sample

		#find prop nodes
		batch_data["prop_nodes"] = torch.LongTensor(list(prop_nodes_2_message.keys()))

		#generate prop message
		batch_data["prop_message"] = []
		if len(prop_nodes_2_message.keys()) == 0:
			batch_data["prop_message"].append([[self.numOfRelation, self.numOfRelation, self.numOfEntity]]*self.graph_sample)
		else:
			for node in  list(prop_nodes_2_message.keys()):
				if  len(prop_nodes_2_message[node]) < self.graph_sample:
					message_b = prop_nodes_2_message[node]
					message_b.extend([[self.numOfRelation, self.numOfRelation, self.numOfEntity]]*(self.graph_sample-len(prop_nodes_2_message[node])))
					batch_data["prop_message"].append(message_b)
				else:
					
					if batch <= self.train_time:
						batch_data["prop_message"].append(random.sample(prop_nodes_2_message[node], self.graph_sample))
					else:
						batch_data["prop_message"].append(prop_nodes_2_message[node][0:self.graph_sample])
					
					#batch_data["prop_message"].append(prop_nodes_2_message[node][0:self.graph_sample])

		batch_data["prop_message"] = torch.LongTensor(batch_data["prop_message"])#prop*sample*3

		batch_data["appeared_nodes"] = torch.LongTensor(list(self.appeared_nodes))
		#print(len(self.appeared_nodes))

		triples = []
		if Type == "Train":
			triples.append(torch.LongTensor(self.data_dict["all_triples"][batch]))
			triples.append(torch.LongTensor(self.filter(self.data_dict["all_triples"][batch+1])))
			self.seen_nodes = set(self.appeared_nodes)
		else:
			triples.append(torch.LongTensor(self.data_dict["all_triples"][batch]))
			triples.append(torch.LongTensor(self.filter_unseen(self.data_dict["all_triples"][batch+1])))
			'''
			if batch+1 in self.data_dict["all_triples"].keys():
				if batch not in self.data_dict["all_triples"].keys():
					triples.append(torch.LongTensor(self.data_dict["all_triples"][batch]))
				else:
					triples.append(torch.LongTensor(self.data_dict["all_triples"][batch]))
				triples.append(torch.LongTensor(self.filter(self.data_dict["all_triples"][batch+1])))
			if batch+1 in self.data_dict["all_triples"].keys():
				if batch not in self.data_dict["all_triples"].keys():
					triples.append(torch.LongTensor(self.data_dict["all_triples"][batch]))
				else:
					triples.append(torch.LongTensor(self.data_dict["all_triples"][batch]))
				triples.append(torch.LongTensor(self.filter_unseen(self.data_dict["all_triples"][batch+1])))
		'''
		return batch_data, triples


	def negativeSampling(self, triples, active_nodes, prop_nodes, batch):
		neg_h_1 = []
		neg_t_1 = []
		neg_r_1 = []
		neg_h_2 = []
		neg_t_2 = []
		neg_r_2 = []
		neg_samples = []
		now_nodes = set(active_nodes) | set(prop_nodes)
		#print(len(active_nodes))
		next_nodes = self.appeared_nodes#  | set(list(self.data_dict["T-e-r-e"][batch+1].keys())) #set(list(self.data_dict["T-e-r-e"][batch+1].keys())) | self.appeared_nodes# - now_nodes # | set(list(self.data_dict["T-e-r-e"][max(batch-1,0)].keys()))
		#print(len(self.appeared_nodes))
		for triple in triples[0]:
			head = int(triple[0])
			rel = int(triple[1])
			tail = int(triple[2])

			candidate_nodes_h = now_nodes - set([head])  - set(self.data_dict["e-e-[r]"][tail].keys()) #list(torch.cat([active_nodes, prop_nodes]))
			candidate_nodes_t = now_nodes - set([tail])  - set(self.data_dict["e-e-[r]"][head].keys()) #  - set(sum(self.data_dict["e-r-[e]"][tail][-rel], []))
			#candidate_nodes.extend(random.sample(list(self.all_ent - set([head]) - set([tail])), self.negative_sample-len(candidate_nodes)))
			sample_h = random.sample(list(candidate_nodes_h), self.negative_sample)
			sample_t = random.sample(list(candidate_nodes_t), self.negative_sample)

			neg_h_1.extend(sample_h)
			neg_r_1.extend(random.sample(self.all_rel, self.negative_sample))
			neg_t_1.extend(sample_t)

		neg_samples.append(torch.LongTensor([neg_h_1, neg_r_1, neg_t_1]))

		for triple in triples[1]:
			head = int(triple[0])
			rel = int(triple[1])
			tail = int(triple[2])

			candidate_nodes_h = next_nodes - set([head])  - set(self.data_dict["e-e-[r]"][tail].keys()) #list(torch.cat([active_nodes, prop_nodes]))
			candidate_nodes_t = next_nodes - set([tail])  - set(self.data_dict["e-e-[r]"][head].keys())
			#candidate_nodes.extend(random.sample(list(self.all_ent - set([head]) - set([tail])), self.negative_sample-len(candidate_nodes)))
			sample_h = random.sample(list(candidate_nodes_h), self.negative_sample)
			sample_t = random.sample(list(candidate_nodes_t), self.negative_sample)
			#print(candidate_nodes)

			neg_h_2.extend(sample_h)
			neg_r_2.extend(random.sample(self.all_rel, self.negative_sample))
			neg_t_2.extend(sample_t)

		neg_samples.append(torch.LongTensor([neg_h_2, neg_r_2, neg_t_2]))

		return neg_samples

