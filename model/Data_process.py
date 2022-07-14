import os
import torch
import pickle
import numpy as np
import random as rd
from datetime import date

class Data_process:

	def __init__(self, dataset):
		self.base_path = "../dataset/" + dataset + "/"
		self.dataset = dataset

		self.prop_length = 3
		self.batch_size = 0

		self.data_dict = {}
		self.read_data()

	def read_stat(self):
		path = self.base_path + "stat.txt"
		stat_input = open(path)
		line = stat_input.readline()
		self.data_dict["data_size"] = int(line.strip().split()[0])
		self.data_dict["all_time"] = int(line.strip().split()[1])
		self.data_dict["train_time"] = int(line.strip().split()[2])
		self.data_dict["entity_num"] = int(line.strip().split()[3])
		self.data_dict["relation_num"] = int(line.strip().split()[4])

	def read_triples(self):
		self.data_dict["train_triples"] = {}
		self.data_dict["valid_triples"] = {}
		self.data_dict["test_triples"] = {}
		self.data_dict["all_triples"] = {}
		for data_type in ["train", "valid", "test"]:
			path = self.base_path + data_type + ".txt"
			data_input = open(path)
			lines = data_input.readlines()
			for line in lines:
				tmpHead = int(line.strip().split()[0])
				tmpRel = int(line.strip().split()[1])
				tmpTail = int(line.strip().split()[2])
				tmpTime = int(line.strip().split()[3])
				if "ICEWS" in self.dataset:
					tmpTime = tmpTime//24
				if tmpTime not in self.data_dict["all_triples"].keys():
					self.data_dict["all_triples"][tmpTime] = []
					self.data_dict["all_triples"][tmpTime].append([tmpHead, tmpRel, tmpTail])
				else:
					self.data_dict["all_triples"][tmpTime].append([tmpHead, tmpRel, tmpTail])
				if tmpTime not in self.data_dict[data_type+"_triples"].keys():
					self.data_dict[data_type+"_triples"][tmpTime] = []
					self.data_dict[data_type+"_triples"][tmpTime].append([tmpHead, tmpRel, tmpTail])
				else:
					self.data_dict[data_type+"_triples"][tmpTime].append([tmpHead, tmpRel, tmpTail])

	def read_data(self):
		tmp_dict = {}

		if os.path.exists(self.base_path + "data_dict.pickle"):
			data_input = open(self.base_path + "data_dict.pickle", "rb")
			self.data_dict.update(pickle.load(data_input))
			data_input.close()
			return 0

		#read stat
		self.read_stat()

		#read triples
		self.read_triples()

		path = self.base_path + "all_data.txt"
		data_input = open(path)
		lines = data_input.readlines()

		#pre-read data
		tmp_dict["e-e-T-r"] = {}
		tmp_dict["e-r-T-e"] = {}
		self.data_dict["T-e-r-e"] = {}
		self.data_dict["r-[e]"] = {}
		for line in lines:
			tmpHead = int(line.strip().split()[0])
			tmpRel = int(line.strip().split()[1])
			tmpTail = int(line.strip().split()[2])
			tmpTime = int(line.strip().split()[3])
			if "ICEWS" in self.dataset:
				tmpTime = tmpTime//24

			if tmpRel not in self.data_dict["r-[e]"].keys():
				self.data_dict["r-[e]"][tmpRel] = {}
				self.data_dict["r-[e]"][tmpRel]["s"] = set()
				self.data_dict["r-[e]"][tmpRel]["o"] = set()

				self.data_dict["r-[e]"][tmpRel]["s"].add(tmpHead)
				self.data_dict["r-[e]"][tmpRel]["o"].add(tmpTail)
			else:
				self.data_dict["r-[e]"][tmpRel]["s"].add(tmpHead)
				self.data_dict["r-[e]"][tmpRel]["o"].add(tmpTail)

			if tmpTime not in self.data_dict["T-e-r-e"].keys():
				self.data_dict["T-e-r-e"][tmpTime] = {}

				self.data_dict["T-e-r-e"][tmpTime][tmpHead] = {}
				self.data_dict["T-e-r-e"][tmpTime][tmpHead][tmpRel] = []
				self.data_dict["T-e-r-e"][tmpTime][tmpHead][tmpRel].append(tmpTail)

				self.data_dict["T-e-r-e"][tmpTime][tmpTail] = {}
				self.data_dict["T-e-r-e"][tmpTime][tmpTail][-tmpRel] = []
				self.data_dict["T-e-r-e"][tmpTime][tmpTail][-tmpRel].append(tmpHead)
			else:
				if tmpHead not in self.data_dict["T-e-r-e"][tmpTime].keys():
					self.data_dict["T-e-r-e"][tmpTime][tmpHead] = {}
					self.data_dict["T-e-r-e"][tmpTime][tmpHead][tmpRel] = []
					self.data_dict["T-e-r-e"][tmpTime][tmpHead][tmpRel].append(tmpTail)
				else:
					if tmpRel not in self.data_dict["T-e-r-e"][tmpTime][tmpHead].keys():
						self.data_dict["T-e-r-e"][tmpTime][tmpHead][tmpRel] = []
						self.data_dict["T-e-r-e"][tmpTime][tmpHead][tmpRel].append(tmpTail)
					else:
						self.data_dict["T-e-r-e"][tmpTime][tmpHead][tmpRel].append(tmpTail)

				if tmpTail not in self.data_dict["T-e-r-e"][tmpTime].keys():
					self.data_dict["T-e-r-e"][tmpTime][tmpTail] = {}
					self.data_dict["T-e-r-e"][tmpTime][tmpTail][-tmpRel] = []
					self.data_dict["T-e-r-e"][tmpTime][tmpTail][-tmpRel].append(tmpHead)
				else:
					if -tmpRel not in self.data_dict["T-e-r-e"][tmpTime][tmpTail].keys():
						self.data_dict["T-e-r-e"][tmpTime][tmpTail][-tmpRel] = []
						self.data_dict["T-e-r-e"][tmpTime][tmpTail][-tmpRel].append(tmpHead)
					else:
						self.data_dict["T-e-r-e"][tmpTime][tmpTail][-tmpRel].append(tmpHead)

			if tmpHead not in tmp_dict["e-e-T-r"].keys():
				tmp_dict["e-e-T-r"][tmpHead] = {}
				tmp_dict["e-e-T-r"][tmpHead][tmpTail] = {}
				tmp_dict["e-e-T-r"][tmpHead][tmpTail][tmpTime] = []
				tmp_dict["e-e-T-r"][tmpHead][tmpTail][tmpTime].append(tmpRel)

			else:
				if tmpTail not in tmp_dict["e-e-T-r"][tmpHead].keys():
					tmp_dict["e-e-T-r"][tmpHead][tmpTail] = {}
					tmp_dict["e-e-T-r"][tmpHead][tmpTail][tmpTime] = []
					tmp_dict["e-e-T-r"][tmpHead][tmpTail][tmpTime].append(tmpRel)
				else:
					if tmpTime not in tmp_dict["e-e-T-r"][tmpHead][tmpTail].keys():
						tmp_dict["e-e-T-r"][tmpHead][tmpTail][tmpTime] = []
						tmp_dict["e-e-T-r"][tmpHead][tmpTail][tmpTime].append(tmpRel)
					else:
						tmp_dict["e-e-T-r"][tmpHead][tmpTail][tmpTime].append(tmpRel)

			if tmpTail not in tmp_dict["e-e-T-r"].keys():
				tmp_dict["e-e-T-r"][tmpTail] = {}
				tmp_dict["e-e-T-r"][tmpTail][tmpHead] = {}
				tmp_dict["e-e-T-r"][tmpTail][tmpHead][tmpTime] = []
				tmp_dict["e-e-T-r"][tmpTail][tmpHead][tmpTime].append(-tmpRel)
			else:
				if tmpHead not in tmp_dict["e-e-T-r"][tmpTail].keys():
					tmp_dict["e-e-T-r"][tmpTail][tmpHead] = {}
					tmp_dict["e-e-T-r"][tmpTail][tmpHead][tmpTime] = []
					tmp_dict["e-e-T-r"][tmpTail][tmpHead][tmpTime].append(-tmpRel)
				else:
					if tmpTime not in tmp_dict["e-e-T-r"][tmpTail][tmpHead].keys():
						tmp_dict["e-e-T-r"][tmpTail][tmpHead][tmpTime] = []
						tmp_dict["e-e-T-r"][tmpTail][tmpHead][tmpTime].append(-tmpRel)
					else:
						tmp_dict["e-e-T-r"][tmpTail][tmpHead][tmpTime].append(-tmpRel)

			if tmpHead not in tmp_dict["e-r-T-e"].keys():
				tmp_dict["e-r-T-e"][tmpHead] = {}
				tmp_dict["e-r-T-e"][tmpHead][tmpRel] = {}
				tmp_dict["e-r-T-e"][tmpHead][tmpRel][tmpTime] = []
				tmp_dict["e-r-T-e"][tmpHead][tmpRel][tmpTime].append(tmpTail)

			else:
				if tmpRel not in tmp_dict["e-r-T-e"][tmpHead].keys():
					tmp_dict["e-r-T-e"][tmpHead][tmpRel] = {}
					tmp_dict["e-r-T-e"][tmpHead][tmpRel][tmpTime] = []
					tmp_dict["e-r-T-e"][tmpHead][tmpRel][tmpTime].append(tmpTail)

				else:
					if tmpTime not in tmp_dict["e-r-T-e"][tmpHead][tmpRel].keys():
						tmp_dict["e-r-T-e"][tmpHead][tmpRel][tmpTime] = []
						tmp_dict["e-r-T-e"][tmpHead][tmpRel][tmpTime].append(tmpTail)
					else:
						tmp_dict["e-r-T-e"][tmpHead][tmpRel][tmpTime].append(tmpTail)

			if tmpTail not in tmp_dict["e-r-T-e"].keys():
				tmp_dict["e-r-T-e"][tmpTail] = {}
				tmp_dict["e-r-T-e"][tmpTail][-tmpRel] = {}
				tmp_dict["e-r-T-e"][tmpTail][-tmpRel][tmpTime] = []
				tmp_dict["e-r-T-e"][tmpTail][-tmpRel][tmpTime].append(tmpHead)

			else:
				if -tmpRel not in tmp_dict["e-r-T-e"][tmpTail].keys():
					tmp_dict["e-r-T-e"][tmpTail][-tmpRel] = {}
					tmp_dict["e-r-T-e"][tmpTail][-tmpRel][tmpTime] = []
					tmp_dict["e-r-T-e"][tmpTail][-tmpRel][tmpTime].append(tmpHead)

				else:
					if tmpTime not in tmp_dict["e-r-T-e"][tmpTail][-tmpRel].keys():
						tmp_dict["e-r-T-e"][tmpTail][-tmpRel][tmpTime] = []
						tmp_dict["e-r-T-e"][tmpTail][-tmpRel][tmpTime].append(tmpHead)
					else:
						tmp_dict["e-r-T-e"][tmpTail][-tmpRel][tmpTime].append(tmpHead)

		time_num = self.data_dict["all_time"]
		
		#find update and build rel
		self.data_dict["e-e-[r]"] = {}
		for key in tmp_dict["e-e-T-r"].keys():
			self.data_dict["e-e-[r]"][key] = {}
			for in_key in tmp_dict["e-e-T-r"][key].keys():
				self.data_dict["e-e-[r]"][key][in_key] = []
				for i in range(time_num):
					if i in tmp_dict["e-e-T-r"][key][in_key].keys():
						self.data_dict["e-e-[r]"][key][in_key].append(tmp_dict["e-e-T-r"][key][in_key][i])
					else:
						self.data_dict["e-e-[r]"][key][in_key].append([])					

		#find proped nodes
		self.data_dict["e-[r,e]"] = {}
		for key in self.data_dict["e-e-[r]"].keys():
			self.data_dict["e-[r,e]"][key] = []
			for i in range(time_num):
				if key in self.data_dict["T-e-r-e"][i].keys():
					tmp_list = []
					rel_list = self.data_dict["T-e-r-e"][i][key].keys()
					for rel_key in rel_list:
						ent_list = self.data_dict["T-e-r-e"][i][key][rel_key] 
						for ent_key in ent_list:
							tmp_list.append([rel_key,ent_key])
					self.data_dict["e-[r,e]"][key].append(tmp_list)
				else:
					self.data_dict["e-[r,e]"][key].append([])

		#find init
		self.data_dict["[r,e]-e"] = {}
		for key in self.data_dict["e-[r,e]"].keys():
			tmp_list = self.data_dict["e-[r,e]"][key]
			for r_e_list in tmp_list:
				for r_e in r_e_list:
					new_key = " ".join('%s' %id for id in r_e)
					if new_key not in self.data_dict["[r,e]-e"].keys():
						self.data_dict["[r,e]-e"][new_key] = set()
						self.data_dict["[r,e]-e"][new_key].add(key)
					else:
						self.data_dict["[r,e]-e"][new_key].add(key)

		dict_path = self.base_path + "data_dict.pickle"
		output = open(dict_path, "wb")
		pickle.dump(self.data_dict, output)
		output.close()

		return 0


