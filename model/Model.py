import Loss
import torch
import math
import time
import numpy as np
import random as rd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):

	def __init__(self, numOfEntity, numOfRelation, numOfhidden, dataset, raw_ent, raw_rel, init_type):
		super(Model,self).__init__()
		self.numOfEntity = numOfEntity
		self.numOfRelation = numOfRelation
		self.numOfhidden = numOfhidden 
		self.dataset = dataset
		self.init_type = init_type
		self.ns = 20
		self.numOfbase = 200
		self.graph_sample = 10
		self.his_length = 2

		sqrtR = self.numOfhidden**0.5
		sqrtE = self.numOfbase**0.5

		self.relation_embeddings_1 = nn.Embedding(self.numOfRelation+1, self.numOfhidden, padding_idx=self.numOfRelation).cuda()
		self.relation_embeddings_1.weight.data[0:numOfRelation] = torch.cuda.FloatTensor(self.numOfRelation, self.numOfhidden).uniform_(-1./sqrtR, 1./sqrtR)
		#self.relation_embeddings_1.weight.requires_grad = False

		#self.relation_embeddings = nn.Embedding.from_pretrained(raw_rel) #raw_rel
		#self.relation_embeddings.weight.requires_grad = True

		self.entity_weights = nn.Embedding(self.numOfEntity+1, self.numOfbase, padding_idx=self.numOfEntity).cuda()
		self.entity_weights.weight.data[0:self.numOfEntity] = torch.cuda.FloatTensor(self.numOfEntity, self.numOfbase).uniform_(-1./sqrtE, 1./sqrtE)
		#self.entity_weights.weight.data[0:self.numOfEntity] = F.normalize(self.entity_weights.weight.data[0:self.numOfEntity])
		self.entity_weights.weight.requires_grad = False


		#print(raw_ent.size())
		#self.entity_weights = nn.Embedding.from_pretrained(raw_ent) #raw_ent
		#self.entity_weights.weight.requires_grad = True

		self.base_matrix = nn.Linear(self.numOfbase, self.numOfhidden, bias = False)
		nn.init.xavier_normal_(self.base_matrix.weight.data)
		self.update_matrix = nn.Linear(2*self.numOfhidden, self.numOfhidden, bias = False)
		nn.init.xavier_normal_(self.update_matrix.weight.data)
		self.build_matrix = nn.Linear(2*self.numOfhidden, self.numOfhidden, bias = False)
		nn.init.xavier_normal_(self.build_matrix.weight)
		self.dir_influence_matrix = nn.Linear(2*self.numOfbase, self.numOfbase, bias = False)
		nn.init.xavier_normal_(self.dir_influence_matrix.weight.data)
		self.prop_influence_matrix = nn.Linear(3*self.numOfbase, self.numOfbase, bias = False)
		nn.init.xavier_normal_(self.prop_influence_matrix.weight.data)
		self.gate_matrix = nn.Linear(3*self.numOfbase, self.numOfbase, bias = False)
		nn.init.xavier_normal_(self.gate_matrix.weight.data)
		self.project_matrix = nn.Linear(2*self.numOfbase, self.numOfbase, bias = True)
		nn.init.xavier_normal_(self.project_matrix.weight.data)
		self.init_matrix = nn.Linear(self.numOfbase, self.numOfbase, bias = True)
		nn.init.xavier_normal_(self.init_matrix.weight.data)

		self.merge_matrix_d = nn.Linear(self.numOfbase, self.numOfbase, bias = False)
		nn.init.xavier_normal_(self.merge_matrix_d.weight.data)

		self.merge_matrix_p = nn.Linear(self.numOfbase, self.numOfbase, bias = False)
		nn.init.xavier_normal_(self.merge_matrix_p.weight.data)

		self.predict_matrix = nn.Linear(self.numOfbase, self.numOfbase, bias = False)
		nn.init.xavier_normal_(self.predict_matrix.weight.data)
		
		self.score_vector_1 = torch.nn.Parameter(data = torch.cuda.FloatTensor(1, self.numOfbase), requires_grad = True)
		nn.init.xavier_normal_(self.score_vector_1)

		self.score_vector_2 = torch.nn.Parameter(data = torch.cuda.FloatTensor(1, self.numOfbase), requires_grad = True)
		nn.init.xavier_normal_(self.score_vector_2)

		self.period_matrix = nn.Linear(2*self.numOfbase, self.numOfbase, bias = False)
		nn.init.xavier_normal_(self.period_matrix.weight.data)

		self.trade_vector = torch.nn.Parameter(data = torch.cuda.FloatTensor(10, self.numOfbase), requires_grad = True)
		nn.init.xavier_normal_(self.trade_vector)

		self.bias = torch.nn.Parameter(data = torch.cuda.FloatTensor(1, self.numOfbase), requires_grad = True)
		nn.init.xavier_normal_(self.bias)

		self.time_embedding = torch.nn.Parameter(data = torch.cuda.FloatTensor(1, self.numOfbase), requires_grad = True)
		nn.init.xavier_normal_(self.time_embedding)

		self.lamda = torch.nn.Parameter(data = torch.cuda.FloatTensor(1, 1), requires_grad = True)
		nn.init.xavier_normal_(self.time_embedding)

		#self.combiner_u = GRUCell_u(self.numOfbase, self.numOfbase, self.his_length, bias=False)
		#self.combiner_b = GRUCell_b(self.numOfbase, self.numOfbase, self.his_length, bias=False)
		#self.combiner_p = GRUCell_p(self.numOfbase, self.numOfbase, self.his_length, bias=False)

		if self.dataset == "ICEWS18":
			self.update = updater(self.numOfbase, self.numOfbase, self.his_length, bias=True)
		else:
			self.update = updater(self.numOfbase, self.numOfbase, self.his_length, bias=False)

		self.entity_weights_copy = nn.Embedding.from_pretrained(self.entity_weights.weight.clone()).cuda()
		self.entity_weights_copy.weight.requires_grad = False

		self.his_e = torch.cuda.LongTensor(self.numOfEntity+1,self.graph_sample) #torch.zeros(self.numOfEntity+1, self.numOfbase, requires_grad=False).cuda()
		self.his_r = torch.cuda.LongTensor(self.numOfEntity+1,self.graph_sample)
		self.trend = torch.zeros(self.entity_weights.weight.data.size(), requires_grad=True).cuda()

		self.all_ent_list = [i for i in range(self.numOfEntity+1)]
		self.step_dict = torch.zeros(self.numOfEntity+1, requires_grad=False).cuda()

		self.act = nn.Tanh()
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(p=0.0)
		self.score_func = DistMult() #TransE(1) #ConvE(self.numOfbase)

	def init_weights(self, weights, new_entities, related_nodes, step):
		self.step_dict[new_entities] = step
		if self.init_type == "average":
			random_weights = weights[new_entities]
			#print(random_weights.size())
			related_node_weights = torch.mean(weights[related_nodes],-2)#.unsqueeze(0).repeat(random_weights.size()[0], 1) #new_entity*graphsample*base
			#print(related_embeddings.size())
			new_embeddings = self.init_matrix(related_node_weights) #new_entity*base
			weights[new_entities] = related_node_weights+random_weights #+self.score_vector_1

	def reset(self):
		self.step_dict = torch.zeros(self.numOfEntity+1, requires_grad=False).cuda()
		#self.entity_weights.weight.data = self.entity_weights_copy.clone() #nn.Embedding.from_pretrained(self.entity_weights_copy.weight.data)
		self.entity_weights = nn.Embedding.from_pretrained(self.entity_weights_copy.weight.clone()).cuda()
		self.entity_weights.weight.requires_grad = False

		self.his_e[:] = self.numOfEntity
		self.his_r[:] = self.numOfRelation
		self.trend = torch.zeros(self.entity_weights.weight.data.size(), requires_grad=True).cuda()
		'''
		with torch.no_grad():
			self.step_dict = torch.zeros(self.numOfEntity+1, requires_grad=False).cuda()
			self.entity_weights.weight.data = self.entity_weights_copy.clone() #nn.Embedding.from_pretrained(self.entity_weights_copy.weight.data)
			self.entity_weights = nn.Embedding.from_pretrained(self.entity_weights_copy.weight.clone())
			self.entity_weights.weight.requires_grad = False
		'''
	def masked_maxpool(self, A):

		tmp_max = torch.max(A, -2)[0]
		tmp_min = torch.min(-A, -2)[0]

		tmp_max = tmp_max-tmp_min*(tmp_max <= 0.0).float()
		return tmp_max

	def bilinear_agg(self, A):
		A_proj = self.agg_matrix(A)
		tmp = torch.bmm(torch.transpose(A_proj,1,2), A)
		out = torch.sum(A, 1)

		return out

	def get_time_embedding(self, time):
		in_embedding = self.time_embedding*time.unsqueeze(1).cuda() +self.bias
		cos_emb = torch.cos(in_embedding).unsqueeze(1)*((2/self.numOfbase)**(1/2))
		sin_emb = torch.sin(in_embedding).unsqueeze(1)*((2/self.numOfbase)**(1/2))

		combined = torch.cat([cos_emb, sin_emb], 1).transpose(2,1)
		out = torch.flatten(combined, start_dim=1)

		return out


	def forward(self, i, new_nodes, active_nodes, interact_message, \
				prop_nodes, prop_message, \
				pos_sample, neg_sample, steps, appeared_nodes, related_nodes):
		
		tmp_weights = self.entity_weights(torch.cuda.LongTensor(self.all_ent_list))
		#print(self.entity_weights.weight.requires_grad)
		#print(self.relation_embeddings_1.weight.data[0][0:10])
		Loss = 0
		for step in range(len(new_nodes)):
			new_weights, norm1, norm2 = self.update_weights(tmp_weights, new_nodes[step].cuda(), active_nodes[step].cuda(), interact_message[step].cuda(), \
								prop_nodes[step].cuda(), prop_message[step].cuda(), steps[step], active_nodes[step].cuda(), related_nodes[step].cuda())
			Loss += self.caculate_loss_L(new_weights, pos_sample[step][0].cuda(), neg_sample[step][0].cuda(), steps[step], active_nodes[step].cuda())
			if self.dataset == "YAGO":
				Loss += 0.1*self.caculate_loss_F(new_weights, pos_sample[step][1].cuda(), neg_sample[step][1].cuda(), steps[step]+1, active_nodes[step].cuda(), new_nodes[step].cuda())
			else:
				Loss += self.caculate_loss_F(new_weights, pos_sample[step][1].cuda(), neg_sample[step][1].cuda(), steps[step]+1, active_nodes[step].cuda(), new_nodes[step].cuda())
			
			if self.dataset == "YAGO":
				Loss += 0.001*norm1
			
			if self.dataset == "ICEWS18":
				Loss += 0.001*norm1+0.001*norm2#+0.01*torch.norm(self.entity_emb.weight)
			if self.dataset == "ICEWS14":
				Loss += 0.01*norm1+0.01*norm2
			if self.dataset == "ICEWS0515":
				Loss += 0.1*norm1+0.1*norm2
			
			
			
		with torch.no_grad():
			self.entity_weights.weight[0:self.numOfEntity] = tmp_weights[0:self.numOfEntity]#.clone() #.detach()
			self.entity_weights.weight.requires_grad = False
		
		
		norm_loss = 0
		
		
		
		for W in self.gate_matrix.parameters():
			norm_loss += W.norm(2)
		'''
		for W in self.update_matrix.parameters():
			norm_loss += W.norm(2)
		'''

		return Loss# + 0.01*norm_loss


	def update_weights(self, weights, new_entities, active_nodes, interact_message, prop_nodes, prop_message, step, appeared_nodes, related_nodes):
		#generate interaction information for direct influence
		self.init_weights(weights, new_entities, related_nodes, step)
		active_nodes_weights = weights[active_nodes] #self.get_weights(weights, active_nodes, step, appeared_nodes)

		rel_embedding = self.get_rel_weights(interact_message[:,0], step)#.detach()
		inter_e_weights = self.read_weights(weights[interact_message[:,1].flatten()]).view(-1, self.graph_sample, self.numOfbase) #self.get_weights(weights, interact_message_b[:,1].flatten(), step, appeared_nodes).view(-1, self.graph_sample, self.numOfbase)
		#read_h = torch.sigmoid(self.gate_matrix(torch.cat([active_nodes_weights.unsqueeze(1).repeat(1,self.graph_sample,1)-inter_e_weights_build, rel_b_embedding],-1)))*(inter_e_weights_build.sum(2)!=0).unsqueeze(2).float()
		#read_t = torch.tanh(self.gate_matrix(torch.cat([active_nodes_weights.unsqueeze(1).repeat(1,self.graph_sample,1),inter_e_weights_build,rel_b_embedding],-1)))*(inter_e_weights_build.sum(2)!=0).unsqueeze(2).float()
		interact_information = self.interact(active_nodes_weights.unsqueeze(1).repeat(1,self.graph_sample,1), inter_e_weights, rel_embedding) #self.ccorr(inter_e_weights_build.view(-1,self.numOfbase), rel_1_embedding.view(-1,self.numOfbase)).view(rel_2_embedding.size()) #inter_e_weights_build - rel_b_embedding #rel_b_embedding*inter_e_weights_build #rel_b_embedding*read_gate*inter_e_weights_build #torch.tanh(self.build_matrix(torch.cat([rel_b_embedding, inter_e_weights_build], -1))) #self.combiner_b(inter_e_weights_build, rel_b_embedding, active_nodes_weights.unsqueeze(1).repeat(1,self.graph_sample,1), step-self.step_dict[active_nodes])*(inter_e_weights_build.sum(2)!=0).unsqueeze(2).float() #torch.tanh(self.dir_influence_matrix(torch.cat([rel_b_embedding,inter_e_weights_build], -1)))*(inter_e_weights_build.sum(2)!=0).unsqueeze(2).float()

		interact_e = interact_message[:,1].view(len(active_nodes), -1)
		interact_r = torch.abs(interact_message[:,0]).view(len(active_nodes), -1)
		
		#inter_e_weights_update = weights[interact_message_u[:,2].flatten()].view(-1, self.graph_sample, self.numOfbase) #self.get_weights(weights, interact_message_u[:,2].flatten(), step, appeared_nodes).view(-1, self.graph_sample, self.numOfbase)
		#inter_e_weights_build = weights[interact_message_b[:,1].flatten()].view(-1, self.graph_sample, self.numOfbase) #self.get_weights(weights, interact_message_b[:,1].flatten(), step, appeared_nodes).view(-1, self.graph_sample, self.numOfbase)
		#rel_2_embedding = self.relation_embeddings(torch.abs(interact_message_u[:,1])) * (torch.sign(interact_message_u[:,1]).float()+(interact_message_u[:,1]==0).float()).unsqueeze(-1)
		#rel_b_embedding = self.relation_embeddings(torch.abs(interact_message_b[:,0])) * (torch.sign(interact_message_b[:,0]).float()+(interact_message_b[:,0]==0).float()).unsqueeze(-1)
		#interact_e_weights = torch.cat([inter_e_weights_update, inter_e_weights_build], 1)
		#interact_r_emb = torch.cat([rel_2_embedding, rel_b_embedding],1)
		#interact_information = F.normalize(interact_information)
		#caculate direct influence
		#influence = interact_information
		#influence = F.normalize(influence)
		#e_score = torch.sum(self.entity_emb.weight[active_nodes].unsqueeze(1)*self.entity_emb.weight[interact_message[:,1].flatten()].view(-1, self.graph_sample, self.numOfbase), 2, keepdim=True)
		#att = self.masked_softmax(e_score, 1)#active_nodes*graph_sample*1
		#gru_update = self.combiner(influence, active_nodes_weights.unsqueeze(1).repeat(1,2*self.graph_sample,1), step-self.step_dict[active_nodes])*(interact_e_weights.sum(2)!=0).unsqueeze(2).float()
		#print(influence.size())
		direct_influence = torch.sum(interact_information,1)#/((inter_e_weights.sum(2)!=0).sum(1).float() + ((inter_e_weights.sum(2)!=0).sum(1)==0).float()).unsqueeze(1) #torch.sum(interact_e_weights, 1) #torch.sum(att*influence, 1)#/((interact_e_weights.sum(2)!=0).sum(1).float() + ((interact_e_weights.sum(2)!=0).sum(1)==0).float()).unsqueeze(1) #torch.sum(interact_e_weights, 1)
		#print(direct_influence.size())
		#print(direct_influence[0][0:10])
		#direct_influence = F.normalize(direct_influence)

		#generate interaction information for prop influence

		ent_weights = self.read_weights(weights[prop_message[:,:,2].flatten()]).view(-1, self.graph_sample, self.numOfbase) #self.get_weights(weights, prop_message_b[:,:,0].flatten(), step, appeared_nodes).view(-1, self.graph_sample, self.numOfbase)
		#ent_1_embedding_b = self.act(self.base_matrix(ent_1_weights_b_trade))
		#ent_2_weights_b = weights[prop_message_b[:,:,2].flatten()].view(-1, self.graph_sample, self.numOfbase) #self.get_weights(weights, prop_message_b[:,:,2].flatten(), step, appeared_nodes).view(-1, self.graph_sample, self.numOfbase)
		#ent_2_embedding_b = self.act(self.base_matrix(ent_2_weights_b_trade))
		rel_1_embedding = self.get_rel_weights(prop_message[:,:,0], step)#.detach() # * (torch.sign(prop_message_b[:,:,0]).float()+(prop_message_b[:,:,0]==0).float()).unsqueeze(-1)
		rel_2_embedding = self.get_rel_weights(prop_message[:,:,1], step)#.detach() # * (torch.sign(prop_message_b[:,:,1]).float()+(prop_message_b[:,:,1]==0).float()).unsqueeze(-1)
		#interact_information_build = torch.tanh(self.build_matrix(torch.cat([ent_1_embedding_b, rel_embedding, ent_2_embedding_b], -1)))*(ent_2_weights_u_trade.sum(2)!=0).unsqueeze(2).float()

		#interact_information = torch.cat([interact_information_update, interact_information_build], 1)
		#inter_node_weights = torch.cat([ent_1_weights_u, ent_1_weights_b], 1)
		
		#interact_information = F.normalize(interact_information)
		
		#caculate prop influence
		prop_nodes_weights = weights[prop_nodes] #self.get_weights(weights, prop_nodes, step, appeared_nodes)
		prop_information = torch.tanh(self.prop_influence_matrix(torch.cat([rel_1_embedding, rel_2_embedding, ent_weights],-1))) #*(related_node_weights.sum(2)!=0).unsqueeze(2).float() #torch.tanh(self.prop_influence_matrix(interact_information)*((prop_nodes_weights_trade.unsqueeze(1)+related_node_weights)*(related_node_weights.sum(2)!=0).unsqueeze(2).float()))
		#e_score = torch.sum(self.score_vector_2.unsqueeze(1)*influence, 2, keepdim=True)
		#att = self.masked_softmax(e_score, 1)
		#gru_update = self.GRU_cell_p(influence, prop_nodes_weights_trade.unsqueeze(1).repeat(1,2*self.graph_sample,1))*(related_node_weights.sum(2)!=0).unsqueeze(2).float()
		prop_influence = torch.sum(prop_information,1)#/((ent_weights.sum(2)!=0).sum(1).float() + ((ent_weights.sum(2)!=0).sum(1)==0).float()).unsqueeze(1)
		#prop_influence = F.normalize(prop_influence)
		
		#select entities
		new_weights = weights.clone()
		active_nodes = active_nodes.tolist()
		prop_nodes = prop_nodes.tolist()

		corres_nodes = list(set(active_nodes)&set(prop_nodes))
		only_interact_nodes = list(set(active_nodes)-set(corres_nodes))
		only_prop_nodes = list(set(prop_nodes)-set(corres_nodes))
		other_nodes = list(set(self.all_ent_list)-set(active_nodes))

		corres_nodes_index_a = [active_nodes.index(i) for i in corres_nodes]
		corres_nodes_index_p = [prop_nodes.index(i) for i in corres_nodes]
		only_interact_index = [active_nodes.index(i) for i in only_interact_nodes]
		only_prop_index = [prop_nodes.index(i) for i in only_prop_nodes]

		#update node weights
		norm1 = 0
		norm2 = 0
		#pre_weights_trade = self.get_weights(weights, self.all_ent_list, step, appeared_nodes)
		
		if len(corres_nodes) !=0:
			corr_weights = weights[corres_nodes] #self.get_weights(weights,corres_nodes,step,appeared_nodes)
			pre_weights_trade = self.get_weights(weights, corres_nodes, step, appeared_nodes)
			#corr_weights_fix = self.entity_weights_copy(torch.cuda.LongTensor(corres_nodes))
			cur_weights_d = direct_influence[corres_nodes_index_a]
			cur_weights_p = torch.zeros(cur_weights_d.size(), requires_grad=False).cuda()#prop_influence[corres_nodes_index_p]
			#print(cur_weights_p[0][0:10])
			#cur_weights = torch.tanh(cur_weights_d)
			new_corr = self.update(cur_weights_d, cur_weights_p, corr_weights)#(1-gate)*corr_weights + gate*cur_weights
			weights[corres_nodes] = new_corr #.detach() #.clone()#.requires_grad_() #F.normalize(new_corr_trade)#.clone())
			self.trend[corres_nodes] = self.get_trend(self.trend[corres_nodes], corr_weights, new_corr)
			self.step_dict[corres_nodes] = step
			norm1 += torch.norm(pre_weights_trade-new_corr.detach()).mean()
			norm2 += torch.norm(corr_weights-new_corr).mean()

		
		if len(only_interact_nodes) !=0:
			inter_weights = weights[only_interact_nodes] #self.get_weights(weights,only_interact_nodes,step,appeared_nodes)
			#inter_weights_fix = self.entity_weights_copy(torch.cuda.LongTensor(only_interact_nodes))
			pre_weights_trade = self.get_weights(weights, only_interact_nodes, step, appeared_nodes) #self.entity_weights_copy(torch.cuda.LongTensor(only_interact_nodes))
			cur_weights_d = direct_influence[only_interact_index]
			cur_weights_p = torch.zeros(cur_weights_d.size(), requires_grad=False).cuda()
			#gate = torch.sigmoid(self.gate_matrix(torch.cat([inter_weights, cur_weights], 1)))
			#cur_weights = torch.tanh(cur_weights_d)
			new_inter = self.update(cur_weights_d, cur_weights_p, inter_weights) #cur_weights #(1-gate)*inter_weights + gate*cur_weights
			weights[only_interact_nodes] = new_inter#.detach() #.clone()#.requires_grad_() #F.normalize(new_inter_trade)#.clone()
			self.trend[only_interact_nodes] = self.get_trend(self.trend[only_interact_nodes], inter_weights, new_inter)
			self.step_dict[only_interact_nodes] = step
			norm1 += torch.norm(pre_weights_trade-new_inter.detach()).mean() #.sum()
			norm2 += torch.norm(inter_weights-new_inter).mean()
		
		
		
		if len(only_prop_nodes) !=0:
			prop_weights = weights[only_prop_nodes] #self.get_weights(weights, only_prop_nodes, step, appeared_nodes)
			pre_weights_trade = self.get_weights(weights, only_prop_nodes, step, appeared_nodes)
			cur_weights_p = prop_influence[only_prop_index]
			cur_weights_d = torch.zeros(cur_weights_p.size(), requires_grad=False).cuda()
			#cur_weights = torch.tanh(cur_weights_p)
			new_prop = self.update(cur_weights_p, cur_weights_p, prop_weights) #(1-gate)*prop_weights + gate*cur_weights
			weights[only_prop_nodes] = new_prop #.detach() #.clone()#.requires_grad_() #F.normalize(new_prop_trade) #.clone() #new_prop
			self.step_dict[only_prop_nodes] = step
		
			norm1 += torch.norm(pre_weights_trade-new_prop).mean()
			norm2 += torch.norm(prop_weights-new_prop).mean()
		
		self.his_e[active_nodes] = interact_e
		self.his_r[active_nodes] = interact_r
		
		#new_weights[other_nodes] = weights[other_nodes]#.clone()#.requires_grad_() #.requires_grad_()
		#new_weights = F.normalize(new_weights,1,1)
		#norm1 += torch.norm((weights[interact_e.flatten()].view(len(active_nodes),-1, self.numOfbase) - torch.sum(weights[interact_e.flatten()].view(len(active_nodes),-1, self.numOfbase), 1).unsqueeze(1))*(interact_e != 0).unsqueeze(2).float().cuda() )

		return weights, norm1, norm2 #.clone()

	def read_weights(self, weights):
		return self.dropout(weights)

	def get_weights(self, weights, index, step, all_ent):
		
		#weights = F.normalize(weights)
		raw_weights = weights[index]	
		related_e = self.his_e[index]
		
		related_e_weights = weights[related_e.flatten()].view(related_e.size(0),related_e.size(1),-1)#.detach()
		#related_r_weights = self.relation_embeddings_1(related_r)
		#related_r_weights = torch.sum(related_r_weights, 1)
		#score = torch.mm(self.project_matrix(raw_weights), related_e_weights.t())
		#att =  nn.Softmax(1)(score)
		#related_e_weights = torch.sum(related_e_weights.unsqueeze(0)*att.unsqueeze(2),1)
		#print(changes.size())
		time_span = (step-self.step_dict[related_e.flatten()]).view(related_e.size(0),related_e.size(1)).unsqueeze(-1).repeat(1,1,self.numOfbase)
		time_span[time_span<=0] = 0
		time_span[time_span>=3] = 3
		related_e_weights = torch.mean(related_e_weights*torch.sigmoid(time_span*self.lamda), 1)
		#pre_time_emb = self.get_time_embedding(torch.FloatTensor([step]*len(index)))
		#aft_time_emb = self.get_time_embedding(self.step_dict[index].float())
		#span_emb = aft_time_emb - pre_time_emb
		#time_decay = torch.tanh(self.bias*time_span) #aft_time_emb - pre_time_emb
		#print(len(index))
		#org_weights = self.entity_weights_copy(torch.cuda.LongTensor(index))
		
		#all_weights_pooling = torch.mean(weights,0).repeat(len(index),1)#.detach()
		#print((step-self.step_dict[index])+1)
		'''
		pre_time_emb = self.get_time_embedding(torch.FloatTensor([step]*len(index)))
		aft_time_emb = self.get_time_embedding(self.step_dict[index].float())
		time_span = ((step-self.step_dict[index])*(self.step_dict[index] != 0).float()).unsqueeze(1).repeat(1,self.numOfbase)
		time_span[time_span >5] = 5
		time_decay = torch.tanh(self.bias*time_span) #aft_time_emb - pre_time_emb
		'''
		#trade = torch.tanh(self.project_matrix(torch.cat([raw_weights,all_weights_pooling],-1)))#*time_decay
		#sin_matrix = torch.sin(self.period_matrix.unsqueeze(0)*time_span.unsqueeze(1))
		#tanh_matrix = torch.tanh(self.period_matrix.unsqueeze(0)*time_span.unsqueeze(1))
		#gate = torch.sigmoid(self.base_matrix(torch.cat([raw_weights, related_e_weights],-1)))
		#print(self.time_embedding[0][0:10])
		h =  self.relu(self.base_matrix(related_e_weights))#*torch.sigmoid(time_span*self.lamda)
		periods = torch.cuda.FloatTensor([ (i+1)**2 for i in range(self.trade_vector.size()[0])])
		t_tmp = (self.trade_vector*step)//periods.unsqueeze(-1)
		#print(raw_weights.size())
		#print(t_tmp.size())
		t = torch.mean(torch.sin(self.period_matrix(torch.cat([raw_weights.unsqueeze(1).repeat(1,self.trade_vector.size()[0],1), t_tmp.unsqueeze(0).repeat(raw_weights.size()[0], 1, 1)], -1))), 1)
		#t = torch.mean(self.relu(raw_weights.unsqueeze(1)*self.trade_vector.unsqueeze(0)), 1) #raw_weights*torch.tanh(self.time_embedding*time_span)
		#gate = torch.tanh(self.project_matrix(torch.cat([raw_weights,related_e_weights],-1)))#*torch.sigmoid(time_span*self.trade_vector)
		weights_trade = raw_weights + h# + t
		#weights_trade = self.read_weights(weights_trade)
		#weights_trade = (1-gate)*raw_weights + gate*h #*(torch.sin(self.time_embedding*time_span) +  torch.tanh(self.time_embedding*time_span))*related_e_weights #(torch.sin(self.time_embedding*time_span)+torch.tanh(self.bias*time_span))*raw_weights #raw_weights+(torch.sin(raw_weights*time_span)+torch.tanh(self.trade_matrix(raw_weights)*time_span)) # + trade # + trade #self.update(raw_weights, time_decay) #raw_weights+trade*raw_weights#.detach() #*(1+trade) #(1+trade)*raw_weights #(1-time_decay)*trade + time_decay*raw_weights #*gate #*all_weights_pooling #0.1*self.bias*self.relu(step-self.step_dict[index]).unsqueeze(1).repeat(1,self.numOfbase)  #gate*(step-self.step_dict[index]).unsqueeze(1).repeat(1,self.numOfbase) #*time_decay #*time_decay # + 0.1*self.bias*(step-self.step_dict[index]).unsqueeze(1).repeat(1,self.numOfbase) # + gate#*time_decay #gate*raw_weights + (1-gate)*all_weights_pooling #*(step-self.step_dict[index]).unsqueeze(1).repeat(1,self.numOfbase)
		
		#print(((1-time_decay)*self.bias).size())
		return weights_trade
		

	def caculate_loss_L(self, weights, triples, neg_sample, step, appeared_nodes):
		#self.init_weights(weights, new_entities, appeared_nodes, step)
		#generate embedding
		#tmp_embeddings = nn.Embedding.from_pretrained(embeddings)
		#tmp_embeddings = nn.Embedding(self.numOfEntity+1, self.numOfbase, padding_idx=self.numOfEntity).cuda()
		#tmp_embeddings.weight.requires_grad = False
		#tmp_embeddings.weight[0:self.numOfEntity] = embeddings#.clone()
		#print(weights[0])
		#caculate loss
		#print(triples.size())
		#weights_clone = weights#/torch.sum(torch.abs(weights),1).unsqueeze(1) #.clone()
		#print(weights_clone[0][0:10])
		#weights[ents] = weights[ents] + torch.tanh(self.project_matrix(weights[ents]))*0.01*(step-self.step_dict[ents]).unsqueeze(1)#.clone()#.requires_grad_()
		all_embeddings = weights
		
		pos_h_embedding = all_embeddings[triples[:,0]]#self.base_matrix(weights[triples[:,0]])
		pos_r_embedding = self.get_rel_weights(triples[:,1], step)
		pos_t_embedding = all_embeddings[triples[:,2]]#self.base_matrix(weights[triples[:,2]])

		pos_score = self.score_func(pos_h_embedding,pos_r_embedding,pos_t_embedding).unsqueeze(1)#.detach()
		#pos_score = torch.norm(pos_h_embedding+pos_r_embedding-pos_t_embedding, 2, 1).unsqueeze(1)

		#print(neg_sample.size())
		neg_h_embedding = all_embeddings[neg_sample[0]]#.view(pos_h_embedding.size()[0],-1,self.numOfhidden)#.clone() #.detach() #self.base_matrix(weights[neg_sample[0]]).view(pos_h_embedding.size()[0],-1,self.numOfhidden)
		#neg_r_embedding = self.relation_embeddings(neg_sample[1]).view(-1,self.numOfhidden)
		neg_t_embedding = all_embeddings[neg_sample[2]]#.view(pos_h_embedding.size()[0],-1,self.numOfhidden)#.clone() #.detach() #self.base_matrix(weights[neg_sample[2]]).view(pos_h_embedding.size()[0],-1,self.numOfhidden)

		neg_score_h = self.score_func(neg_h_embedding,pos_r_embedding.unsqueeze(1).repeat(1,self.ns,1).view(-1,self.numOfbase),pos_t_embedding.unsqueeze(1).repeat(1,self.ns,1).view(-1,self.numOfbase)).view(-1,self.ns) #.detach()
		#neg_score_r = self.score_func(pos_h_embedding.unsqueeze(1).repeat(1,10,1).view(-1,self.numOfbase),neg_r_embedding,pos_t_embedding.unsqueeze(1).repeat(1,10,1).view(-1,self.numOfbase)).view(-1,10) #.sigmoid()
		neg_score_t = self.score_func(pos_h_embedding.unsqueeze(1).repeat(1,self.ns,1).view(-1,self.numOfbase),pos_r_embedding.unsqueeze(1).repeat(1,self.ns,1).view(-1,self.numOfbase),neg_t_embedding).view(-1,self.ns) #.detach()
		
		#neg_score_h = torch.norm((neg_h_embedding - pos_t_embedding.unsqueeze(1))+pos_r_embedding.unsqueeze(1), 2, 2)
		#neg_score_r = torch.sum(pos_h_embedding.unsqueeze(1) - pos_t_embedding.unsqueeze(1), 2, 2)
		#neg_score_t = torch.norm((pos_h_embedding.unsqueeze(1) - neg_t_embedding)+pos_r_embedding.unsqueeze(1), 2, 2)

		#neg_score_h = torch.sum(self.predict_vector.unsqueeze(1)*torch.tanh(self.predict_matrix(torch.cat([neg_h_embedding, pos_r_embedding.unsqueeze(1).repeat(1,neg_h_embedding.size()[1],1), pos_t_embedding.unsqueeze(1).repeat(1,neg_h_embedding.size()[1],1)], -1))), 2)
		#neg_score_t = torch.sum(self.predict_vector.unsqueeze(1)*torch.tanh(self.predict_matrix(torch.cat([pos_h_embedding.unsqueeze(1).repeat(1,neg_h_embedding.size()[1],1),pos_r_embedding.unsqueeze(1).repeat(1,neg_h_embedding.size()[1],1), neg_t_embedding], -1))), 2)
		#neg_score_h = -self.ConvE(neg_h_embedding, pos_r_embedding.unsqueeze(1).repeat(1,neg_h_embedding.size()[1],1), pos_t_embedding.unsqueeze(1).repeat(1,neg_h_embedding.size()[1],1))
		#Eneg_score_t = -self.ConvE(pos_t_embedding.unsqueeze(1).repeat(1,neg_t_embedding.size()[1],1), pos_r_embedding.unsqueeze(1).repeat(1,neg_t_embedding.size()[1],1), neg_t_embedding)
		
		#becloss
		'''
		labels_p = torch.ones(pos_score.size()).float().cuda()
		labels_n = torch.zeros(neg_score_h.size()).float().cuda()
		labels = torch.cat([labels_p, labels_n], 1)

		scores_h = torch.cat([pos_score, neg_score_h], 1)
		scores_t = torch.cat([pos_score, neg_score_t], 1)
		
		Loss_func =  nn.BCEWithLogitsLoss()

		h_loss = Loss_func(scores_h, labels)
		t_loss = Loss_func(scores_t, labels)
		'''
		
		#margin loss
		
		h_loss = self.get_loss(pos_score, neg_score_h)
		#r_loss = self.get_loss(pos_score, neg_score_r)
		t_loss = self.get_loss(pos_score, neg_score_t)
		
		#softpuls loss
		#Loss_func = torch.nn.Softplus()
		#h_loss = torch.sum(Loss_func(pos_score))
		#t_loss = torch.sum(Loss_func(pos_score))



		#norm loss
		norm_loss = 0
		norm_loss += torch.norm(pos_r_embedding)# + torch.norm(pos_h_embedding) + Loss.normLoss(pos_t_embedding)# + Loss.normLoss(neg_t_embedding,1) #torch.mean(pos_h_embedding ** 2) + torch.mean(pos_r_embedding ** 2) + torch.mean(pos_t_embedding ** 2) + torch.mean(neg_h_embedding ** 2) + torch.mean(neg_t_embedding ** 2)
		#norm_loss += self.update_matrix.weight.norm(2) + self.build_matrix.weight.norm(2) + self.prop_matrix.weight.norm(2)

		#print(h_loss + t_loss)
		#print(norm_loss)
		return h_loss + t_loss# + 0.01*norm_loss

	def caculate_loss_F(self, weights, triples, neg_sample, step, appeared_nodes, new_entities):
		#self.init_weights(weights, new_entities, appeared_nodes, step)
		#generate embedding
		#tmp_embeddings = nn.Embedding.from_pretrained(embeddings)
		#tmp_embeddings = nn.Embedding(self.numOfEntity+1, self.numOfbase, padding_idx=self.numOfEntity).cuda()
		#tmp_embeddings.weight.requires_grad = False
		#tmp_embeddings.weight[0:self.numOfEntity] = embeddings#.clone()
		#print(weights[0])
		#caculate loss
		#print(triples.size())
		weights_clone = weights #/torch.sum(torch.abs(weights),1).unsqueeze(1) #.clone()
		#print(weights_clone[0][0:10])
		#weights[ents] = weights[ents] + torch.tanh(self.project_matrix(weights[ents]))*0.01*(step-self.step_dict[ents]).unsqueeze(1)#.clone()#.requires_grad_()
		all_embeddings = self.get_weights(weights_clone,self.all_ent_list,step,appeared_nodes) #self.act(self.base_matrix(self.get_weights(weights_clone,self.all_ent_list,step,appeared_nodes)))

		pos_h_embedding = all_embeddings[triples[:,0]]#.detach() #self.base_matrix(weights[triples[:,0]])
		pos_r_embedding = self.get_rel_weights(triples[:,1], step)
		pos_t_embedding = all_embeddings[triples[:,2]]#.detach() #self.base_matrix(weights[triples[:,2]])

		#pos_h_embedding_fix = self.entity_weights_copy(triples[:,0])
		#pos_t_embedding_fix = self.entity_weights_copy(triples[:,2])

		pos_score = self.score_func(pos_h_embedding,pos_r_embedding,pos_t_embedding).unsqueeze(1)#.detach()
		#pos_score = torch.norm(pos_h_embedding+pos_r_embedding-pos_t_embedding, 2, 1).unsqueeze(1)
		
		#print(neg_sample.size())
		neg_h_embedding = all_embeddings[neg_sample[0]]#.view(pos_h_embedding.size()[0],-1,self.numOfhidden)#.detach() #self.base_matrix(weights[neg_sample[0]]).view(pos_h_embedding.size()[0],-1,self.numOfhidden)
		#neg_r_embedding = self.relation_embeddings(neg_sample[1]).view(-1,self.numOfhidden)
		neg_t_embedding = all_embeddings[neg_sample[2]]#.view(pos_h_embedding.size()[0],-1,self.numOfhidden)#.detach() #self.base_matrix(weights[neg_sample[2]]).view(pos_h_embedding.size()[0],-1,self.numOfhidden)

		#neg_h_embedding_fix = self.entity_weights_copy(neg_sample[0]).view(pos_h_embedding.size()[0],-1,self.numOfhidden)
		#neg_t_embedding_fix = self.entity_weights_copy(neg_sample[2]).view(pos_h_embedding.size()[0],-1,self.numOfhidden)

		neg_score_h = self.score_func(neg_h_embedding,pos_r_embedding.unsqueeze(1).repeat(1,self.ns,1).view(-1,self.numOfbase),pos_t_embedding.unsqueeze(1).repeat(1,self.ns,1).view(-1,self.numOfbase)).view(-1,self.ns) #.detach()
		#neg_score_r = self.score_func(pos_h_embedding.unsqueeze(1).repeat(1,10,1).view(-1,self.numOfbase),neg_r_embedding,pos_t_embedding.unsqueeze(1).repeat(1,10,1).view(-1,self.numOfbase)).view(-1,10) #.sigmoid()
		neg_score_t = self.score_func(pos_h_embedding.unsqueeze(1).repeat(1,self.ns,1).view(-1,self.numOfbase),pos_r_embedding.unsqueeze(1).repeat(1,self.ns,1).view(-1,self.numOfbase),neg_t_embedding).view(-1,self.ns) #.detach()
		
		#neg_score_h = torch.norm((neg_h_embedding - pos_t_embedding.unsqueeze(1))+pos_r_embedding.unsqueeze(1), 2, 2)
		#neg_score_r = torch.sum(pos_h_embedding.unsqueeze(1) - pos_t_embedding.unsqueeze(1), 2, 2)
		#neg_score_t = torch.norm((pos_h_embedding.unsqueeze(1) - neg_t_embedding)+pos_r_embedding.unsqueeze(1), 2, 2)

		#neg_score_h = torch.sum(self.predict_vector.unsqueeze(1)*torch.tanh(self.predict_matrix(torch.cat([neg_h_embedding, pos_r_embedding.unsqueeze(1).repeat(1,neg_h_embedding.size()[1],1), pos_t_embedding.unsqueeze(1).repeat(1,neg_h_embedding.size()[1],1)], -1))), 2)
		#neg_score_t = torch.sum(self.predict_vector.unsqueeze(1)*torch.tanh(self.predict_matrix(torch.cat([pos_h_embedding.unsqueeze(1).repeat(1,neg_h_embedding.size()[1],1),pos_r_embedding.unsqueeze(1).repeat(1,neg_h_embedding.size()[1],1), neg_t_embedding], -1))), 2)
		#neg_score_h = -self.ConvE(neg_h_embedding, pos_r_embedding.unsqueeze(1).repeat(1,neg_h_embedding.size()[1],1), pos_t_embedding.unsqueeze(1).repeat(1,neg_h_embedding.size()[1],1))
		#Eneg_score_t = -self.ConvE(pos_t_embedding.unsqueeze(1).repeat(1,neg_t_embedding.size()[1],1), pos_r_embedding.unsqueeze(1).repeat(1,neg_t_embedding.size()[1],1), neg_t_embedding)
		
		#becloss
		'''
		labels_p = torch.ones(pos_score.size()).float().cuda()
		labels_n = torch.zeros(neg_score_h.size()).float().cuda()
		labels = torch.cat([labels_p, labels_n], 1)

		scores_h = torch.cat([pos_score, neg_score_h], 1)
		scores_t = torch.cat([pos_score, neg_score_t], 1)
		
		Loss_func =  nn.BCEWithLogitsLoss()

		h_loss = Loss_func(scores_h, labels)
		t_loss = Loss_func(scores_t, labels)
		'''
		
		#margin loss
		
		h_loss = self.get_loss(pos_score, neg_score_h)
		#r_loss = self.get_loss(pos_score, neg_score_r)
		t_loss = self.get_loss(pos_score, neg_score_t)
		
		#softpuls loss
		#Loss_func = torch.nn.Softplus()
		#h_loss = torch.sum(Loss_func(torch.cat([pos_score, neg_score_h],0)))
		#t_loss = torch.sum(Loss_func(torch.cat([pos_score, neg_score_t],0)))


		#norm loss
		norm_loss = 0
		#co_occur_node_weights = all_embeddings[norm1_batch.flatten()].view(-1, self.graph_sample, self.numOfbase)
		#norm_loss  = torch.norm((co_occur_node_weights-torch.sum(co_occur_node_weights, 1).unsqueeze(1))*(torch.sum(co_occur_node_weights, 2).unsqueeze(2) != 0).float().cuda() )
		#norm_loss = torch.norm(pos_h_embedding-pos_t_embedding)
		'''
		norm_loss = torch.norm(pos_h_embedding-pos_h_embedding_fix).mean()+\
					torch.norm(pos_t_embedding-pos_t_embedding_fix).mean()+\
					torch.norm(neg_h_embedding-neg_h_embedding_fix).mean()+\
					torch.norm(neg_t_embedding-neg_t_embedding_fix).mean()
		'''
		norm_loss += torch.norm(pos_r_embedding)# + torch.norm(pos_t_embedding) + torch.norm(pos_h_embedding)# + Loss.normLoss(pos_h_embedding,1) + Loss.normLoss(pos_t_embedding,1) + Loss.normLoss(neg_h_embedding,1) + Loss.normLoss(neg_t_embedding,1) #torch.mean(pos_h_embedding ** 2) + torch.mean(pos_r_embedding ** 2) + torch.mean(pos_t_embedding ** 2) + torch.mean(neg_h_embedding ** 2) + torch.mean(neg_t_embedding ** 2)
		#norm_loss += self.update_matrix.weight.norm(2) + self.build_matrix.weight.norm(2) + self.prop_matrix.weight.norm(2)

		#print(h_loss + t_loss)
		#print(norm_loss)
		return h_loss + t_loss# + 0.01*norm_loss


	def forecast(self, weights, all_ent, triples, step, appeared_nodes, new_entities, batch_data): # low efficency because of the gpu memory
		result_e = [0, 0, 0, 0, 0]# MRR H1 H3 H5 H10
		result_r = [0, 0, 0, 0, 0]
		#generate embedding
		#tmp_weights = nn.Embedding.from_pretrained(weights)
		#weights = F.normalize(weights, 2, 1)
		weights_clone = weights #F.normalize(weights)
		#F.normalize(weights) #/torch.sum(torch.abs(weights),1).unsqueeze(1)#F.normalize(weights)
		#print(weights_clone[0][0:10])
		#weights[all_ent] = weights[all_ent] + torch.tanh(self.project_matrix(weights[all_ent]))*0.01*(step-self.step_dict[all_ent]).unsqueeze(1)
		all_embeddings = self.get_weights(weights_clone,self.all_ent_list,step,appeared_nodes) #self.act(self.base_matrix(self.get_weights(weights_clone,self.all_ent_list,step,appeared_nodes))) #self.base_matrix(weights[0:self.numOfEntity])
		#all_embeddings = F.normalize(all_embeddings)
		#print(all_embeddings.size())
		#return
		#appeared_embeddings = all_embeddings[all_ent] #self.get_weights(weights_clone,all_ent,step,appeared_nodes) #self.act(self.base_matrix(self.get_weights(weights_clone,all_ent,step,appeared_nodes)))
		#return
		#appeared_embeddings = F.normalize(appeared_embeddings)
		#print(all_embeddings[0][0:10])

		#all_embeddings = self.base_matrix(embeddings)
		#print(all_embeddings[0][0:10])

		#entity forecast
		for triple in triples:
			h_embedding = all_embeddings[triple[0]]
			r_embedding = self.get_rel_weights(triple[1], step)
			t_embedding = all_embeddings[triple[2]]

			tmp_h_embedding = h_embedding.repeat(len(all_embeddings), 1)
			tmp_r_embedding = r_embedding.repeat(len(all_embeddings), 1)
			tmp_t_embedding = t_embedding.repeat(len(all_embeddings), 1)

			pos_score = self.score_func(tmp_h_embedding, tmp_r_embedding,tmp_t_embedding) #torch.sum(tmp_h_embedding*tmp_t_embedding*tmp_r_embedding, 1)#.sigmoid()
			neg_h_score = self.score_func(all_embeddings, tmp_r_embedding, tmp_t_embedding) #torch.sum(appeared_embeddings*tmp_t_embedding*tmp_r_embedding, 1)#.sigmoid()
			neg_t_score = self.score_func(tmp_h_embedding, tmp_r_embedding, all_embeddings) #torch.sum(tmp_h_embedding*appeared_embeddings*tmp_r_embedding, 1)#.sigmoid()
			#pos_score = torch.norm((tmp_h_embedding - tmp_t_embedding)+tmp_r_embedding, 2, 1)#.sigmoid()
			#neg_h_score = torch.norm((appeared_embeddings - tmp_t_embedding)+tmp_r_embedding, 2, 1)#.sigmoid()
			#neg_t_score = torch.norm((tmp_h_embedding - appeared_embeddings)+tmp_r_embedding, 2, 1)#.sigmoid()
			#pos_score = trch.sum(self.predict_vector.unsqueeze(1)*torch.tanh(self.predict_matrix(torch.cat([tmp_h_embedding, tmp_r_embedding, tmp_t_embedding], -1))), 1)
			#neg_h_score = torch.sum(self.predict_vector.unsqueeze(1)*torch.tanh(self.predict_matrix(torch.cat([all_embeddings, tmp_r_embedding, tmp_t_embedding], -1))), 1)
			#neg_t_score = torch.sum(self.predict_vector.unsqueeze(1)*torch.tanh(self.predict_matrix(torch.cat([tmp_h_embedding, tmp_r_embedding, all_embeddings], -1))), 1)
			#pos_score = self.ConvE(tmp_h_embedding, tmp_r_embedding, tmp_t_embedding)
			#neg_h_score = self.ConvE(appeared_embeddings, tmp_r_embedding, tmp_t_embedding)
			#neg_t_score = self.ConvE(tmp_h_embedding, tmp_r_embedding, appeared_embeddings)

			tmp_head = (pos_score - neg_h_score)
			tmp_tail = (pos_score - neg_t_score)

			filter_h = batch_data[int(triple[2])][-int(triple[1])]
			filter_t = batch_data[int(triple[0])][int(triple[1])]
			
			label_h = torch.zeros(len(all_embeddings), requires_grad=False).cuda()
			label_h[all_ent] = 1
			label_h[filter_h] = 0

			label_t = torch.zeros(len(all_embeddings), requires_grad=False).cuda()
			label_t[all_ent] = 1
			label_t[filter_t] = 0

			wrongHead = torch.nonzero(self.relu(tmp_head*label_h))
			wrongTail = torch.nonzero(self.relu(tmp_tail*label_t))

			Rank_H = wrongHead.size()[0]+1
			Rank_T = wrongTail.size()[0]+1

			result_e[0] += 1/Rank_H + 1/Rank_T

			if Rank_H<=1:
				result_e[1]+=1
			if Rank_T<=1:
				result_e[1]+=1

			if Rank_H<=3:
				result_e[2]+=1
			if Rank_T<=3:
				result_e[2]+=1

			if Rank_H<=5:
				result_e[3]+=1
			if Rank_T<=5:
				result_e[3]+=1

			if Rank_H<=10:
				result_e[4]+=1
			if Rank_T<=10:
				result_e[4]+=1

		return result_e

		'''
		#relation forecast
		'''
	def reconstruct(self, weights, all_ent, triples, step,appeared_nodes):
		result_e = [0, 0, 0, 0, 0]# MRR H1 H3 H5 H10
		result_r = [0, 0, 0, 0, 0]
		#generate embedding
		#tmp_weights = nn.Embedding.from_pretrained(weights)
		#weights = F.normalize(weights, 2, 1)
		#weights_clone = weights.clone()
		#F.normalize(weights) #/torch.sum(torch.abs(weights),1).unsqueeze(1)#F.normalize(weights)
		#print(weights_clone[0][0:10])
		#weights[all_ent] = weights[all_ent] + torch.tanh(self.project_matrix(weights[all_ent]))*0.01*(step-self.step_dict[all_ent]).unsqueeze(1)
		all_embeddings = weights #F.normalize(weights)
		#print(all_embeddings[0][0:10])
		#all_embeddings = F.normalize(all_embeddings)
		appeared_embeddings = weights[all_ent] #self.act(self.base_matrix(self.get_weights(weights_clone,all_ent,step,appeared_nodes)))
		#appeared_embeddings = F.normalize(appeared_embeddings)
		#print(len(all_ent))

		#all_embeddings = self.base_matrix(embeddings)
		#print(all_embeddings[0][0:10])

		#entity forecast
		for triple in triples:
			h_embedding = all_embeddings[triple[0]]
			r_embedding = self.get_rel_weights(triple[1], step)
			t_embedding = all_embeddings[triple[2]]

			tmp_h_embedding = h_embedding.repeat(len(all_ent), 1)
			tmp_r_embedding = r_embedding.repeat(len(all_ent), 1)
			tmp_t_embedding = t_embedding.repeat(len(all_ent), 1)

			pos_score = self.score_func(tmp_h_embedding, tmp_r_embedding,tmp_t_embedding) #torch.sum(tmp_h_embedding*tmp_t_embedding*tmp_r_embedding, 1)#.sigmoid()
			neg_h_score = self.score_func(appeared_embeddings, tmp_r_embedding, tmp_t_embedding) #torch.sum(appeared_embeddings*tmp_t_embedding*tmp_r_embedding, 1)#.sigmoid()
			neg_t_score = self.score_func(tmp_h_embedding, tmp_r_embedding, appeared_embeddings) #torch.sum(tmp_h_embedding*appeared_embeddings*tmp_r_embedding, 1)#.sigmoid()
			#pos_score = torch.norm((tmp_h_embedding - tmp_t_embedding)+tmp_r_embedding, 2, 1)#.sigmoid()
			#neg_h_score = torch.norm((appeared_embeddings - tmp_t_embedding)+tmp_r_embedding, 2, 1)#.sigmoid()
			#neg_t_score = torch.norm((tmp_h_embedding - appeared_embeddings)+tmp_r_embedding, 2, 1)#.sigmoid()
			#pos_score = trch.sum(self.predict_vector.unsqueeze(1)*torch.tanh(self.predict_matrix(torch.cat([tmp_h_embedding, tmp_r_embedding, tmp_t_embedding], -1))), 1)
			#neg_h_score = torch.sum(self.predict_vector.unsqueeze(1)*torch.tanh(self.predict_matrix(torch.cat([all_embeddings, tmp_r_embedding, tmp_t_embedding], -1))), 1)
			#neg_t_score = torch.sum(self.predict_vector.unsqueeze(1)*torch.tanh(self.predict_matrix(torch.cat([tmp_h_embedding, tmp_r_embedding, all_embeddings], -1))), 1)
			#pos_score = self.ConvE(tmp_h_embedding, tmp_r_embedding, tmp_t_embedding)
			#neg_h_score = self.ConvE(appeared_embeddings, tmp_r_embedding, tmp_t_embedding)
			#neg_t_score = self.ConvE(tmp_h_embedding, tmp_r_embedding, appeared_embeddings)

			tmp_head = (pos_score - neg_h_score)
			tmp_tail = (pos_score - neg_t_score)

			wrongHead = torch.nonzero(self.relu(tmp_head))
			wrongTail = torch.nonzero(self.relu(tmp_tail))

			Rank_H = wrongHead.size()[0] + 1
			Rank_T = wrongTail.size()[0] + 1

			result_e[0] += 1/Rank_H + 1/Rank_T

			if Rank_H<=1:
				result_e[1]+=1
			if Rank_T<=1:
				result_e[1]+=1

			if Rank_H<=3:
				result_e[2]+=1
			if Rank_T<=3:
				result_e[2]+=1

			if Rank_H<=5:
				result_e[3]+=1
			if Rank_T<=5:
				result_e[3]+=1

			if Rank_H<=10:
				result_e[4]+=1
			if Rank_T<=10:
				result_e[4]+=1

		return result_e

		'''
		#relation forecast


		'''

	def masked_softmax(self, A, dim):
		
		A = A.float()
		A_max = torch.max(A,dim=dim,keepdim=True)[0]
		A_exp = torch.exp(A-A_max)
		A_exp = A_exp * (A != 0).float()
		Sum = torch.sum(A_exp,dim=dim,keepdim=True)
		Sum = Sum + (Sum == 0.0).float()
		score = A_exp / Sum
		
		'''
		A_masked = A*(B != k).float() + (B == k).float()*(-1e6)
		A_exp = torch.exp(A_masked)
		A_sum = torch.sum(A_exp, dim, keepdim = True)
		score = A_exp/(A_sum + (A_sum == 0.0).float())
		'''
		#score = nn.Softmax(dim)(A)
		return score

	def com_mult(self, a, b):
		r1, i1 = a[..., 0], a[..., 1]
		r2, i2 = b[..., 0], b[..., 1]
		out = torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)

		return out

	def conj(self, a):
		a[..., 1] = -a[..., 1]
		return a

	def ccorr(self ,a, b):
		return torch.irfft(self.com_mult(self.conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

	def interact(self, s, r, o):
		#mul
		return r*o#torch.tanh(self.update_matrix(torch.cat([r,o],-1)))
		#sub
		#return o-r
		#ccorr
		#return self.ccorr(a.view(-1,self.numOfbase),b.view(-1,self.numOfbase)).view(a.size())
		#cross
		#return o+r*o
		#concat
		#return self.relu(self.dir_influence_matrix(torch.cat([r,o],-1)))

	def get_loss(self, pos, neg):
		'''
		calc_loss = nn.Softplus()
		score = torch.cat([pos,neg],1)
		label = torch.cat([torch.ones(pos.size()[0],1, requires_grad=False),-torch.ones(pos.size()[0],self.ns, requires_grad=False)],1)
		loss = calc_loss(score*label.cuda())
		loss = torch.mean(loss)
		'''
		'''
		calc_loss = nn.BCELoss()
		score = torch.sigmoid(torch.cat([pos,neg],1))
		label = torch.cat([torch.ones(pos.size()[0],1, requires_grad=False),torch.zeros(pos.size()[0],10, requires_grad=False)],1)
		loss = calc_loss(score,label.cuda())
		loss = torch.mean(loss)
		'''
		
		calc_loss = Loss.marginLoss()
		loss = calc_loss(pos, neg, 3)
		
		return loss	

	def get_rel_weights(self, index, step):
		rel_emb_1 = self.relation_embeddings_1(torch.abs(index))

		'''
		rel_weights = torch.tanh(rel_emb_1*step + rel_emb_2)#*rel_emb_2 #rel_emb_2*torch.cos(rel_emb_1*step) + rel_emb_3
		'''
		'''
		rel_emb_1_p = rel_emb_1*(torch.relu(torch.sign(index).float()+(index==0).float())).unsqueeze(-1)
		rel_emb_1_d = torch.tanh(self.merge_matrix_d(rel_emb_1))*(torch.relu(torch.sign(-index).float())).unsqueeze(-1)
		rel_weights = rel_emb_1+rel_emb_2
		'''
		return rel_emb_1

	def get_trend(self, trend, pre_weights, new_weights):
		return (new_weights - pre_weights)



class GRUCell(nn.Module):

	"""
	An implementation of GRUCell.

	"""

	def __init__(self, input_size, hidden_size, his_length, bias):
		super(GRUCell, self).__init__()
		self.update_gate = nn.Linear(2*input_size, hidden_size, bias=bias)
		nn.init.xavier_normal_(self.update_gate.weight.data)
		self.reset_gate = nn.Linear(2*input_size, hidden_size, bias=bias)
		nn.init.xavier_normal_(self.reset_gate.weight.data)
		self.new_gate = nn.Linear(2*input_size, hidden_size, bias=bias)
		nn.init.xavier_normal_(self.new_gate.weight.data)
		self.merge_matrix = nn.Linear(2*input_size, hidden_size, bias=bias)
		nn.init.xavier_normal_(self.merge_matrix.weight.data)
		self.his_length = his_length

		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x1, x2, hidden):
		x = x1+x2 #torch.relu(self.merge_matrix(torch.cat([x1,x2],-1)))

		z = torch.sigmoid(self.update_gate(torch.cat([x,hidden],-1)))
		r = torch.sigmoid(self.reset_gate(torch.cat([x,hidden],-1)))
		h = torch.tanh(self.new_gate(torch.cat([x,r*hidden],-1)))

		hy = (1-z)*h + z*hidden		

		return hy

class updater(nn.Module):

	"""
	An implementation of GRUCell.

	"""

	def __init__(self, input_size, hidden_size, his_length, bias):
		super(updater, self).__init__()
		self.update_gate = nn.Linear(2*input_size, hidden_size, bias=bias)
		nn.init.xavier_normal_(self.update_gate.weight.data)
		self.reset_gate = nn.Linear(2*input_size, hidden_size, bias=bias)
		nn.init.xavier_normal_(self.reset_gate.weight.data)
		self.new_gate = nn.Linear(2*input_size, hidden_size, bias=bias)
		nn.init.xavier_normal_(self.new_gate.weight.data)
		self.inter_gate = nn.Linear(input_size, hidden_size, bias=bias)
		nn.init.xavier_normal_(self.inter_gate.weight.data)
		self.his_length = his_length

		self.lamda = self.time_embedding = torch.nn.Parameter(data = torch.cuda.FloatTensor(1, 1), requires_grad = True)
		nn.init.xavier_normal_(self.time_embedding)

		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x1, x2, hidden):
		#print(torch.exp(-span).unsqueeze(1))
		#span[span>self.his_length] = self.his_length
		#hidden = hidden*torch.exp(-span).unsqueeze(1)#.unsqueeze(1)
		#x = torch.tanh(self.inter_gate(torch.cat([rel,x],-1)))
		x = x1# + x2
		z = self.sigmoid(self.update_gate(torch.cat([x,hidden],-1)))#*torch.exp(-span).unsqueeze(1)#.unsqueeze(1)
		#z = F.normalize(z,2,1)
		#r = self.sigmoid(self.reset_gate(torch.cat([x,hidden],-1)))#*torch.exp(-span).unsqueeze(1)#.unsqueeze(1)
		h = self.tanh(self.new_gate(torch.cat([x,hidden],-1)))

		#gate = self.sigmoid(self.reset_gate(torch.cat([z,h], -1)))

		#h_r = self.tanh(self.inter_gate(torch.cat([rel,hidden],-1)))
		
		hy = (1-z)*hidden + z*h #(1-z*torch.sigmoid(torch.abs(self.lamda)*span.unsqueeze(1)))*hidden + h*torch.sigmoid(torch.abs(self.lamda)*span.unsqueeze(1))
		#print(torch.min(hy,1)[0].size())
		#hy = F.normalize(hy,2,1) #(hy-torch.min(hy,1)[0].unsqueeze(1))/(torch.max(hy,1)[0].unsqueeze(1)-torch.min(hy,1)[0].unsqueeze(1))
		'''
		gate = torch.tanh(self.gate(torch.cat([x,hidden.repeat(1,x.size()[1],1)], -1)))
		hy = gate*hidden + (1-gate)*x
		'''

		return hy


class ConvTransE(torch.nn.Module):
    def __init__(self):
        super(ConvTransE, self).__init__()
        self.inp_drop = torch.nn.Dropout(0)
        self.hidden_drop = torch.nn.Dropout(0.25)
        self.feature_map_drop = torch.nn.Dropout(0.25)

        self.conv1 =  nn.Conv1d(2, 3, 5, stride=1, padding= int(math.floor(5/2))) # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(200)
        self.bn1 = torch.nn.BatchNorm1d(300)
        self.bn2 = torch.nn.BatchNorm1d(100)
        self.fc = torch.nn.Linear(600,200)
        #self.bn3 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        #self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(100)

    def forward(self, e1, rel, X):

        e1_embedded_all = X #self.bn_init(X)
        e1_embedded = e1.unsqueeze(1) #self.bn_init(e1)
        rel_embedded = rel.unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        #stacked_inputs = self.bn0(stacked_inputs)
        x= stacked_inputs #self.inp_drop(stacked_inputs)
        x= self.conv1(x)
       # print(x.size())
        #x= self.bn1(x)
        #x= F.relu(x)
        #x = self.feature_map_drop(x)
        #print(x.size())
        x = x.view(-1, 600)
        x = self.fc(x)
        # x = self.hidden_drop(x)
        #x = self.bn2(x)
        x = torch.tanh(x)
        x = torch.sum(x*e1_embedded_all,1)#.view(-1)
        #print(x.size())

        return x

class DistMult(nn.Module):
    def __init__(self):
        super(DistMult, self).__init__()
        self.trans = nn.Linear(200,100)
        self.inp_drop = torch.nn.Dropout(0.4)
        self.loss = torch.nn.BCELoss()


    def forward(self, s, r, o):
        #s = self.inp_drop(s)
        #r = self.inp_drop(r)
        #o = self.inp_drop(o)
        #s = F.normalize(s, 2, 1)
        #r = F.normalize(r, 2, 1)
        #o = F.normalize(o, 2, 1)
        #s_trans = torch.tanh(self.trans(torch.cat([s,r],-1)))
        #o_trans = torch.tanh(self.trans(torch.cat([o,r],-1)))
        score = torch.sum(self.inp_drop(s*r*o), 1)

        return score

class TransE(nn.Module):
	"""docstring for TransE"""
	def __init__(self, norm):
		super(TransE, self).__init__()
		self.norm = norm

	def forward(self, s,r,o):
		#s = F.normalize(s, 2, 1)
		#r = F.normalize(r, 2, 1)
		#o = F.normalize(o, 2, 1)
		score = torch.norm(s+r-o, self.norm, 1)

		return score
		

class ConvE(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(ConvE, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 8, (3, 3), 1, 0, True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.fc = torch.nn.Linear(1152,embedding_dim)
        #self.register_parameter('b', Parameter(torch.zeros(num_entities)))

        self.inp_drop = torch.nn.Dropout(0.3)
        self.hidden_drop = torch.nn.Dropout(0.2)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)

    def forward(self, e1, rel, e2):
        e1_embedded= e1.view(-1, 1, 10, 10)
        rel_embedded = rel.view(-1, 1, 10, 10)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        x = self.bn0(stacked_inputs)
        x= self.inp_drop(x)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        #print(x.size())
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.sum(x*e2, -1)
        #x += self.b.expand_as(x)
        #pred = torch.sigmoid(x)

        return x
