import torch
import torch.nn as nn

class single_net(nn.Module):
    def __init__(self, input_size, n_tasks=1, n_primary = 3, n_aux=0):
        super(single_net, self).__init__()

        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048

        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size, num_layers=self.numLayers, batch_first=True)
        self.linear3 = nn.Linear(self.memory_size, self.penul)
        
        self.primary_layers = nn.ModuleList()
        for i in range(n_primary):
            self.primary_layers.append(nn.Linear(self.penul, 1))
        
        self.aux_layers = nn.ModuleList()
        for i in range(n_aux):
            self.aux_layers.append(nn.Linear(self.penul, 1))

    def forward(self, x, task_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        out = self.linear1(x).relu()
        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim, self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)
        out_s = self.linear3(out).relu()
        
        out_primary = torch.zeros(batch_dim, time_dim, len(self.primary_layers), device=x.device)
        for i in range(len(self.primary_layers)):
            out_primary[:,:,i] = (self.primary_layers[i](out_s))[:,:,0]
        
        out_aux = torch.zeros(len(self.aux_layers), batch_dim, time_dim, 1, device=x.device)
        for i in range(len(self.aux_layers)):
            out_aux[i,:,:,:] = (self.aux_layers[i](out_s).sigmoid())

        return torch.unsqueeze(out_primary[:,:,1],2), out_aux, h_next


class multihead_net(nn.Module):
    def __init__(self, input_size, n_tasks=1, n_primary = 3, n_aux=0):
        super(multihead_net, self).__init__()

        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size, num_layers=self.numLayers, batch_first=True)
        self.linear3 = nn.Linear(self.memory_size, self.penul)
        
        self.primary_layers = nn.ModuleList()
        for i in range(n_primary):
            self.primary_layers.append(nn.ModuleList([nn.Linear(self.penul, 1) for _ in range(n_tasks)]))
        
        self.aux_layers = nn.ModuleList()
        for i in range(n_aux):
            self.aux_layers.append(nn.ModuleList([nn.Linear(self.penul, 1) for _ in range(n_tasks)]))

    def forward(self, x, task_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        out = self.linear1(x).relu()
        out = self.linear2(out).relu()
        out, h_next = self.rnn(out, h)
        out_s = self.linear3(out).relu()

        labels = task_label[:,0] #the whole sequence has the same label
        batch_size = task_label.shape[0]
        
        out_primary = torch.zeros(batch_dim, time_dim, len(self.primary_layers), device=x.device)
        for j in range(len(self.primary_layers)):
            out_primary[:,:,j] = torch.stack([self.primary_layers[j][labels[i]](out_s[i]) for i in range(batch_size)])[:,:,0]
        
        out_aux = torch.zeros(len(self.aux_layers), batch_dim, time_dim, 1, device=x.device)
        for j in range(len(self.aux_layers)):
            out_aux[i,:,:,:] = (torch.stack([self.aux_layers[j][labels[i]](out_s[i]) for i in range(batch_size)]).sigmoid())
        
        return out_primary, out_aux, h_next

 
class multihead_net_finetune(nn.Module):  
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(multihead_net_finetune, self).__init__()

        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size, num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        
        self.primary_layers = nn.ModuleList()
        for i in range(n_primary):
            self.primary_layers.append(nn.ModuleList([nn.Linear(self.penul, 1) for _ in range(n_tasks)]))
        
        self.aux_layers = nn.ModuleList()
        for i in range(n_aux):
            self.aux_layers.append(nn.ModuleList([nn.Linear(self.penul, 1) for _ in range(n_tasks)]))

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        out = self.linear1(x).relu()
        out = self.linear2(out).relu()
        out, h_next = self.rnn(out, h)
        out_s = self.linear3(out).relu()

        labels = task_label[:,0] #the whole sequence has the same label
        batch_size = task_label.shape[0]
        
        out_primary = torch.zeros(batch_dim, time_dim, len(self.primary_layers), device=x.device)
        for j in range(len(self.primary_layers)):
            out_primary[:,:,j] = torch.stack([self.primary_layers[j][labels[i]](out_s[i]) for i in range(batch_size)])[:,:,0]
        
        out_aux = torch.zeros(len(self.aux_layers), batch_dim, time_dim, 1, device=x.device)
        for j in range(len(self.aux_layers)):
            out_aux[i,:,:,:] = (torch.stack([self.aux_layers[j][labels[i]](out_s[i]) for i in range(batch_size)]).sigmoid())
        
        return out_primary, out_aux, h_next


class concat_embedding_net(nn.Module):  
    def __init__(self, input_size, n_tasks=1, n_primary = 3, n_aux=0):
        super(concat_embedding_net, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        
        self.embedding = nn.Embedding(n_tasks, input_size)
        self.linear1 = nn.Linear(input_size*2, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size, num_layers=self.numLayers, batch_first=True)
        self.linear3 = nn.Linear(self.memory_size, self.penul)
        
        self.primary = nn.Linear(self.penul, n_primary)
        self.var = nn.Linear(self.penul, n_primary).requires_grad_(False) #not currently used but added so can be tuned in the future
        self.auxilary = nn.Linear(self.penul, n_aux)

    def forward(self, x, task_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(task_label)
        
        #add x, embedding_out
        x = torch.cat((x,embedding_out),axis=-1)
        
        out = self.linear1(x).relu()
        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim, self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)
        out_s = self.linear3(out).relu()
        
        out_primary = self.primary(out_s)
        out_aux = self.auxilary(out_s).sigmoid()

        return out_primary, out_aux, h_next


class concat_embedding_net_finetune(nn.Module):
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(concat_embedding_net_finetune, self).__init__()

        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        
        self.embedding = nn.Embedding(n_tasks, input_size)
        self.linear1 = nn.Linear(input_size*2, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size, num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)
        
        self.primary = nn.Linear(self.penul, n_primary).requires_grad_(False)
        self.var = nn.Linear(self.penul, n_primary).requires_grad_(False) #not currently used but added so can be tuned in the future
        self.auxilary = nn.Linear(self.penul, n_aux).requires_grad_(False)


    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(task_label)
        
        #add x, embedding_out
        x = torch.cat((x,embedding_out),axis=-1)
        
        out = self.linear1(x).relu()
        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim, self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)
        out_s = self.linear3(out).relu() 
        
        out_primary = self.primary(out_s)
        out_aux = self.auxilary(out_s).sigmoid()

        return out_primary, out_aux, h_next
