import torch
import torch.nn as nn

class AttensionsEncoder(nn.Module):
    def __init__(self,head_count,embedding_size) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.head_count = head_count
        self.fcq = nn.Linear(embedding_size,embedding_size)
        self.fck = nn.Linear(embedding_size,embedding_size)
        self.fcv = nn.Linear(embedding_size,embedding_size)
        self.attn = nn.MultiheadAttention(embed_dim=embedding_size,num_heads=head_count,batch_first=True)

    def forward(self,X:torch.Tensor):
        device = X.device
        shape = X.shape
        distance = torch.arange(shape[-2]*self.embedding_size,device=device).reshape(shape[-2],self.embedding_size)
        idx = torch.arange(shape[-2],device=device)
        idx=idx*(self.embedding_size-1)
        distance = torch.sin(distance-idx.reshape(shape[-2],1))
        X = X + distance
        attn, _ = self.attn(self.fcq(X),self.fck(X),self.fcv(X),need_weights=False)
        return attn


class YourCodeNet(nn.Module):
    """
        Originally implemented in the paper
             "A large annotated corpus for learning natural language inference"
    """

    def __init__(self,
                 hidden_size,
                 num_classes,
                 embeddings,
                 embedding_size,
                 num_layers,
                 head_num,
                 depth):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.emb = nn.Embedding.from_pretrained(embeddings=embeddings,
                                                freeze=True)
        self.lstm_hyp = nn.LSTM(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.lstm_prem = nn.LSTM(input_size=embedding_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_first=True)

        self.hyp_attn = AttensionsEncoder(head_count=head_num,embedding_size=embedding_size)
        self.prem_attn = AttensionsEncoder(head_count=head_num,embedding_size=embedding_size)

        layers = [2 * hidden_size] + (depth-1)*[200]+ [num_classes]
        fc_arr = [nn.Linear(layers[idx//3],layers[idx//3+1]) if idx%3==0 else nn.Tanh() if idx%3==1 else nn.BatchNorm1d(200) for idx in range(3*depth-2)]
        self.net = nn.Sequential(*fc_arr)
        self.depth =depth

    def forward(self, X):
        """
            Extract the last hidden layer of each LSTM
            Concatenate these two hiddens layers and then run a FC
        """
        
        if len(X.shape) < 2:
          X = X[None]

        half = X.shape[1]//2

        hyp_batch, premise_batch = X[:,:half],X[:,half:]

        hyp_embedding_layer = self.emb(hyp_batch)
        prem_embedding_layer = self.emb(premise_batch)

        hyp_embedding_layer = self.hyp_attn(hyp_embedding_layer)
        prem_embedding_layer = self.prem_attn(prem_embedding_layer)

        hyp_out, (hyp_hn, hyp_cn) = self.lstm_hyp(hyp_embedding_layer)
        prem_out, (prem_hn, prem_cn) = self.lstm_prem(prem_embedding_layer)

        hyp_hn = hyp_hn[0]
        prem_hn = prem_hn[0]


        out = torch.cat((hyp_hn, prem_hn), dim=1)

        out = self.net(out)
        return out


