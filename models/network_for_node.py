import torch
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv
from torch_geometric.data import Data
import torch.nn as nn
import torch.optim as optim
from utils import init_weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device="cpu"
class EncoderDecoderNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_features,in_edges,output_dim):
        super(EncoderDecoderNet, self).__init__()
        self.edge_mlp = nn.Linear(in_edges, hidden_features)
        self.mlp=nn.Linear(input_dim,hidden_features)
        self.mlp.apply(init_weights)
        self.encoder_conv_1 = GeneralConv(in_channels=hidden_features, out_channels=hidden_features, in_edge_channels=hidden_features,heads=1)
        self.encoder_conv_2 = GeneralConv(in_channels=hidden_features, out_channels=hidden_features, in_edge_channels=hidden_features,heads=1)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.bn2 = nn.BatchNorm1d(hidden_features)
        self.bn3 = nn.BatchNorm1d(hidden_features)
        self.predictor = torch.nn.Linear(hidden_features*2, output_dim)

    def forward(self,input_features, edge_index,node_mask=None, edge_weight=None,cluster_dict=None):

        edge_weight=F.relu(self.edge_mlp(edge_weight))
        x_ini = F.relu(self.bn1(self.mlp(input_features)))
        x = F.relu(self.bn2(self.encoder_conv_1(x_ini, edge_index, edge_attr=edge_weight)))
        x = F.relu(self.bn3(self.encoder_conv_2(x, edge_index, edge_attr=edge_weight)))
        combined_features = torch.cat([x_ini[node_mask],x[node_mask]], dim=1)
        node_predict=self.predictor(combined_features)
        cluster_list=[]
        for inter in cluster_dict:
            cluster_list.append(node_predict[cluster_dict[inter]].mean(0).reshape(1,-1))
        cluster_predict=torch.cat(cluster_list,dim=0)

        return F.log_softmax(cluster_predict, dim=1)

class form_data:
    def __init__(self):
        pass
    def formulation(self,node_features, edge_index,
                     edge_weight,mask_train,mask_validate,mask_test,cluster_dict_out,cluster_decision):


        node_features = torch.tensor(node_features, dtype=torch.float)
        min_node = node_features.min()
        max_node = node_features.max()
        node_features = (node_features - min_node) / (max_node - min_node)
        cluster_decision=torch.tensor(cluster_decision, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float).reshape(-1,1)
        min_val = edge_weight.min()
        max_val = edge_weight.max()
        edge_weight = (edge_weight - min_val) / (max_val - min_val)
        data = Data(node_features=node_features.to(device), edge_index=edge_index.to(device),
                        edge_attr=edge_weight.to(device),label=cluster_decision.to(device),mask_train=mask_train.to(device),mask_validate=mask_validate.to(device),mask_test=mask_test.to(device),cluster_dict_out=cluster_dict_out)

        return data


class GNN_train:
    def __init__(self, input_dim, hidden_features,in_edges,output_dim,wandb):
        self.model = EncoderDecoderNet(input_dim, hidden_features,in_edges,output_dim).to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.wandb=wandb

    def train_validate(self,data,iter_num,mask_rate=0.5,batch_size=32,model_path='model_path/best_model.pth',learning_rate=1e-3):

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        best_val_accuracy = -500
        self.save_path=model_path
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, iter_num)
        self.cluster_dict=data.cluster_dict_out
        for epoch in range(iter_num):
            self.model.train()
            loss_mean=0
            mask_train = data.mask_train
            for inter in range(batch_size):
                mask=mask_train.clone()
                mask_for_ones = mask == 1
                random_mask = torch.rand(mask_for_ones.sum()) > mask_rate
                mask[mask_for_ones] = random_mask.float().to(device)
                mask = mask.bool()
                idx_list=[]
                mask_long_all=torch.zeros(len(data.node_features))
                sample_cluster_dict={}
                iner_count = 0
                for idex in range(len(self.cluster_dict.keys())):
                    if mask[idex]==True:
                        idx_list+=self.cluster_dict[idex]
                        sample_cluster_dict[idex]=list(range(iner_count,iner_count+len(self.cluster_dict[idex])))
                        iner_count+=len(self.cluster_dict[idex])
                mask_long_all[idx_list]=1
                mask_long_all=mask_long_all.bool()
                self.optimizer.zero_grad()
                predicted_decisions= self.model(input_features=data.node_features, edge_index=data.edge_index,node_mask=mask_long_all,
                                            edge_weight=data.edge_attr,cluster_dict=sample_cluster_dict)
                loss = self.criterion(predicted_decisions.to(device), data.label[mask].reshape(-1).to(device))
                loss_mean+=loss
            loss_mean=loss_mean/batch_size
            loss_mean.backward()
            self.optimizer.step()
            self.scheduler.step()
            _, pred = predicted_decisions.max(dim=1)
            node_pred_np = pred.cpu().numpy()
            node_label_np = data.label[mask].cpu().numpy()
            train_accuracy = accuracy_score(node_label_np, node_pred_np)

            print("training_loss:",loss_mean, "train_accuracy:", train_accuracy)

            print("num:",epoch)

            self.model.eval()
            mask_validate = torch.tensor(data.mask_validate, dtype=torch.bool)
            idx_list = []
            mask_long_all = torch.zeros(len(data.node_features))
            sample_cluster_dict = {}
            iner_count = 0
            for idex in range(len(self.cluster_dict.keys())):
                if mask_validate[idex] == True:
                    idx_list += self.cluster_dict[idex]
                    sample_cluster_dict[idex] = list(range(iner_count, iner_count + len(self.cluster_dict[idex])))
                    iner_count += len(self.cluster_dict[idex])
            mask_long_all[idx_list] = 1
            mask_long_all = mask_long_all.bool().to(device)

            with torch.no_grad():
                predicted_decisions_validate = self.model(input_features=data.node_features, edge_index=data.edge_index,node_mask=mask_long_all,
                                            edge_weight=data.edge_attr, cluster_dict=sample_cluster_dict)

                loss_validate = self.criterion(predicted_decisions_validate.to(device), data.label[mask_validate].reshape(-1).to(device))
                _, pred = predicted_decisions_validate.max(dim=1)
                node_pred_np = pred.cpu().numpy()
                node_label_np = data.label[mask_validate].cpu().numpy()
                validate_f1 = f1_score(node_label_np, node_pred_np, average='macro')
                validate_accuracy = accuracy_score(node_label_np, node_pred_np)
                print("validate_loss:", loss_validate,"validate_accuracy:",validate_accuracy,
                      "validate_f1:",validate_f1)

                if validate_accuracy > best_val_accuracy  or validate_accuracy == best_val_accuracy :
                    best_val_accuracy = validate_accuracy
                    torch.save(self.model.state_dict(), self.save_path)
                    print("save network")
                self.wandb.log({"train_accuracy": train_accuracy, "validate_accuracy": validate_accuracy,"epoch": epoch})

    def model_test(self, data,load_with_model=False):
        if load_with_model:
            self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        mask = torch.tensor(data.mask_test, dtype=torch.bool).to(device)
        idx_list = []
        mask_long_all = torch.zeros(len(data.node_features)).to(device)
        sample_cluster_dict = {}
        iner_count = 0
        for idex in range(len(self.cluster_dict.keys())):
            if mask[idex] == True:
                idx_list += self.cluster_dict[idex]
                sample_cluster_dict[idex] = list(range(iner_count, iner_count + len(self.cluster_dict[idex])))
                iner_count += len(self.cluster_dict[idex])
        mask_long_all[idx_list] = 1
        mask_long_all = mask_long_all.bool()
        with torch.no_grad():
            predicted_decisions = self.model(input_features=data.node_features, edge_index=data.edge_index,node_mask=mask_long_all,
                                            edge_weight=data.edge_attr,cluster_dict=sample_cluster_dict)
        label = data.label[mask].reshape(-1)
        loss_test = self.criterion(predicted_decisions.to(device), label.reshape(-1).to(device))
        _, pred = predicted_decisions.max(dim=1)
        node_pred_np = pred.cpu().numpy()
        node_label_np = label.cpu().numpy()
        accuracy = accuracy_score(node_label_np, node_pred_np)
        precision = precision_score(node_label_np, node_pred_np, average='macro')
        recall = recall_score(node_label_np, node_pred_np, average='macro')
        f1 = f1_score(node_label_np, node_pred_np, average='macro')
        print("test_loss:", loss_test,"node_predict:",pred,"node_label:",label)
        print(f"accuracy: {accuracy}")
        print(f"precision_macro: {precision}")
        print(f"recall_macro: {recall}")
        print(f"f1_macro: {f1}")


        return  accuracy,f1
