import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class VisualEncoder(nn.Module):
    def __init__(self, img_size, embedding_size):
        super(VisualEncoder, self).__init__()

        self.img_size = img_size
        self.resnet = models.resnet18(weights='DEFAULT')
        self.encoder = nn.Linear(list(self.resnet.modules())[-1].out_features, embedding_size)
    
    def forward(self, x):
        with torch.no_grad():
            x = self.resnet(x)
        return F.relu(self.encoder(x))

class TaskIdEncoder(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(TaskIdEncoder, self).__init__()

        self.clip, _ = clip.load("ViT-B/32")
        self.encoder = nn.Linear(input_size, embedding_size)
    
    def forward(self, x):
        with torch.no_grad():
            x = self.clip.encode_text(x).float()
        return F.relu(self.encoder(x))

class ActionEncoder(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(ActionEncoder, self).__init__()

        self.encoder = nn.Linear(input_size, embedding_size)
    
    def forward(self, x):
        return F.relu(self.encoder(x))

class Backbone(nn.Module):
    def __init__(self, embedding_size=192, intermediate_size=256, img_size=224, task_size=512, action_size=12):
        super(Backbone, self).__init__()

        self.visual_encoder = VisualEncoder(img_size, embedding_size)
        self.task_id_encoder = TaskIdEncoder(task_size, embedding_size)
        self.action_encoder = ActionEncoder(action_size, embedding_size)

        self.discriminator = nn.Sequential(
            nn.Linear(embedding_size * 3, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, 1),
        )
    
    def forward(self, img, task, action):
        visual_embed = self.visual_encoder(img)
        task_id_embed = self.task_id_encoder(task)
        action_embed = self.action_encoder(action)
        x = torch.cat((visual_embed, task_id_embed, action_embed), dim=-1)
        return self.discriminator(x)
