# #微调全连接层
# model = torchvision.models.resnet18(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
# model.fc = nn.Linear(512, 100)  # Replace the last fc layer
# optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

# #以较大学习率微调全连接层，较小学习率微调卷积层
# model = torchvision.models.resnet18(pretrained=True)
# finetuned_parameters = list(map(id, model.fc.parameters()))
# conv_parameters = (p for p in model.parameters() if id(p) not in finetuned_parameters)
# parameters = [{'params': conv_parameters, 'lr': 1e-3},
#               {'params': model.fc.parameters()}]
# optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)