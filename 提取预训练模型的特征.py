# 提取 ImageNet 预训练模型某层的卷积特征
# # # VGG-16 relu5-3 feature.
# # model = torchvision.models.vgg16(pretrained=True).features[:-1]
# # # VGG-16 pool5 feature.
# # model = torchvision.models.vgg16(pretrained=True).features
# # # VGG-16 fc7 feature.
# # model = torchvision.models.vgg16(pretrained=True)
# # model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-3])
# # # ResNet GAP feature.
# # model = torchvision.models.resnet18(pretrained=True)
# # model = torch.nn.Sequential(collections.OrderedDict(
# #     list(model.named_children())[:-1]))
# #
# # with torch.no_grad():
# #     model.eval()
# #     conv_representation = model(image)

# 提取 ImageNet 预训练模型多层的卷积特征
# class FeatureExtractor(torch.nn.Module):
#     """Helper class to extract several convolution features from the given
#     pre-trained model.
#
#     Attributes:
#         _model, torch.nn.Module.
#         _layers_to_extract, list<str> or set<str>
#
#     Example:
#         >>> model = torchvision.models.resnet152(pretrained=True)
#         >>> model = torch.nn.Sequential(collections.OrderedDict(
#                 list(model.named_children())[:-1]))
#         >>> conv_representation = FeatureExtractor(
#                 pretrained_model=model,
#                 layers_to_extract={'layer1', 'layer2', 'layer3', 'layer4'})(image)
#     """
#     def __init__(self, pretrained_model, layers_to_extract):
#         torch.nn.Module.__init__(self)
#         self._model = pretrained_model
#         self._model.eval()
#         self._layers_to_extract = set(layers_to_extract)
#
#     def forward(self, x):
#         with torch.no_grad():
#             conv_representation = []
#             for name, layer in self._model.named_children():
#                 x = layer(x)
#                 if name in self._layers_to_extract:
#                     conv_representation.append(x)
#             return conv_representation