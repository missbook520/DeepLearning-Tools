##分类模型训练代码
# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# # Train the model
# total_step = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # Backward and optimizer
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (i + 1) % 100 == 0:
#             print('Epoch: [{}/{}], Step: [{}/{}], Loss: {}'
#                   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# # Test the model
# # model.eval()  # eval mode(batch norm uses moving mean/variance
# # # instead of mini-batch mean/variance)
# # with torch.no_grad():
# #     correct = 0
# #     total = 0
# #     for images, labels in test_loader:
# #         images = images.to(device)
# #         labels = labels.to(device)
# #         outputs = model(images)
# #         _, predicted = torch.max(outputs.data, 1)
# #         total += labels.size(0)
# #         correct += (predicted == labels).sum().item()
# #
# #     print('Test accuracy of the model on the 10000 test images: {} %'
# #           .format(100 * correct / total))