#注意为了能够恢复训练，我们需要同时保存模型和优化器的状态，以及当前的训练轮数。
# start_epoch = 0
# # Load checkpoint.
# if resume:  # resume为参数，第一次训练时设为0，中断再训练时设为1
#     model_path = os.path.join('model', 'best_checkpoint.pth.tar')
#     assert os.path.isfile(model_path)
#     checkpoint = torch.load(model_path)
#     best_acc = checkpoint['best_acc']
#     start_epoch = checkpoint['epoch']
#     model.load_state_dict(checkpoint['model'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     print('Load checkpoint at epoch {}.'.format(start_epoch))
#     print('Best accuracy so far {}.'.format(best_acc))
#
# # Train the model
# for epoch in range(start_epoch, num_epochs):
#     ...
#
#     # Test the model
#     ...
#
#     # save checkpoint
#     is_best = current_acc > best_acc
#     best_acc = max(current_acc, best_acc)
#     checkpoint = {
#         'best_acc': best_acc,
#         'epoch': epoch + 1,
#         'model': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#     }
#     model_path = os.path.join('model', 'checkpoint.pth.tar')
#     best_model_path = os.path.join('model', 'best_checkpoint.pth.tar')
#     torch.save(checkpoint, model_path)
#     if is_best:
#         shutil.copy(model_path, best_model_path)