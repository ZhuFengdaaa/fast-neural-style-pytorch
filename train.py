from fast_neural_style import network
from options import Options
import torchvision
import torch.utils.data.dataset as dset
import torch
import time
import folder

opt = Options().parse()
print(opt)

datasets = []
for path in opt.data_roots:
    dataset = folder.ImageFolder(path)
    datasets.append(dataset)
total_dataset = dset.ConcatDataset(datasets)
data_loader = torch.utils.data.DataLoader(total_dataset,
                                          batch_size=opt.batch_size,
                                          shuffle=(opt.mode == 'train'),
                                          num_workers=opt.num_threads)
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = network.SimpleModel(opt)
print(model)

total_steps = 0
start_time = time.time()
for epoch in range(opt.num_epochs):
    epoch_iter = 0
    total_steps = 0

    for i, data in enumerate(data_loader):
        epoch_iter += 1
        total_steps += 1
        generated_img = model.forward(data)
        content_score, style_score = model.backward()

        if epoch_iter % opt.log_iter == 0:
            print("epoch_iter {}:".format(epoch_iter))
            print('Content Loss: {:4f} Style Loss : {:4f} '.format(
                content_score.data[0], style_score.data[0]))
            print('Time Taken: %d sec' %
                  (time.time() - start_time))
            print()
            start_time = time.time()
        model.generator_optimizer.step()

        # if total_steps % opt.display_freq == 0:
        #     visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save(model.model.state_dict['generator'], 'generator', total_steps)

    # model.update_learning_rate()

    # TODO: vgg preprocess
    # TODO: weight init
    # TODO: visualize
