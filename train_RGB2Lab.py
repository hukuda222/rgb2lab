import cv2
import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import dataset, Chain, training, optimizers, \
    iterators, reporter, cuda, serializers, Reporter, report, report_scope
import argparse
from model import Net, Loss_Link
from dataset import DataSet
import os

if cuda.available:
    xp = cuda.cupy
else:
    xp = np


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--width',  type=int, default=28,
                        help='Width of ixput image')
    parser.add_argument('--height', type=int, default=28,
                        help='height of ixput image')
    parser.add_argument('--iteration', '-i', type=int, default=10000,
                        help='Number of examples in iteration')
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--model', '-m', default='',
                        help='Load model')
    parser.add_argument('--test', '-t', action='store_true',
                        help='evaluation only')
    parser.add_argument('--image', action='store_true',
                        help='put image for test')

    args = parser.parse_args()

    train_dataset = DataSet(
        "RGB2Lab",   args.iteration * args.batchsize, args.width, args.height)

    test_dataset = DataSet("RGB2Lab", 100, args.width, args.height)

    model = Loss_Link(Net(args.width * args.height * 3))

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    if args.model:
        serializers.load_npz(args.model, model)

    train_iter = iterators.SerialIterator(
        train_dataset, batch_size=args.batchsize)
    test_iter = iterators.SerialIterator(
        test_dataset,  batch_size=args.batchsize, repeat=False)

    if args.test:
        eva = training.extensions.Evaluator(
            test_iter, model, device=args.gpu)()
        for key in eva:
            print(key + ":" + str(eva[key]))

    elif args.image:
        if not os.path.exists(args.out):
            os.mkdir(args.out)
        img = np.array(np.random.rand(
            args.height, args.width, 3) * 255, np.uint8)
        cv2.imwrite(args.out + '/input_image.png', img)
        t_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        cv2.imwrite(args.out + '/ideal_image.png', t_img)

        pred = model.model(xp.array([[np.ndarray.flatten(img / 255)]])
                           .astype('float32')).data[0]
        cv2.imwrite(args.out + '/output_image.png',
                    np.array(pred * 255).reshape(args.width, args.height, 3).
                    astype("uint8"))

    else:
        optimizer = optimizers.Adam()
        optimizer.setup(model)

        updater = training.StandardUpdater(train_iter, optimizer)
        trainer = training.Trainer(
            updater, (args.iteration, 'iteration'), out=args.out)

        trainer.extend(training.extensions.Evaluator(
            test_iter, model, device=args.gpu),
            trigger=(int(args.iteration / 20), 'iteration'),)

        trainer.extend(training.extensions.LogReport(
            trigger=(int(args.iteration / 20), 'iteration')))
        trainer.extend(training.extensions.PrintReport(
            entries=['iteration', 'main/mean_loss',
                     'main/worst_loss', 'elapsed_time']),
            trigger=(int(args.iteration / 20), 'iteration'))
        trainer.extend(training.extensions.snapshot(), trigger=(
            int(args.iteration / 20), 'iteration'))
        if args.resume:
            serializers.load_npz(args.resume, trainer)
        trainer.run()

        serializers.save_npz('model.npz', model)


if __name__ == "__main__":
    main()
