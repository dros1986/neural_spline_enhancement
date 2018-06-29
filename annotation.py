#!/usr/bin/env python3
import argparse

import torch
import torch.nn.functional as F
import torchvision


class AnnotationNet(torch.nn.Module):
    def __init__(self, outs):
        super(AnnotationNet, self).__init__()
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)

        self.resnet = torchvision.models.resnet50(True)
        for p in self.resnet.parameters():
            p.requires_grad = False;
        self.resnet.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.resnet.fc = torch.nn.ReLU()  # This should do nothing, it's just to disable the final fc layer
        self.annotators = torch.nn.ModuleList([torch.nn.Linear(2048, n) for n in outs])

    def forward(self, input):
        x = (input - self.mean) / self.std
        x = self.resnet(x)
        logits = [a(x) for a in self.annotators]
        return logits

    def tunable_parameters(self):
        return self.annotators.parameters()


def loss(scores, target):
    losses = [F.cross_entropy(s, target[:, i]) for (i, s) in enumerate(scores)]
    return sum(losses)


class AnnotatedFivek(torch.utils.data.Dataset):
    TAGS = [tags.split("|") for tags in (
        "abstract|animal(s)|man-made object|nature|person(s)|unknown",
        "artificial|mixed|sun or sky",
        "indoors|outdoors|unknown",
        "dawn or dusk|day|night|unknown"
    )]

    def __init__(self, imagedir, annotation_file, train=True):
        self.images = []
        self.annotations = []
        self.train = train
        with open(annotation_file) as csv_file:
            for record in csv.reader(csv_file):
                self.images.append(os.path.join(imagedir, record[0]))
                tags = [self._class_index(k, record[k + 1]) for k in range(len(record) - 1)]
                self.annotations.append(tags)

    def _class_index(self, tagset, classname):
        i = self.TAGS[tagset].index(classname)
        if i < 0:
            raise ValueError("Unknow tag '{}'".format(classname))
        return i

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image = scipy.ndimage.imread(self.images[index], mode="RGB")
        if self.train and random.random() < 0.5:
            image = image[:, ::-1, :]
        image = image.astype("float32") / 255.0
        image = image.transpose([2, 0, 1])
        image = torch.from_numpy(image)
        tags = torch.tensor(self.annotations[index]).long()
        return image, tags

def _repeat(self):
    while True:
        for x in self:
            yield x


def _parse_args():
    parser = argparse.ArgumentParser(description="Train or test the annotation NN.")
    a = parser.add_argument
    a("--batch_size", type=int, default=10)
    a("--iterations", type=int, default=1000)
    a("--learning_rate", type=float, default=0.001)
    a("--weight_decay", type=float, default=0.0)
    a("--model_file", default="annotation.pth")
    a("--num_workers", type=int, default=torch.get_num_threads())
    default_device = ("cuda" if torch.cuda.is_available() else "cpu")
    a("--device", default=default_device)
    a("--train_annotation", default="train_annotations.csv")
    a("--test_annotation", default="test_random250_annotations.csv")
    a("--image_dir", default="/mnt/data/dataset/fivek/raw")
    a("--eval_only", action="store_true")
    return parser.parse_args()


def _eval(net, args):
    dataset = AnnotatedFivek(args.image_dir, args.test_annotation, train=False)
    loader = torch.utils.data.DataLoader(dataset, args.batch_size, pin_memory=True, num_workers=args.num_workers)
    total = torch.zeros(4)
    count = 0
    net.eval()
    for images, target in loader:
        images = images.to(args.device)
        target = target.to(args.device)
        scores = net(images)
        predictions = torch.stack([s.argmax(1) for s in scores], 1)
        ok = torch.sum(torch.eq(predictions, target), 0)
        total = total + ok.float().cpu()
        count = count + target.size(0)
    net.train()
    accuracy = 100.0 * ((total / count).cpu().numpy())
    print("Test accuracies: {:.1f} {:.1f} {:.1f} {:.1f}".format(*accuracy), "  Average: {:.2f}".format(accuracy.mean()))


def _train():
    args = _parse_args()
    print(args)
    if args.eval_only:
        net = AnnotationNet([6, 3, 3, 4])
        net.load_state_dict(torch.load(args.model_file, map_location="cpu"))
        net.to(args.device)
        _eval(net, args)
        return
    dataset = AnnotatedFivek(args.image_dir, args.train_annotation, train=True)
    loader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    get_data = iter(_repeat(loader))
    net = AnnotationNet([6, 3, 3, 4])
    net.to(args.device)
    optimizer = torch.optim.Adam(net.tunable_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    for step in range(1, args.iterations + 1):
        optimizer.zero_grad()
        images, target = next(get_data)
        images = images.to(args.device)
        target = target.to(args.device)
        logits = net(images)
        l = loss(logits, target)
        l.backward()
        optimizer.step()
        if step % 50 == 0:
            print(step, l.item())
        if step % 200 == 0 or step == args.iterations:
            with open(args.model_file, "wb") as f:
                torch.save(net.state_dict(), f)
            _eval(net, args)


def _test():
    net = AnnotationNet([6, 3, 3, 4])
    images = torch.zeros(2, 3, 128, 128)
    logits = net(images)
    target = torch.zeros(2, 4, dtype=torch.int64)
    print(loss(logits, target))


if __name__ == '__main__':
    import csv
    import os.path
    import random
    import scipy.ndimage
    # _test()
    _train()
