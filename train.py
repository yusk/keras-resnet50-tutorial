import sys

from model import ResNetModel


def main():
    args = sys.argv

    if len(args) < 3:
        print("python main.py [data_dir] [n_epoch] [batch_size]")
        exit(-1)
    else:
        data_dir = args[1]
        n_epoch = int(args[2])
        batch_size = int(args[3])
    model = ResNetModel(data_dir=data_dir)
    model.train(n_epoch, batch_size)

    model.test(batch_size)


if __name__ == "__main__":
    main()
