import sys

from model import ResNetModel


def main():
    args = sys.argv

    if len(args) < 3:
        print("python main.py [data_dir] [model_path] [batch_size]")
        exit(-1)
    else:
        data_dir = args[1]
        model_path = args[2]
        batch_size = int(args[3])
    model = ResNetModel(data_dir=data_dir, model_path=model_path)

    model.test(batch_size)


if __name__ == "__main__":
    main()
