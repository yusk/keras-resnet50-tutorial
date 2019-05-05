import sys

from model import ResNetModel


def main():
    args = sys.argv

    if len(args) < 2:
        print("python main.py [model_path] [img_path]")
        exit(-1)
    else:
        model_path = args[1]
        img_path = args[2]
    model = ResNetModel(model_path=model_path)
    model.test_with_file_path(img_path)


if __name__ == "__main__":
    main()
