import classification_KOR as KOR
import classification_EN as EN
import sys


def main():
    if len(sys.argv) < 2:
        print("Insufficient arguments")
        sys.exit()

    mode = sys.argv[1]
    path = sys.argv[2]

    # KOR case
    if mode == '0':
        KOR.classification(path)

    # EN case
    elif mode == '1':
        EN.classification(path)


if __name__ == "__main__":
    main()