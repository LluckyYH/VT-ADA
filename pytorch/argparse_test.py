import argparse

parser = argparse.ArgumentParser(description='calculting the area of rectangle!')
parser.add_argument('--length', type=int, help='The length of rectangle ! (type=int)')
parser.add_argument('--width', type=int, help='The width of rectangle! (type=int)')

args = parser.parse_args()

if __name__ == '__main__':
    args.length = 20
    args.width = 10
    result = args.length * args.width
    print("the rectangle's area is " + str(result))
