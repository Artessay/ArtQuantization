from generation import generate
from evaluation import evaluate
from utils import get_args

def main():
    args = get_args()
    generate(args)
    evaluate(args)

if __name__ == "__main__":
    main()