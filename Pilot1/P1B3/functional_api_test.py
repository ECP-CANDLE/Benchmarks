# python p1b3_baseline_keras2.py --feature_subsample 500 -e 5 --train_steps 100 --val_steps 10 --test_steps 10

import p1b3_baseline_keras2 as pbk

def main():
    parser = pbk.get_parser()
    # args is a argparse.Namespace
    args = parser.parse_args()
    args.feature_subsample = 500
    args.epochs = 5
    args.train_steps = 100
    args.val_steps = 10
    args.test_steps = 10
    args.save = "/tmp/save-"

    history = pbk.run(args)
    print(history.history)

if __name__ == '__main__':
    main()
