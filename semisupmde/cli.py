"""Command line interface: semisupmde {download, eval, finetune, predict} (or python -m semisupmde)."""
import argparse
import os

from . import config


def main(argv=None):
    parser = argparse.ArgumentParser(prog='semisupmde', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('download', help='download the models or data')
    p.add_argument('what', choices=['models', 'kitti', 'vkitti'])
    p.add_argument('splits', nargs='*', help='KITTI splits: test, val, train (default: test)')

    p = sub.add_parser('eval', help='evaluate on the KITTI Eigen split (downloads it if needed)')
    p.add_argument('--model', default='finetuned', help="'finetuned', 'pretrained' or a checkpoint path")
    p.add_argument('--split', default='test', choices=['test', 'val'])
    p.add_argument('--caps', type=int, nargs='+', default=[50, 80])

    p = sub.add_parser('finetune', help='vKITTI -> KITTI finetuning (downloads the data if needed)')
    p.add_argument('--init', default='pretrained', help="'pretrained', 'finetuned' or a checkpoint path")
    p.add_argument('--out_dir', default=config.CHECKPOINT_DIR)
    p.add_argument('--seed', type=int, default=0)
    for k, v in config.FINETUNE.items():
        p.add_argument('--' + k, type=type(v) if not isinstance(v, bool) else int, default=v)

    p = sub.add_parser('predict', help='predict depth for images')
    p.add_argument('images', nargs='+')
    p.add_argument('--model', default='finetuned', help="'finetuned', 'pretrained' or a checkpoint path")
    p.add_argument('--out_dir', default='.', help='where to write <name>_depth.png (colourised) and <name>_depth.npy (metres)')

    args = parser.parse_args(argv)
    from . import download

    if args.command == 'download':
        for split in args.splits:
            if split not in ('test', 'val', 'train'):
                parser.error('unknown split %s' % split)
        if args.what == 'models':
            download.download_models()
        elif args.what == 'kitti':
            download.download_kitti(args.splits or ['test'])
        else:
            download.download_vkitti()

    elif args.command == 'eval':
        from .evaluate import evaluate, print_results
        from .model import load_model
        download.download_kitti([args.split])
        print_results(evaluate(load_model(args.model), args.split, args.caps), args.model)

    elif args.command == 'finetune':
        from .finetune import finetune
        download.download_kitti(['test', 'val', 'train'])
        download.download_vkitti()
        settings = {k: getattr(args, k) for k in config.FINETUNE}
        settings['accumulate_grads'] = bool(settings['accumulate_grads'])
        finetune(args.init, args.out_dir, args.seed, **settings)

    elif args.command == 'predict':
        import matplotlib.cm
        import numpy as np
        from PIL import Image
        from .model import load_model, predict_depth
        model = load_model(args.model)
        os.makedirs(args.out_dir, exist_ok=True)
        for path in args.images:
            depth = predict_depth(model, path)
            out = os.path.join(args.out_dir, os.path.splitext(os.path.basename(path))[0] + '_depth')
            np.save(out + '.npy', depth)
            # inverse depth, scaled to its 95th percentile (as in monodepth2)
            inv = 1 / np.maximum(depth, 1e-3)
            inv = np.clip((inv - inv.min()) / (np.percentile(inv, 95) - inv.min() + 1e-8), 0, 1)
            Image.fromarray((matplotlib.cm.magma(inv)[..., :3] * 255).astype(np.uint8)).save(out + '.png')
            print('%s: depth %.1f-%.1f m -> %s.png, %s.npy' % (path, depth.min(), depth.max(), out, out))


if __name__ == '__main__':
    main()
