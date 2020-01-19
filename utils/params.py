import argparse
import numpy as np

'''
    category file loader:
    formattor:
        class_name1
        class_name2
           .
           .
           .
'''
class LoadCates(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        default = kwargs.get("default")
        kwargs["default"] = self.load_categories(default)

        super(LoadCates, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        cates = self.load_categories(values)
        setattr(namespace, "categories", cates)


    def load_categories(self, cate_txt):
        cates = {}
        with open(cate_txt) as fd:
            lines = fd.readlines()
            for idx, l in enumerate(lines):
                name = l.strip()
                cates[name] = idx
        return cates



'''
    default anchor 2 * 3 * 2, we only keep large and mid 
'''
class LoadAnchors(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        default = kwargs.get("default")
        kwargs["default"] = self.load_anchors(default)
        super(LoadAnchors, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        anchors = self.load_anchors(values)
        setattr(namespace, "anchors", anchors)

    def load_anchors(self, anchor_txt):
        with open(anchor_txt) as fd:
            line = fd.read()
            anchors = np.array(list(map(float, line.split(","))))
            return np.reshape(anchors, (-1, 3, 2)) 

def build_args():
    parser = argparse.ArgumentParser()

    # ------train prams-------
    parser.add_argument('-lr', type=float, default=0.001, help='learn rate')
    parser.add_argument("--strides", nargs='*', default=[16, 32])
    parser.add_argument("--train_input_sizes", nargs='*', default=[128, 160, 192, 224, 256, 288, 320, 352, 384, 416])
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--log_dir", default="./log")
    parser.add_argument("--save_path", default="./models/cp-{epoch:02d}-{val_loss:02f}")
    parser.add_argument("--restore", default=False, action="store_true")
    parser.add_argument("--train", default=True, action="store_true")
    parser.add_argument("--pretrain_model", default="./pretrained/cp-89-4.569014")
    parser.add_argument("--iou_thres", default=0.5)
    parser.add_argument("--anchors_path", default="./data/anchors/anchors.txt", dest="anchors", action=LoadAnchors)
    parser.add_argument("--cate_path", default="./data/categories/cates.txt", dest="categories", action=LoadCates)
    parser.add_argument("--train_ano", default="./data/train.ano")
    parser.add_argument("--test_ano", default="./data/test.ano")
    parser.add_argument("--eval_ano", default="./data/test.ano")
    parser.add_argument("--epoch", default=200)
    parser.add_argument("--batch_size", default=8, type=int)

    # ------- test / evaluating params -------
    parser.add_argument("--mode", choices=["batch", "test", "video", "freeze"], default="video")
    parser.add_argument("--test_input", default=224, type=int)

    args = parser.parse_args()
    setattr(args, "class_num", len(args.categories))
    print(args)
    return args
