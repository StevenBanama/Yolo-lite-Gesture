import re
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from demo import run_batch
from utils.utils import build_params 

class GestureEval(COCO):

    def __init__(self, ano="./data/test.ano", params=None):
        # https://github.com/cocodataset/cocoapi/blob/e140a084d678eacd18e85a9d8cfa45d1d5911db9/PythonAPI/pycocotools/coco.py#L179
        super().__init__(None, )
        ano = ano if ano is not None else params.eval_ano
        self.dataset = {"annotations": [], "categories": [{"name": name, "id": nid} for name, nid in params.categories.items()], "images": []}
        self.transform(ano)
        self.createIndex()


    def transform(self, ano):
        '''
           hack function
           ano: type == str, the formation of anotation file meets demands which is "../XX/XX/\d+.jpg x1,y1,x2,y2,cid x1,y1x2,y2,cid ...."
                type = np.array NX7 [imageid, x1, y1, x2,y2, prob, cid]
        '''
        if isinstance(ano, str):
            result = []
            with open(ano) as fd:
                lines = fd.readlines()
                for line in lines:
                    sp = line.strip().split(" ")
                    fpath, locs = sp[0], sp[1:]
                    mtc = re.search("\d+", fpath)
                    image_id = int(mtc.group(0)) if mtc else None 
                    if image_id == None:
                        raise Exception("image id must be a integer")

                    for loc in locs:
                        elm = list(map(int ,loc.split(",")))
                        if len(elm) == 5:
                            x1, y1, x2, y2, cid, score = elm + [1]
                        elif elm == 6:
                            x1, y1, x2, y2, score, cid = elm

                        result.append([image_id, x1, y1, x2, y2, score, cid])
                result =  np.array(result)
        elif type(ano) == np.ndarray:
            result = ano
        else:
            raise Exception("fata ano")
        for elm in result:
            image_id, x1, y1, x2, y2, score, cid = elm
            self.dataset["images"].append({"id": image_id})
            self.dataset["annotations"].append({"id": image_id, "image_id": image_id, "category_id": cid, "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "score": 1, "iscrowd": False}) 

def evaluating(cocoGt, cocoDt):
    handler = COCOeval(cocoGt, cocoDt, "bbox") 
    print(handler.params.imgIds)
    handler.evaluate()
    handler.accumulate()
    handler.summarize()


if __name__ == "__main__":
    params = build_params()
    result = run_batch(params)
    gt = GestureEval(None, params=params)
    dt = GestureEval(result, params)
    evaluating(gt, dt)
