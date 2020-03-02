import numpy as np
import cv2
import os


class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = filename

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        max_try = 20000
        while True and max_try > 0:

            distances = 1 - self.iou(boxes, clusters)
            max_try -= 1

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            print(max_try, boxes.shape)
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.strip().split()
            boxes = list(map(lambda x: [int(b) for b in x.split(",")[:4]], infos[1:]))
            dataSet += boxes
        locs = np.array(dataSet)
        print(locs.shape)
        result = np.concatenate([locs[:, 2:3] - locs[:, 0:1], locs[:, 3:4] - locs[:, 1:2]], axis=-1)
        f.close()
        return result



    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        anchor_scale = list(map(str, (np.array(result) / np.array([16, 16, 16, 16, 16, 16, 32, 32, 32,32,32,32]).reshape((-1, 2))/ 2).reshape((-1,)).tolist()))
        print(",".join(anchor_scale))
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))
        print(",".join([str(x) for x in result.reshape((-1,))]))

    def txt2boxes_remove_size(self, size=608., workers=10):
        from multiprocessing import Manager, Lock
        from multiprocessing.pool import Pool

        manager = Manager()
        queue = manager.list()
        result = manager.list()
        with open(self.filename, 'r') as f:
            for line in f.readlines():
                infos = line.strip().split()
                image_path = infos[0]
                boxes = list(map(lambda x: [float(b) for b in x.split(",")[:5]], infos[1:]))
                queue.append((image_path, boxes))
        print(len(queue))

        #while(queue):
        #    proc_lines(queue, result, size)
        pool = Pool(workers)
        for x in range(workers):
            r = pool.apply_async(proc_lines, args=(queue, result, size))
            #print(r.get())
        pool.close()
        pool.join()
        
        locs = np.concatenate(list(result), axis=0)
        result = np.concatenate([locs[:, 2:3] - locs[:, 0:1], locs[:, 3:4] - locs[:, 1:2], locs[:, 4:5]], axis=-1)
        cls_ids = set(locs[:, 4:5].reshape((-1,)).tolist())

        bboxes = []
        for cid in cls_ids:
            cls_candi = result[locs[:, 4] == cid]
            np.random.shuffle(cls_candi)
            print(cls_candi.shape)
            bboxes.append(np.array(cls_candi[:150]))

        return np.concatenate(bboxes, axis=0)[:, :2]


    def txt2clusters2(self, sizes=[160.0, 320.0]):
        org_bboxes = self.txt2boxes()
        all_bboxes = np.concatenate([self.txt2boxes_remove_size(size) for size in sizes], axis=0)
        print("ddd",  all_bboxes.shape)
        result = self.kmeans(all_bboxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        anchor_scale = list(map(str, (np.array(result) / np.array([16] * self.cluster_number + [32] * self.cluster_number).reshape((-1, 2))).reshape((-1,)).tolist()))
        print(",".join(anchor_scale))
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(org_bboxes, result) * 100))
        print(",".join([str(x) for x in result.reshape((-1,))]))

def proc_lines(queue, result, size):
    while(queue):
        path, boxes = queue.pop()
        if not os.path.exists(path):
            print("!!!!!!image not exist: %s"%path)
            return
        if len(boxes) < 1:
            continue
        image = cv2.imread(path)
        max_side = max(image.shape[:2])
        wh_scale = np.array([max_side / size, max_side / size, max_side / size, max_side / size, 1])
        boxes = np.array(boxes) / wh_scale
        result.append(boxes.astype(int).tolist())


if __name__ == "__main__":
    cluster_number = 6
    filename = "./data/train.ano"  # "./data/dataset/cancer_train.ano"
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters2([128, 160, 192, 224, 256, 288, 320, 352, 384, 416])
    #kmeans.txt2clusters()
