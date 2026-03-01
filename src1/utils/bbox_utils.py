def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    return int(x_center), int(y_center)

def get_bbox_width(bbox):
    x1, y1, x2, y2 = bbox
    return int(x2 - x1)