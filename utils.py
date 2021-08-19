def parse_txt(path: str) -> list[tuple[int, list[float]]]:
    with open(path, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]

    return [(int(line.split()[0]), list(map(float,
                                            line.split()[1:])))
            for line in lines]


def xywh_to_xyxy(box: list[float]) -> list[float]:
    xcen, ycen, w, h = box
    xmin = xcen - w / 2
    ymin = ycen - h / 2
    xmax = xcen + w / 2
    ymax = ycen + h / 2

    return [xmin, ymin, xmax, ymax]


def denornmalized_vertex(box: list[float], image_size: tuple[int]) -> list[int]:
    h, w = image_size
    box[0], box[2] = box[0] * w, box[2] * w
    box[1], box[3] = box[1] * h, box[3] * h
    box = list(map(int, box))

    return box
