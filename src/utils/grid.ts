import Vec2 from "../helpers/Vec2";
import Node from "../helpers/Node";
import type { AStar, Djikstra, Greedy } from "../@types/helpers/Node";
import type { StoreContextType } from "../@types/Store";
import { sleep } from "./async";
import { createGradient, hexToRGB, RGB } from "./colors";

import type { CornerType } from "../@types/components/Cell";

export const isCorner = (
    index: Vec2,
    rows: number,
    cols: number
): CornerType | undefined => {
    const corners = {
        topLeft: new Vec2(),
        topRight: new Vec2(cols - 1, 0),
        bottomLeft: new Vec2(0, rows - 1),
        bottomRight: new Vec2(cols - 1, rows - 1),
    };

    if (index.equals(corners.topLeft)) return "tl";
    if (index.equals(corners.topRight)) return "tr";
    if (index.equals(corners.bottomLeft)) return "bl";
    if (index.equals(corners.bottomRight)) return "br";

    return undefined;
};

type NodeAlgorithmArray = Node<AStar>[] | Node<Djikstra>[] | Node<Greedy>[];
export const animate = async (
    store: StoreContextType,
    path: NodeAlgorithmArray,
    searched: NodeAlgorithmArray,
    delay: number
) => {
    while (!!searched.length) {
        const nodeToAnimate = searched.shift();
        if (
            nodeToAnimate &&
            store.startIdx &&
            store.targetIdx &&
            !nodeToAnimate.index.equals(store.startIdx) &&
            !nodeToAnimate.index.equals(store.targetIdx)
        )
            store.updateNodeTypeByIndex(nodeToAnimate.index, "visited");
        await sleep(delay);
    }

    const startColor = hexToRGB("#D38312") as RGB;
    const endColor = hexToRGB("#A83279") as RGB;

    const gradient = createGradient({ values: path.length, startColor, endColor, output: "hex" })
    console.log({ gradient });
    let i = 0;
    while (!!path.length) {
        const nodeToAnimate = path.pop();
        if (
            nodeToAnimate &&
            store.startIdx &&
            store.targetIdx &&
            !nodeToAnimate.index.equals(store.startIdx) &&
            !nodeToAnimate.index.equals(store.targetIdx)
        )
            store.updateNodeByIndex(nodeToAnimate.index, (prevNode) => {
                prevNode.type = "path";
                prevNode.bg = gradient[i] as string;
                return prevNode;
            });

            i++
        await sleep(delay);
    }
};
