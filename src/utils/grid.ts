import Vec2 from "../helpers/Vec2";
import Node from "../helpers/Node";
import type { AStar, Djikstra, Greedy } from "../@types/helpers/Node";
import type { StoreContextType } from "../@types/Store";
import { sleep } from "./async";
import {
    createGradient,
    hexToRGB,
    RGB,
    getChakraCSSVar,
    getCSSVarValue,
} from "./colors";
import { generateMaze } from "./divisionMaze";
import { MazeGenerator } from "./maze";

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

    const startHex = getCSSVarValue(getChakraCSSVar("path.start"));
    const endHex = getCSSVarValue(getChakraCSSVar("path.end"));

    const startColor = hexToRGB(startHex) as RGB;
    const endColor = hexToRGB(endHex) as RGB;

    const gradient = createGradient({
        values: path.length + 1,
        startColor,
        endColor,
        output: "hex",
    });
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

        i++;
        await sleep(delay);
    }
};

export const createRandomObstacles = (
    store: StoreContextType,
    percentCoverage: number
) => {
    for (let row = 0; row < store.nodes.length; row++) {
        for (let col = 0; col < store.nodes[row].length; col++) {
            const { index } = store.nodes[row][col];
            if (
                store.startIdx &&
                store.targetIdx &&
                !(store.startIdx.equals(index) || store.targetIdx.equals(index))
            ) {
                store.updateNodeByIndex(index, (prevNode) => {
                    const obs: boolean = Math.random() < percentCoverage;
                    prevNode.obstacle = obs;
                    prevNode.type = obs ? "obstacle" : "base";
                    return prevNode;
                });
            }
        }
    }
};

export const createMaze = (store: StoreContextType) => {
    if (store.gridDim && store.startIdx && store.targetIdx){
        const m = new MazeGenerator(store.gridDim);
        let gridMaze : boolean[][] = m.generate();
        if (gridMaze[store.startIdx.y][store.startIdx.x]){
            // start is obstacle
        }

        if (gridMaze[store.targetIdx.y][store.targetIdx.x]){
            // target is obstacle
        }
        
        for (let row = 0; row < gridMaze.length; row++){
            for (let col = 0; col < gridMaze[row].length; col++){
                const index = new Vec2(col, row);
                const obs : boolean = gridMaze[row][col];
                if (obs){
                    store.updateNodeByIndex(index, (prevNode) => {
                        prevNode.obstacle = obs;
                        prevNode.type = "obstacle";
                        return prevNode
                    })
                }
            }
        }
    }

} 
