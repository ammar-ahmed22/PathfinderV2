import Vec2 from "./Vec2";
import { NodeType, AlgorithmParams } from "../@types/helpers/Node";

export default class Node<A extends AlgorithmParams> {
    public prev: Node<A> | undefined;
    constructor(
        public index: Vec2,
        public size: number,
        public type: NodeType,
        public obstacle: boolean,
        public params: A
    ) {}

    public getNeighbours = (nodes: Node<A>[][]): Node<A>[] => {
        const res: Node<A>[] = [];
        const col = this.index.x;
        const row = this.index.y;

        const rows = nodes.length;
        const cols = nodes[0].length;

        if (row !== 0) {
            // above
            res.push(nodes[row - 1][col]);
        }

        if (row < rows - 1) {
            // below
            res.push(nodes[row + 1][col]);
        }

        if (col !== 0) {
            // left
            res.push(nodes[row][col - 1]);
        }

        if (col < cols - 1) {
            // right
            res.push(nodes[row][col + 1]);
        }

        return res.filter((node) => !node.obstacle);
    };
}
