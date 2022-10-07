import { Solver, SolverParams } from "../solver/Solver";
import { Djikstra } from "../../@types/helpers/Node";
import Node from "../Node";

export class DjikstraSolver extends Solver<Djikstra> {
    public initialize = ({
        nodes,
        start,
        target,
        delay,
    }: SolverParams): void => {
        for (let row = 0; row < nodes.length; row++) {
            const tempRow: Node<Djikstra>[] = [];
            for (let col = 0; col < nodes[row].length; col++) {
                const node = nodes[row][col];
                const aStarNode = new Node<Djikstra>(
                    node.index,
                    node.size,
                    node.type,
                    node.obstacle,
                    {
                        cost: Infinity,
                    }
                );

                tempRow.push(aStarNode);
            }

            this.nodes.push(tempRow);
        }

        this.start = start;
        this.target = target;
        this.delay = delay;

        const startNode: Node<Djikstra> = this.nodes[start.y][start.x];
        startNode.params.cost = 0;
        this.searching.insert(startNode, startNode.params.cost);
    };

    public getOptimalPath = (current: Node<Djikstra>): Node<Djikstra>[] => {
        const res: Node<Djikstra>[] = [];

        let temp = current;
        while (temp.prev) {
            res.push(temp.prev);

            temp = temp.prev;
        }

        return res;
    };

    public solve = () => {
        if (this.nodes.length === 0 || this.searching.isEmpty()) {
            throw new Error(
                "the initialize() method must be called prior to using solve()"
            );
        }

        let current: Node<Djikstra>;

        while (!this.searching.isEmpty()) {
            current = this.searching.pop() as Node<Djikstra>;

            if (current.index.equals(this.target)) {
                console.log("DJIKSTRA DONE");
                return this.getOptimalPath(current);
            }

            this.searched.push(current);

            const neighbours = current.getNeighbours(this.nodes);

            for (let i = 0; i < neighbours.length; i++) {
                const n = neighbours[i];

                const tenativeCost = current.params.cost + 1;
                if (
                    tenativeCost < n.params.cost &&
                    current.params.cost !== Infinity
                ) {
                    n.params.cost = tenativeCost;
                    n.prev = current;

                    if (!this.searching.includes(n))
                        this.searching.insert(n, n.params.cost);
                }
            }
        }

        return undefined;
    };
}
