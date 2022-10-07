import { Solver, SolverParams } from "../solver/Solver";
import { AStar } from "../../@types/helpers/Node";
import Node from "../Node";
import Vec2 from "../Vec2";

export class AStarSolver extends Solver<AStar> {
    public initialize = ({
        nodes,
        start,
        target,
        delay,
    }: SolverParams): void => {
        for (let row = 0; row < nodes.length; row++) {
            const tempRow: Node<AStar>[] = [];
            for (let col = 0; col < nodes[row].length; col++) {
                const node = nodes[row][col];
                const aStarNode = new Node<AStar>(
                    node.index,
                    node.size,
                    node.type,
                    node.obstacle,
                    {
                        heuristic: 0,
                        cost: 0,
                        func: 0,
                    }
                );

                tempRow.push(aStarNode);
            }

            this.nodes.push(tempRow);
        }

        this.start = start;
        this.target = target;
        this.delay = delay;

        const startNode: Node<AStar> = this.nodes[start.y][start.x];
        this.searching.insert(startNode, startNode.params.func);
    };

    private heuristic = (start: Vec2, end: Vec2): number =>
        Vec2.DistanceSquared(start, end);

    public getOptimalPath = (current: Node<AStar>): Node<AStar>[] => {
        const res: Node<AStar>[] = [];

        let temp = current;
        while (temp.prev) {
            res.push(temp.prev);

            temp = temp.prev;
        }

        return res;
    };

    public solve = (): Node<AStar>[] | undefined => {
        if (this.nodes.length === 0 || this.searching.isEmpty()) {
            throw new Error(
                "the initialize() method must be called prior to using solve()"
            );
        }

        let current: Node<AStar>;
        while (!this.searching.isEmpty()) {
            current = this.searching.pop() as Node<AStar>;

            if (current.index.equals(this.target)) {
                console.log("A STAR DONE");
                return this.getOptimalPath(current);
            }

            this.searched.push(current);

            const neighbours = current.getNeighbours(this.nodes);
            for (let i = 0; i < neighbours.length; i++) {
                const n = neighbours[i];

                if (this.searched.includes(n)) {
                    continue;
                }

                const tentativeCost: number = current.params.cost + 1;
                if (
                    this.searching.includes(n) &&
                    tentativeCost > n.params.cost
                ) {
                    continue;
                }

                n.prev = current;
                n.params.cost = tentativeCost;
                n.params.heuristic = this.heuristic(n.index, this.target);

                n.params.func = n.params.cost + n.params.heuristic;

                if (!this.searching.includes(n))
                    this.searching.insert(n, n.params.func);
            }
        }

        return undefined;
    };
}
