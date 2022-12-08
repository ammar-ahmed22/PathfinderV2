import { Solver, SolverParams } from "../solver/Solver";
import { Greedy } from "../../@types/helpers/Node";
import Node from "../Node";
import Vec2 from "../Vec2";
import MinPriorityQueue from "../queue/MinPriorityQueue";

export class GreedySolver extends Solver<Greedy> {
  private heuristic = (start: Vec2, end: Vec2): number =>
    Vec2.Distance(start, end);

  public initialize = ({ nodes, start, target }: SolverParams): void => {
    this.nodes = [];
    for (let row = 0; row < nodes.length; row++) {
      const tempRow: Node<Greedy>[] = [];
      for (let col = 0; col < nodes[row].length; col++) {
        const node = nodes[row][col];
        const greedyNode = new Node<Greedy>(
          node.index,
          node.size,
          node.type,
          node.obstacle,
          { heuristic: Infinity }
        );

        tempRow.push(greedyNode);
      }

      this.nodes.push(tempRow);
    }

    this.start = start;
    this.target = target;

    this.searching = new MinPriorityQueue<Node<Greedy>>();

    const startNode: Node<Greedy> = this.nodes[start.y][start.x];
    startNode.params.heuristic = this.heuristic(start, target);
    this.searching.insert(startNode, startNode.params.heuristic);
  };

  public getOptimalPath = (current: Node<Greedy>): Node<Greedy>[] => {
    const path: Node<Greedy>[] = [];

    let temp: Node<Greedy> = current;

    while (temp.prev) {
      path.push(temp.prev);
      temp = temp.prev;
    }

    return path;
  };

  public solve = (): Node<Greedy>[] | undefined => {
    let current: Node<Greedy>;

    while (!this.searching.isEmpty()) {
      current = this.searching.pop() as Node<Greedy>;

      if (current.index.equals(this.target)) {
        console.log("GREEDY DONE");
        return this.getOptimalPath(current);
      }

      this.searched.push(current);

      const neighbours = current.getNeighbours(this.nodes);

      for (let i = 0; i < neighbours.length; i++) {
        const n = neighbours[i];

        if (this.searched.includes(n)) {
          continue;
        }

        const tentativeHeuristic = this.heuristic(n.index, this.target);

        if (this.searching.includes(n)) {
          continue;
        }

        n.params.heuristic = tentativeHeuristic;
        n.prev = current;

        if (!this.searching.includes(n))
          this.searching.insert(n, n.params.heuristic);
      }
    }
    return undefined;
  };
}
