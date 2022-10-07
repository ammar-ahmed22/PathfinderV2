import { Solver, SolverParams } from "../solver/Solver";
import { Greedy } from "../../@types/helpers/Node";
import Node from "../Node";
import Vec2 from "../Vec2";

export class GreedySolver extends Solver<Greedy>{
  public initialize({ nodes, start, target, delay }: SolverParams): void {
    // implement
  }

  public getOptimalPath(current: Node<Greedy>): Node<Greedy>[] {
    const res : Node<Greedy>[] = [];

    // implement

    return res
  }

  public solve(): Node<Greedy>[] | undefined {
    // implement
    return undefined
  }
}