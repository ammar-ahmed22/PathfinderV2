import { AStarSolver } from "../../../helpers/algorithms/AStar";
import { DjikstraSolver } from "../../../helpers/algorithms/Djikstra";
import { Node } from "../../../helpers/Node";

export type Solvers = {
  astar: AStarSolver,
  djikstra: DjikstraSolver
}

export type Algorithm = keyof Solvers;



