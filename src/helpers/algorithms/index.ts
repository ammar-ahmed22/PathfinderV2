import { AStarSolver } from "./AStar";
import { DjikstraSolver } from "./Djikstra";

export const solvers = {
  astar: new AStarSolver(),
  djikstra: new DjikstraSolver()
}