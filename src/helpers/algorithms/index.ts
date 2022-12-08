import { AStarSolver } from "./AStar";
import { DjikstraSolver } from "./Djikstra";
import { GreedySolver } from "./Greedy";
import type { Solvers } from "../../@types/helpers/algorithms";

export const solvers: Solvers = {
  astar: new AStarSolver(),
  djikstra: new DjikstraSolver(),
  greedy: new GreedySolver(),
};
