import { AStarSolver } from "./AStar";
import { DjikstraSolver } from "./Djikstra";
import type { Solvers } from "../../@types/helpers/algorithms";

export const solvers: Solvers = {
    astar: new AStarSolver(),
    djikstra: new DjikstraSolver(),
};
