import { AStarSolver } from "../../../helpers/algorithms/AStar";
import { DjikstraSolver } from "../../../helpers/algorithms/Djikstra";
import { GreedySolver } from "../../../helpers/algorithms/Greedy";
import { Node } from "../../../helpers/Node";

export type Solvers = {
    astar: AStarSolver;
    djikstra: DjikstraSolver;
    greedy: GreedySolver
};

export type Algorithm = keyof Solvers;
