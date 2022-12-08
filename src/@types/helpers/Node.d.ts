export type NodeType =
  | "base"
  | "start"
  | "target"
  | "path"
  | "visited"
  | "obstacle"
  | "current";

export interface getNeighboursOpts {
  allowDiagonals?: boolean;
}

export interface AStar {
  heuristic: number;
  cost: number;
  func: number;
}

export interface Djikstra {
  cost: number;
}

export interface Greedy {
  heuristic: number;
}

export interface Generic {}

export type AlgorithmParams = AStar | Djikstra | Greedy | Generic;
