import Node from "../Node";
import Vec2 from "../Vec2";
import MinPriorityQueue from "../queue/MinPriorityQueue";
import { AlgorithmParams, AStar, Generic } from "../../@types/helpers/Node";

export interface SolverParams{
  nodes: Node<Generic>[][],
  start: Vec2,
  target: Vec2,
  delay: number
}

export abstract class Solver<A extends AlgorithmParams>{
  public nodes : Node<A>[][] = [];
  public start: Vec2 = new Vec2();
  public target: Vec2 = new Vec2();
  public delay : number = -1;

  public searching : MinPriorityQueue<Node<A>> = new MinPriorityQueue<Node<A>>();
  public searched : Node<A>[] = [];

  abstract initialize(params: SolverParams) : void;
  abstract solve() : Node<A>[] | undefined;
  abstract getOptimalPath(current: Node<A>) : Node<A>[];
}

