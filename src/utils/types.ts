import type { Algorithm } from "../@types/components/AlgorithmMenu";

export const isAlgorithm = (val: string) : val is Algorithm => {
  return ["astar", "djikstra"].includes(val)
}