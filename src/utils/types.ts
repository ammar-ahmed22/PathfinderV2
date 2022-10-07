import type { Algorithm } from "../@types/helpers/algorithms";

export const isAlgorithm = (val: string): val is Algorithm => {
    return ["astar", "djikstra"].includes(val);
};
