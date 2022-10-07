import * as React from "react";
import Vec2 from "../helpers/Vec2";
import Node from "../helpers/Node";
import type { NodeType } from "./helpers/Node";
import { AlgorithmParams } from "./helpers/Node";
import type { Algorithm } from "./helpers/algorithms";

export type StoreContextType = {
    cellSize: number | undefined;
    setCellSize: (val: number) => void;
    gridDim: Vec2 | undefined;
    setGridDim: (v: Vec2) => void;
    updateGridDimensions: (gridElem: HTMLDivElement, cellSize: number) => void;
    nodes: Node<AlgorithmParams>[][];
    setNodes: (nodes: Node<AlgorithmParams>[][]) => void;
    createNodes: () => void;
    updateNodeByIndex: (
        index: Vec2,
        cb: (node: Node<AlgorithmParams>) => Node<AlgorithmParams>
    ) => void;
    updateNodeTypeByIndex: (index: Vec2, type: NodeType) => void;
    startIdx: Vec2 | undefined;
    setStartIdx: (idx: Vec2) => void;
    targetIdx: Vec2 | undefined;
    setTargetIdx: (idx: Vec2) => void;
    isStarted: boolean;
    setIsStarted: (val: boolean) => void;
    selectedAlgorithm: Algorithm;
    setSelectedAlgorithm: (algo: Algorithm) => void;
};

export interface StoreProviderProps {
    children: React.ReactNode;
}
