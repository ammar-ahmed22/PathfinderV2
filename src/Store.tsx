import React, { createContext, useState, useEffect } from "react";
import { StoreContextType, StoreProviderProps } from "./@types/Store";
import Vec2 from "./helpers/Vec2";
import Node from "./helpers/Node";
import type { NodeType } from "./@types/helpers/Node";
import { AlgorithmParams, Generic } from "./@types/helpers/Node";
import type { Algorithm } from "./@types/helpers/algorithms";

export const StoreContext = createContext<StoreContextType | null>(null);

const StoreProvider: React.FC<StoreProviderProps> = ({ children }) => {
  const [cellSize, setCellSize] = useState<number | undefined>(30);
  const [gridDim, setGridDim] = useState<Vec2 | undefined>();
  const [nodes, setNodes] = useState<Node<AlgorithmParams>[][]>([]);
  const [startIdx, setStartIdx] = useState<Vec2 | undefined>();
  const [targetIdx, setTargetIdx] = useState<Vec2 | undefined>();
  const [isStarted, setIsStarted] = useState<boolean>(false);
  const [selectedAlgorithm, setSelectedAlgorithm] =
    useState<Algorithm>("astar");
  const [shiftPressed, setShiftPressed] = useState<boolean>(false);
  const [visualDelay, setVisualDelay] = useState<number>(10);
  const [status, setStatus] = useState<{
    started: boolean;
    finished: boolean;
  }>({
    started: false,
    finished: false,
  });
  const [output, setOutput] = useState<string[]>([]);
  const [startTime, setStartTime] = useState<number>(0);

  useEffect(() => {
    if (status.started && !status.finished) {
      setStartTime(performance.now());
    }

    if (status.started && status.finished && startTime !== 0) {
      const end = performance.now();
      const elapsed = end - startTime;
      const elapsedParsed =
        elapsed >= 1000
          ? (elapsed / 1000).toFixed(2) + "s"
          : elapsed.toFixed(2) + "ms";
      setOutput((prev) => [...prev, `Visualized in: ${elapsedParsed}`]);
      setStartTime(0);
    }
    // eslint-disable-next-line
  }, [status]);

  const createNodes = () => {
    if (!gridDim) {
      throw new Error("gridDim must be set to create nodes!");
    }

    if (!cellSize) {
      throw new Error("cellSize must be set to create nodes!");
    }

    const rows = gridDim.y;
    const cols = gridDim.x;

    const newNodes: Node<Generic>[][] = [];
    for (let row = 0; row < rows; row++) {
      const temp: Node<Generic>[] = [];
      for (let col = 0; col < cols; col++) {
        const node = new Node(new Vec2(col, row), cellSize, "base", false, {});
        temp.push(node);
      }

      newNodes.push(temp);
    }

    setNodes(newNodes);
  };

  const updateNodeByIndex = (
    index: Vec2,
    cb: (node: Node<AlgorithmParams>) => Node<AlgorithmParams>
  ) => {
    setNodes((prevNodes) => {
      const prevNode = prevNodes[index.y][index.x];
      const updatedNode = cb(prevNode);
      const copy = [...prevNodes];
      copy[index.y][index.x] = updatedNode;
      return copy;
    });
  };

  const updateNodeTypeByIndex = (index: Vec2, type: NodeType, unique: boolean = false) => {
    if (nodes && !!nodes.length) {
      setNodes((prevNodes) => {
        const copy = [...prevNodes];
        if (unique){
          copy.forEach((row, rIdx) => {
            row.forEach( (node, cIdx) => {
              if (node.type === type){
                copy[rIdx][cIdx].type = "base";
              }
            })
          })
        }
        if (copy[index.y][index.x]) copy[index.y][index.x].type = type;
        return copy;
      });
    }
  };

  const resetNodes = () => {
    setNodes((prevNodes) => {
      const copy = [...prevNodes];
      for (let row = 0; row < copy.length; row++) {
        for (let col = 0; col < copy[row].length; col++) {
          const node = copy[row][col];
          const { index } = node;
          copy[row][col].prev = undefined;
          if (
            startIdx &&
            targetIdx &&
            !index.equals(startIdx) &&
            !index.equals(targetIdx)
          ) {
            copy[row][col].type = "base";
            copy[row][col].obstacle = false;
            copy[row][col].bg = undefined;
          }
        }
      }
      return copy;
    });
  };

  const resetObstacles = () => {
    setNodes((prevNodes) => {
      const copy = [...prevNodes];
      for (let row = 0; row < copy.length; row++) {
        for (let col = 0; col < copy[row].length; col++) {
          const node = copy[row][col];
          const { index } = node;
          if (
            startIdx &&
            targetIdx &&
            !index.equals(startIdx) &&
            !index.equals(targetIdx) &&
            node.obstacle
          ) {
            copy[row][col].type = "base";
            copy[row][col].obstacle = false;
          }
        }
      }
      return copy;
    });
  };

  useEffect(() => {
    if (startIdx) updateNodeTypeByIndex(startIdx, "start", true);
    // eslint-disable-next-line
  }, [startIdx]);

  useEffect(() => {
    if (targetIdx) updateNodeTypeByIndex(targetIdx, "target", true);
    // eslint-disable-next-line
  }, [targetIdx]);

  const makeEven = (val: number) => val % 2 === 0 ? val - 1 : val;

  const state: StoreContextType = {
    cellSize,
    setCellSize: (val: number) => setCellSize(val),
    gridDim,
    setGridDim: (v: Vec2) => setGridDim(v),
    updateGridDimensions: (gridElem: HTMLDivElement, cellSize: number, maxArea: number = 700) => {
      const rect = gridElem.getBoundingClientRect();

      let height = Math.round(rect.height) - 2 * cellSize;
      let width = Math.round(rect.width) - 2 * cellSize;

      let tentativeRows = Math.floor(height / cellSize);
      let tentativeCols = Math.floor(width / cellSize);
      const rows = makeEven(tentativeRows);
      const cols = makeEven(tentativeCols);
      console.log({ height, width });
      console.log("area:", rows * cols);

      if ((rows * cols ) > maxArea){
        const newCellSize = Math.floor(Math.sqrt((height * width) / maxArea));
        // height = Math.round(rect.height) - 2 * cellSize;
        // width = Math.round(rect.width) - 2 * cellSize;
        tentativeRows = Math.floor(height / newCellSize);
        tentativeCols = Math.floor(width / newCellSize);
        console.log("too big, orig:", cellSize, "updated:", newCellSize, "updated area:", makeEven(tentativeCols) * makeEven(tentativeRows));
        setCellSize(newCellSize);
      }

      setGridDim(
        new Vec2(
          makeEven(tentativeCols),
          makeEven(tentativeRows)
        )
      );
    },
    nodes,
    setNodes: (nodes: Node<AlgorithmParams>[][]) => setNodes(nodes),
    createNodes,
    updateNodeByIndex,
    updateNodeTypeByIndex,
    startIdx,
    targetIdx,
    setStartIdx: (idx: Vec2) => {
      setStartIdx(idx);
      updateNodeTypeByIndex(idx, "start", true);
    },
    setTargetIdx: (idx: Vec2) => {
      setTargetIdx(idx);
      updateNodeTypeByIndex(idx, "target", true);
    },
    isStarted,
    setIsStarted: (val: boolean) => setIsStarted(val),
    selectedAlgorithm,
    setSelectedAlgorithm: (algo: Algorithm) => setSelectedAlgorithm(algo),
    shiftPressed,
    setShiftPressed: (val: boolean) => setShiftPressed(val),
    resetNodes,
    resetObstacles,
    visualDelay,
    setVisualDelay: (val: number) => setVisualDelay(val),
    status,
    setStarted: (val: boolean) =>
      setStatus((prevStatus) => ({ ...prevStatus, started: val })),
    setFinished: (val: boolean) =>
      setStatus((prevStatus) => ({ ...prevStatus, finished: val })),
    output,
    addOutput: (val: string) => setOutput((prev) => [...prev, val]),
    resetOutput: () => setOutput([]),
  };

  return (
    <StoreContext.Provider value={state}>{children}</StoreContext.Provider>
  );
};

export default StoreProvider;
