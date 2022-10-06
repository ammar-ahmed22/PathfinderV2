import React, { createContext, useState, useEffect } from "react";
import { StoreContextType, StoreProviderProps } from "./@types/Store";
import Vec2 from "./helpers/Vec2";
import Node from "./helpers/Node";
import { AlgorithmParams, Generic } from "./@types/helpers/Node";

export const StoreContext = createContext<StoreContextType | null>(null);

const StoreProvider : React.FC<StoreProviderProps> = ({ children }) => {

  const [cellSize, setCellSize] = useState<number | undefined>();
  const [gridDim, setGridDim] = useState<Vec2 | undefined>();
  const [nodes, setNodes] = useState<Node<AlgorithmParams>[][]>([]);
  const [startIdx, setStartIdx] = useState<Vec2 | undefined>()
  const [targetIdx, setTargetIdx] = useState<Vec2 | undefined>();
  // const [state, setState] = useState<StoreContextType>({ 
  //   cellSize: undefined
  // })

  const createNodes = () => {
    if (!gridDim){
      throw new Error("gridDim must be set to create nodes!")
    }

    if (!cellSize){
      throw new Error("cellSize must be set to create nodes!")
    }

    const rows = gridDim.y;
    const cols = gridDim.x;

    const newNodes : Node<Generic>[][] = [];
    for (let row = 0; row < rows; row++){
      const temp : Node<Generic>[] = [];
      for (let col = 0; col < cols; col++){
        const node = new Node(
          new Vec2(col, row),
          cellSize,
          "base",
          false,
          {}
        );
        temp.push(node);
      }

      newNodes.push(temp);
    }

    setNodes(newNodes);
  }

  const updateNodeByIndex = (index: Vec2, cb: (node: Node<AlgorithmParams>) => Node<AlgorithmParams>) => {
    setNodes( prevNodes => {
      const prevNode = prevNodes[index.y][index.x];
      const updatedNode = cb(prevNode);
      const copy = [...prevNodes];
      copy[index.y][index.x] = updatedNode;
      return copy; 
    })
  }

  const updateStartTargetNodes = (idx: Vec2, type: "start" | "target") => {
    if (nodes && !!nodes.length){
      updateNodeByIndex(idx, (prevNode) => {
        prevNode.type = type;
        return prevNode
      })
    }
  }

  useEffect(() => {
    if (startIdx) updateStartTargetNodes(startIdx, "start")
  }, [startIdx])

  useEffect(() => {
    if (targetIdx) updateStartTargetNodes(targetIdx, "target")
  }, [targetIdx])

  const state : StoreContextType = {
    cellSize,
    setCellSize: (val: number) => setCellSize(val),
    gridDim,
    setGridDim: (v: Vec2) => setGridDim(v),
    updateGridDimensions: (gridElem: HTMLDivElement, cellSize: number) => {
      const rect = gridElem.getBoundingClientRect();

      const height = Math.round(rect.height) - (2 * cellSize);
      const width = Math.round(rect.width) - (2 * cellSize);

      setGridDim(new Vec2(
        Math.floor( width / cellSize ),
        Math.floor( height / cellSize )
      ));

    },
    nodes,
    setNodes: (nodes: Node<AlgorithmParams>[][]) => setNodes(nodes),
    createNodes,
    updateNodeByIndex,
    startIdx,
    targetIdx,
    setStartIdx: (idx: Vec2) => {
      setStartIdx(idx)
      updateStartTargetNodes(idx, "start");
    },
    setTargetIdx: (idx: Vec2) => {
      setTargetIdx(idx)
      updateStartTargetNodes(idx, "target")
    }
  }

  return (
    <StoreContext.Provider value={state}>
      {
        children
      }
    </StoreContext.Provider>
  )
}

export default StoreProvider;