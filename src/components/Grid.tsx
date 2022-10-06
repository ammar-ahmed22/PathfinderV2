import React, { useRef, useEffect, useContext } from "react";

import { HStack } from "@chakra-ui/react";
import Panel from "./Panel";
import Cell from "./Cell";

import { isCorner } from "../utils/board";

import { StoreContext } from "../Store";
import { StoreContextType } from "../@types/Store";

import Vec2 from "../helpers/Vec2";

const Grid : React.FC = () => {
  
  const gridRef = useRef<HTMLDivElement>();
  const store = useContext(StoreContext) as StoreContextType;


  // setting Target
  useEffect(() => {
    if (store.gridDim){
      const halfHeight = Math.floor((store.gridDim.y - 1 ) / 2);
      store.setStartIdx(new Vec2(5, halfHeight));
      store.setTargetIdx(new Vec2(store.gridDim.x - 5, halfHeight));
    }
  }, [store.gridDim])

  // Settings initial grid dimensions
  useEffect(() => {
    if (gridRef.current && store.cellSize){
      const { current } = gridRef;
      store.updateGridDimensions(current, store.cellSize);
    }
  }, [gridRef, store.cellSize])

  // Creating nodes after dimensions set
  useEffect(() => {
    if (store.gridDim && store.cellSize){
      store.createNodes();
    }
  }, [store.gridDim, store.cellSize])

  // Resize event
  useEffect(() => {
    const resizeEventListener = () => {
      if (gridRef.current && store.cellSize){
        const { current } = gridRef;
        store.updateGridDimensions(current, store.cellSize);
      }
    }

    window.addEventListener("resize", resizeEventListener)

    return () => window.removeEventListener("resize", resizeEventListener);
  }, [])

  return (
    <Panel 
      height="calc(100vh - var(--pf-space-5) * 2)" 
      width="75vw"
      customRef={gridRef}
      styles={{ display: "flex", justifyContent: "center", alignItems: "center", flexDirection: "column" }}
    >
      {
        store.nodes && !!store.nodes.length && store.nodes.map((row, rowIdx) => {
          return (
            <HStack key={rowIdx} spacing="0" >
              {
                row.map((node, colIdx) => {
                  return (
                    <Cell key={rowIdx + colIdx} node={node} corner={isCorner(node.index, store.nodes.length, row.length)}/>
                  )
                })
              }
            </HStack>
          )
        })
      }
    </Panel>
  )
}

export default Grid;