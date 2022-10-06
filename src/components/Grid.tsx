import React, { useRef, useEffect, useContext, useState } from "react";

import { Box, HStack, useColorModeValue } from "@chakra-ui/react";
import Panel from "./Panel";
import Cell from "./Cell";

import { StoreContext } from "../Store";
import { StoreContextType } from "../@types/Store";

import Vec2 from "../helpers/Vec2";

import type { CornerType } from "../@types/components/Cell";

const Grid : React.FC = () => {
  
  const gridRef = useRef<HTMLDivElement>();
  const store = useContext(StoreContext) as StoreContextType;

  const [start, setStart] = useState<Vec2>(new Vec2());
  const [target, setTarget] = useState<Vec2>(new Vec2());

  // setting Target
  useEffect(() => {
    console.log("start target");
    if (store.gridDim){
      const halfHeight = Math.floor((store.gridDim.y - 1 ) / 2);
      store.setStartIdx(new Vec2(5, halfHeight));
      store.setTargetIdx(new Vec2(store.gridDim.x - 5, halfHeight));
    }
  }, [store.gridDim])

  // Settings initial grid dimensions
  useEffect(() => {
    console.log("intial griddim");
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

  // For corner border radii
  const isCorner = (index: Vec2, rows: number, cols: number) : CornerType | undefined => {
    const corners = {
      topLeft: new Vec2(),
      topRight: new Vec2(cols - 1, 0),
      bottomLeft: new Vec2(0, rows - 1),
      bottomRight: new Vec2(cols - 1, rows - 1)
    }

    if (index.equals(corners.topLeft)) return "tl";
    if (index.equals(corners.topRight)) return "tr";
    if (index.equals(corners.bottomLeft)) return "bl";
    if (index.equals(corners.bottomRight)) return "br";
    
    return undefined
  }

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