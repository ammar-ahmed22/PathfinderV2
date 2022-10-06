import React, { useContext } from "react";
import { Box, Icon, keyframes } from "@chakra-ui/react";
import type { Keyframes } from "@emotion/react";
import { motion } from "framer-motion";

import { StoreContext } from "../Store";
import { StoreContextType } from "../@types/Store";

import { FaMapMarkerAlt, FaCrosshairs } from "react-icons/fa";

import Vec2 from "../helpers/Vec2";

import { CellProps, CornerType } from "../@types/components/Cell";

const Cell : React.FC<CellProps> = ({ node, corner }) => {

  const borderSize = 1;
  
  const animBoxSize = node.size - (2 * borderSize);

  const borderRadii = (corner: CornerType | undefined) : object => {
    return {
      borderTopLeftRadius: corner === "tl" ? "xl" : "none",
      borderTopRightRadius: corner === "tr" ? "xl" : "none",
      borderBottomLeftRadius: corner === "bl" ? "xl" : "none",
      borderBottomRightRadius: corner === "br" ? "xl" : "none" 
    }
  }

  const getCSSVarColor = (chakraColor: string) => {
    const splitted = chakraColor.split(".");
    return `var(--pf-colors-${splitted.join("-")})`
  }

  const animations = {
    visited: keyframes`
      0% { background-color: ${getCSSVarColor("brand.blue.200")}; height: ${animBoxSize / 4}px; width: ${animBoxSize / 4}px; border-radius: 100%; }
      100% { background-color: ${getCSSVarColor("brand.purple.500")}; height: ${animBoxSize}px; width: ${animBoxSize}px; border-radius: 0; }
    `,
    path: keyframes`
      0% { background-color: ${getCSSVarColor("yellow.200")}; height: ${animBoxSize / 4}px; width: ${animBoxSize / 4}px; border-radius: 100%; }
      100% { background-color: ${getCSSVarColor("yellow.400")}; height: ${animBoxSize}px; width: ${animBoxSize}px; border-radius: 0; }
    `
  }

  const createAnimation = (keyframes: Keyframes, duration: number, easing: string ) => {
    return `${keyframes} ${duration}s ${easing}`
  }

  const styleProps = {
    height: node.size + "px",
    width: node.size + "px",
    borderStyle: "solid",
    borderWidth: `${borderSize}px`,
    borderColor: "gray.500",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    //backgroundColor: node.type === "start" ? "green" : node.type === "target" ? "red" : "none"
  }

  const store = useContext(StoreContext) as StoreContextType;

  const isStartTarget = node.type === "start" || node.type === "target";

  const handleDragStart = (ev : React.DragEvent<HTMLDivElement>) => {
    ev.dataTransfer.setData("draggedNode", JSON.stringify(node));
    ev.dataTransfer.dropEffect = "move";
  }

  const handleDragOver = (ev: React.DragEvent<HTMLDivElement>) => {
    ev.preventDefault();
    ev.dataTransfer.dropEffect = isStartTarget ? "none" : "move";
  }

  const handleDragDrop = (ev: React.DragEvent<HTMLDivElement>) => {
    ev.preventDefault();
    const data = ev.dataTransfer.getData("draggedNode")
    const draggedNode = JSON.parse(data);

    if (draggedNode.type === "start"){
      store.setStartIdx(node.index)
    } else {
      store.setTargetIdx(node.index)
    }
    
    const draggedNodeIndex = new Vec2(draggedNode.index.x, draggedNode.index.y);
    store.updateNodeByIndex(draggedNodeIndex, (prevNode) => {
      prevNode.type = "base";
      return prevNode;
    })
  }


  
  return (
    <Box 
      {...styleProps}
      {...borderRadii(corner)}
      draggable={isStartTarget}
      onDragStart={handleDragStart}
      onDragOver={handleDragOver}
      onDrop={handleDragDrop}
    >
      {
        isStartTarget && <Icon as={node.type === "start" ? FaMapMarkerAlt : FaCrosshairs}  />
      }
      {
        node.type === "visited" && (
          <Box 
            as={motion.div}
            animation={createAnimation(animations.visited, 1, "ease-in")}
            height={animBoxSize + "px"}
            width={animBoxSize + "px"}
            bg="brand.purple.500"
          />
        )
      }
      {
        node.type === "path" && (
          <Box 
            as={motion.div}
            animation={createAnimation(animations.path, 1, "ease-in")}
            height={animBoxSize + "px"}
            width={animBoxSize + "px"}
            bg="yellow.400"
          />
        )
      }
    </Box>
  )
}

export default Cell;