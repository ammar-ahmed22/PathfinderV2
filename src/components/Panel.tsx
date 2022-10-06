import * as React from "react";
import { 
  Box,
  useColorModeValue 
} from "@chakra-ui/react";
import { PanelProps } from "../@types/components/Panel";

const Panel : React.FC<PanelProps> = ({ children, width, bg, height, styles, customRef }) => {

  const defaultBg = useColorModeValue("white", "gray.700");
  const color = useColorModeValue("gray.800", "white");

  return (
    <Box 
      w={width ? width : "100%"} 
      h={height ? height : "auto"}
      bg={bg ? bg : defaultBg}
      padding="5" 
      color={color} 
      shadow="panel"
      borderRadius="xl"
      ref={customRef}
      {...styles} 
    >
      {
        children
      }
    </Box>
  )
}

export default Panel;