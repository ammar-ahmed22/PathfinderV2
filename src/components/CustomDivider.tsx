import React from "react";
import { Box, BoxProps } from "@chakra-ui/react";

type CustomDividerProps = Omit<BoxProps, "bgGradient" | "bg" | "background">;

const CustomDivider : React.FC<CustomDividerProps> = (props) => {
  
  return (
    <Box 
      h="1px"
      w="100%"
      bgGradient="linear(to-r, transparent 0%, brand.purple.500 20%, brand.blue.500 80%, transparent 100%)"
      {...props}
    />
  )
}

export default CustomDivider;