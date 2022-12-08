import React from "react";
import { VStack, Box, Text, useColorModeValue, Icon } from "@chakra-ui/react";

import type { LegendCellProps } from "../@types/components/LegendCell";

const LegendCell: React.FC<LegendCellProps> = ({
  size,
  bg,
  borderWidth,
  cellName,
  bgGradient,
  icon,
}) => {
  const borderColor = useColorModeValue("gray.600", "gray.400");

  return (
    <VStack align="center">
      <Box
        w={size}
        h={size}
        borderColor={borderColor}
        borderWidth={borderWidth ? borderWidth : "2px"}
        borderStyle="solid"
        bg={bg === "borderColor" ? borderColor : bg}
        bgGradient={bgGradient}
        display="flex"
        justifyContent="center"
        alignItems="center"
      >
        {icon && (
          <Icon
            as={icon}
            w={`calc(${size} / 2)`}
            h={`calc(${size} / 2)`}
            color={borderColor}
          />
        )}
      </Box>
      <Text fontSize="sm" fontWeight="semibold">
        {cellName}
      </Text>
    </VStack>
  );
};

export default LegendCell;
