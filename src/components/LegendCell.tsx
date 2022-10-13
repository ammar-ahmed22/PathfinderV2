import React from "react";
import { VStack, Box, Text, useColorModeValue } from "@chakra-ui/react";

import type { LegendCellProps } from "../@types/components/LegendCell";

const LegendCell: React.FC<LegendCellProps> = ({
    size,
    bg,
    borderWidth,
    cellName,
    bgGradient,
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
            />
            <Text fontSize="sm">{cellName}</Text>
        </VStack>
    );
};

export default LegendCell;
