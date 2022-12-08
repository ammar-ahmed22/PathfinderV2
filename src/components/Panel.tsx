import * as React from "react";
import {
    Box,
    useColorModeValue,
    Accordion,
    AccordionButton,
    AccordionIcon,
    AccordionPanel,
    AccordionItem,
    HStack,
    Icon,
    Heading,
} from "@chakra-ui/react";
import { PanelProps } from "../@types/components/Panel";

const Panel: React.FC<PanelProps> = ({
    children,
    width,
    bg,
    height,
    styles,
    customRef,
    accordion,
    heading,
    headingIcon,
    accordionDefaultOpen,
}) => {
    const defaultBg = useColorModeValue("white", "gray.700");
    const color = useColorModeValue("gray.800", "white");

    if (accordion) {
        return (
            <Accordion
                allowToggle
                defaultIndex={accordionDefaultOpen ? [0] : undefined}
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
                <AccordionItem border="none">
                    <AccordionButton
                        p="0"
                        _hover={{
                            cursor: "pointer",
                        }}
                    >
                        <HStack flex="1">
                            <Heading as="h2" size="md" variant="gradient">
                                {heading ?? "ERR"}
                            </Heading>
                            <Icon as={headingIcon} color="brand.blue.500" />
                        </HStack>
                        <AccordionIcon color="brand.blue.500" />
                    </AccordionButton>

                    <AccordionPanel p="0">{children}</AccordionPanel>
                </AccordionItem>
            </Accordion>
        );
    }

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
            {heading && (
                <HStack>
                    <Heading as="h2" size="md" variant="gradient">
                        {heading ?? "ERR"}
                    </Heading>
                    <Icon as={headingIcon} color="brand.blue.500" />
                </HStack>
            )}
            {children}
        </Box>
    );
};

export default Panel;
