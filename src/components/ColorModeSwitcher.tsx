import * as React from "react";
import {
    useColorMode,
    useColorModeValue,
    Switch,
    HStack,
    StackProps,
    Text,
} from "@chakra-ui/react";

type ColorModeSwitcherProps = Omit<StackProps, "aria-label">;

const ColorModeSwitcher: React.FC<ColorModeSwitcherProps> = (props) => {
    const { toggleColorMode, colorMode } = useColorMode();
    const text = useColorModeValue("dark", "light");

    return (
        <HStack {...props} aria-label={`Switch to ${text} mode`}>
            <Text>☀️</Text>
            <Switch
                size="sm"
                onChange={() => toggleColorMode()}
                isChecked={colorMode === "dark"}
            />
            <Text>🌙</Text>
        </HStack>
    );
};

export default ColorModeSwitcher;
